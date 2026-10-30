import argparse
import csv
import os
import pickle as pkl
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse

sys.path.append(".")

from collate import collate_fn
from dataset import ProcessedDataset
from mgp import layers


parser = argparse.ArgumentParser(description="Non-MDN property finetuning on the 778 energetic dataset.")
parser.add_argument("--config_path", type=str, required=True, help="Path to yaml config.")
parser.add_argument("--property", type=str, default="", help="Override target property.")
parser.add_argument("--model_name", type=str, default="", help="Override model name.")
parser.add_argument("--restore_path", type=str, default="", help="Override pretrained checkpoint path.")
args = parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_npz(base_path: str, split: str) -> dict[str, torch.Tensor]:
    path = os.path.join(base_path, f"{split}.npz")
    with np.load(path) as f:
        return {key: torch.from_numpy(val) for key, val in f.items()}


def get_all_species(datasets: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    all_species = torch.cat([dataset["charges"].unique() for dataset in datasets.values()]).unique(sorted=True)
    if all_species[0] == 0:
        all_species = all_species[1:]
    return all_species


def retrieve_dataloaders(base_path: str, batch_size: int, num_workers: int) -> tuple[dict[str, DataLoader], torch.Tensor]:
    raw_splits = {split: load_split_npz(base_path, split) for split in ["train", "valid", "test"]}
    included_species = get_all_species(raw_splits)
    datasets = {
        split: ProcessedDataset(raw_splits[split], included_species=included_species, shuffle=(split == "train"))
        for split in ["train", "valid", "test"]
    }
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=(split == "train"),
        )
        for split in ["train", "valid", "test"]
    }
    return dataloaders, datasets["train"].max_charge


def compute_mean_mad(train_loader: DataLoader, property_name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for batch in train_loader:
        values.append(batch[property_name].to(device, torch.float32))
    values = torch.cat(values, dim=0)
    mean = values.mean()
    mad = (values - mean).abs().mean()
    if mad.abs().item() < 1e-12:
        mad = torch.tensor(1.0, device=device)
    return mean, mad


def load_pretrained_encoder(model: torch.nn.Module, model_path: str, device: torch.device) -> None:
    state = torch.load(model_path, map_location=device)
    new_dict = OrderedDict()
    for k, v in state["model"].items():
        if k.startswith("module.model."):
            new_dict[k[13:]] = v
        elif k.startswith("model."):
            new_dict[k[6:]] = v
    model.load_state_dict(new_dict, strict=False)


def process_input(atom_type: torch.Tensor, max_atom_type: int = 100, charge_power: int = 2) -> torch.Tensor:
    one_hot = nn.functional.one_hot(atom_type, max_atom_type)
    charge_tensor = (atom_type.unsqueeze(-1) / max_atom_type).pow(
        torch.arange(charge_power + 1.0, dtype=torch.float32, device=atom_type.device)
    )
    charge_tensor = charge_tensor.view(atom_type.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(atom_type.shape + (-1,))
    return atom_scalars


def binarize(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))


def get_higher_order_adj_matrix(adj: torch.Tensor, order: int) -> torch.Tensor:
    adj_mats = [
        torch.eye(adj.size(0), device=adj.device),
        binarize(adj + torch.eye(adj.size(0), device=adj.device)),
    ]
    for i in range(2, order + 1):
        adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
    order_mat = torch.zeros_like(adj).float()
    for i in range(1, order + 1):
        order_mat += (adj_mats[i] - adj_mats[i - 1]) * i
    return order_mat.long()


def gen_fully_connected(pos: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, nodes = mask.shape
    batch_mask = mask[:, :, None] * mask[:, None, :]
    batch_mask = batch_mask.bool() & (~torch.eye(nodes, device=mask.device).bool())
    batch_mask = torch.block_diag(*batch_mask)
    edge_index, edge_type = dense_to_sparse(batch_mask)
    return edge_index, edge_type


def gen_fully_connected_with_hop(pos: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, nodes = mask.shape
    batch_adj = torch.norm(pos.unsqueeze(1) - pos.unsqueeze(2), p=2, dim=-1)
    batch_mask_fc = mask[:, :, None] * mask[:, None, :]
    batch_mask = batch_mask_fc.bool() & (batch_adj <= 1.6) & (~torch.eye(nodes, device=mask.device).bool())
    batch_mask = torch.block_diag(*batch_mask)
    adj_order = get_higher_order_adj_matrix(batch_mask, 3)
    type_highorder = torch.where(adj_order > 1, adj_order, torch.zeros_like(adj_order))
    fc_mask = batch_mask_fc.bool() & (~torch.eye(nodes, device=mask.device).bool())
    fc_mask = torch.block_diag(*fc_mask)
    type_new = batch_mask + type_highorder + fc_mask
    edge_index, edge_type = dense_to_sparse(type_new)
    return edge_index, edge_type - 1


def regression_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    residual = preds - labels
    mse = torch.mean(residual.pow(2))
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(residual))
    denom = torch.sum((labels - labels.mean()).pow(2))
    if denom.item() <= 0:
        r2 = torch.tensor(0.0, device=labels.device)
    else:
        r2 = 1.0 - torch.sum(residual.pow(2)) / denom
    return {
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
        "r2": float(r2.item()),
    }


def main() -> None:
    config = load_yaml(args.config_path)
    if args.property:
        config["train"]["property"] = args.property
    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.restore_path:
        config["train"]["restore_path"] = args.restore_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    seed = int(config["train"]["seed"])
    set_seed(seed)
    print("fix seed:", seed)

    batch_size = int(config["train"].get("batch_size", 4))
    num_workers = int(config["train"].get("num_workers", 4))
    dataloaders, charge_scale = retrieve_dataloaders(config["data"]["base_path"], batch_size, num_workers)
    mean, mad = compute_mean_mad(dataloaders["train"], config["train"]["property"], device)

    model = layers.EGNN_finetune_last(
        in_node_nf=config["model"]["max_atom_type"] * (config["model"]["charge_power"] + 1),
        in_edge_nf=0 if config["model"]["no_edge_types"] else 4,
        hidden_nf=config["model"]["hidden_dim"],
        n_layers=config["model"]["n_layers"],
        attention=config["model"]["attention"],
        use_layer_norm=config["model"]["layernorm"],
    ).to(device)

    restore_path = config["train"].get("restore_path", "")
    if restore_path:
        load_pretrained_encoder(model, restore_path, device)
        print("load model from", restore_path)

    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=float(config["train"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        int(config["train"]["epochs"]),
        eta_min=float(config["train"]["min_lr"]),
    )
    loss_l1 = nn.L1Loss()

    save_root = os.path.join(
        config["train"]["save_path"],
        config["model"]["name"],
        config["train"]["property"],
    )
    os.makedirs(save_root, exist_ok=True)

    history_path = os.path.join(save_root, "metrics.csv")
    loss_path = os.path.join(save_root, "loss.pkl")
    best_ckpt_path = os.path.join(save_root, "checkpoint_best.pth")

    history_rows = []
    all_train_loss, all_val_loss, all_test_loss = [], [], []
    best_val_mae = float("inf")
    best_epoch = -1

    def run_epoch(epoch: int, loader: DataLoader, partition: str) -> dict[str, float]:
        is_train = partition == "train"
        model.train() if is_train else model.eval()
        total_loss = 0.0
        total_count = 0
        preds_all = []
        labels_all = []
        running = []

        for i, data in enumerate(loader):
            if is_train:
                optimizer.zero_grad()

            batch_size_local, n_nodes, _ = data["positions"].size()
            atom_positions = data["positions"].view(batch_size_local, n_nodes, -1).to(device, dtype)
            atom_mask = data["atom_mask"].view(batch_size_local, n_nodes).to(device)
            charges = data["charges"].to(device, torch.long).view(batch_size_local * n_nodes, -1)
            nodes = process_input(
                charges,
                max_atom_type=config["model"]["max_atom_type"],
                charge_power=config["model"]["charge_power"],
            ).view(batch_size_local * n_nodes, -1)

            if config["model"]["no_edge_types"]:
                edges, edge_types = gen_fully_connected(atom_positions, atom_mask)
                edge_attr = None
            else:
                edges, edge_types = gen_fully_connected_with_hop(atom_positions, atom_mask)
                edge_attr = nn.functional.one_hot(edge_types, 4)

            label = data[config["train"]["property"]].to(device, dtype)
            atom_positions_flat = atom_positions.view(batch_size_local * n_nodes, -1)
            atom_mask_flat = atom_mask.view(batch_size_local * n_nodes, -1)

            with torch.set_grad_enabled(is_train):
                pred_norm = model(
                    h=nodes,
                    x=atom_positions_flat,
                    edges=edges,
                    edge_attr=edge_attr,
                    node_mask=atom_mask_flat,
                    n_nodes=n_nodes,
                    adapter=config["train"]["property"],
                )
                if is_train:
                    loss = loss_l1(pred_norm, (label - mean) / mad)
                    loss.backward()
                    optimizer.step()
                    pred = mad * pred_norm.detach() + mean
                    loss_to_log = float(loss.item())
                else:
                    pred = mad * pred_norm + mean
                    loss = loss_l1(pred, label)
                    loss_to_log = float(loss.item())

            preds_all.append(pred.detach())
            labels_all.append(label.detach())
            total_loss += loss_to_log * batch_size_local
            total_count += batch_size_local
            running.append(loss_to_log)

            if i % int(config["train"]["log_interval"]) == 0:
                prefix = "" if is_train else f">> {partition}\t"
                print(
                    prefix
                    + "Epoch: {:4d} | Iter: {:4d} | loss: {:.4f} | lr: {:.5f}".format(
                        epoch,
                        i,
                        sum(running[-10:]) / len(running[-10:]),
                        optimizer.state_dict()["param_groups"][0]["lr"],
                    )
                )

        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        metrics = regression_metrics(preds_all, labels_all)
        metrics["loss"] = total_loss / max(total_count, 1)
        return metrics

    for epoch in range(int(config["train"]["epochs"])):
        train_metrics = run_epoch(epoch, dataloaders["train"], "train")
        all_train_loss.append(train_metrics["loss"])
        scheduler.step()

        if epoch % int(config["test"]["test_interval"]) == 0:
            val_metrics = run_epoch(epoch, dataloaders["valid"], "valid")
            test_metrics = run_epoch(epoch, dataloaders["test"], "test")

            all_val_loss.append(val_metrics["loss"])
            all_test_loss.append(test_metrics["loss"])

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_rmse": train_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "train_r2": train_metrics["r2"],
                    "val_loss": val_metrics["loss"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "val_r2": val_metrics["r2"],
                    "test_loss": test_metrics["loss"],
                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_r2": test_metrics["r2"],
                }
            )

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best_epoch = epoch
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_mae": best_val_mae,
                    "property": config["train"]["property"],
                }
                torch.save(state, best_ckpt_path)

            print(
                "Val: loss {:.4f} | rmse {:.4f} | mae {:.4f} | r2 {:.4f}".format(
                    val_metrics["loss"], val_metrics["rmse"], val_metrics["mae"], val_metrics["r2"]
                )
            )
            print(
                "Test: loss {:.4f} | rmse {:.4f} | mae {:.4f} | r2 {:.4f}".format(
                    test_metrics["loss"], test_metrics["rmse"], test_metrics["mae"], test_metrics["r2"]
                )
            )
            print("Best checkpoint by val_mae: {:.4f} at epoch {}".format(best_val_mae, best_epoch))

            with open(history_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
                writer.writeheader()
                writer.writerows(history_rows)

        with open(loss_path, "wb") as f:
            pkl.dump((all_train_loss, all_val_loss, all_test_loss), f)


if __name__ == "__main__":
    main()
