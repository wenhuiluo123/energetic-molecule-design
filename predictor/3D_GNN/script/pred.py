import load_dataset
import torch
from torch import nn, optim
import argparse
import sys

sys.path.append('.')
import arg_utils as qm9_utils
import os
from mgp import layers
import yaml
from easydict import EasyDict
from collections import OrderedDict
import random
import numpy as np
import pickle as pkl
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_sparse import coalesce
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

all_tasks = ['alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv', 'G', 'H', 'r2', 'U', 'U0', 'zpve']

parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--config_path', type=str, default='predictor/3D_GNN/config/finetune_qm9.yml', metavar='N',
                    help='Path of config yaml.')
parser.add_argument('--property', type=str, default='', metavar='N',
                    help='Property to predict.')
parser.add_argument('--model_name', type=str, default='', metavar='N',
                    help='Model name.')
parser.add_argument('--restore_path', type=str, default='', metavar='N',
                    help='Restore path.')
args = parser.parse_args()

device = torch.device("cuda")
dtype = torch.float32

with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)

if args.property != '':
    config.train.property = args.property

if args.model_name != '':
    config.model.name = args.model_name

if args.restore_path != '':
    config.train.restore_path = args.restore_path

os.makedirs(config.train.save_path + "/" + config.model.name + "/" + config.train.property, exist_ok=True)

# fix seed
seed = config.train.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('fix seed:', seed)


def mdn_loss_fn(pi, sigma, mu, y):
    """
    Compute the loss and predicted value for a Mixture Density Network.

    Parameters:
    - pi: Tensor of shape (batch_size, num_components) representing the mixture coefficients.
    - sigma: Tensor of shape (batch_size, num_components) representing the standard deviations of the Gaussian components.
    - mu: Tensor of shape (batch_size, num_components) representing the means of the Gaussian components.
    - y: Tensor of shape (batch_size, 1) representing the target values.

    Returns:
    - loss: The negative log-likelihood loss for the MDN.
    - predicted_value: Tensor of shape (batch_size, 1) representing the expected value.
    """
    distribution = torch.distributions.normal.Normal(mu, sigma)

    likelihood = distribution.log_prob(y)

    loss = -torch.mean(likelihood)

    # Compute the predicted value (expected value)
    predicted_value = torch.sum(pi * mu, dim=1, keepdim=True)

    return loss, predicted_value


def load_model(model, model_path):
    state = torch.load(model_path, map_location=device)
    new_dict = OrderedDict()
    for k, v in state['model'].items():
        if k.startswith('module.model.'):
            new_dict[k[13:]] = v
        if k.startswith('model.'):
            new_dict[k[6:]] = v
        # if k.startswith('module.node_dec.'):
        #     new_dict[k[7:]] = v
    model.load_state_dict(new_dict, strict=False)
    return new_dict


dataloaders, charge_scale = load_dataset.retrieve_dataloaders(config.data.base_path, config.train.batch_size, config.train.num_workers)
# compute mean and mean absolute deviation
meann, mad = qm9_utils.compute_mean_mad(dataloaders, config.train.property)

model = layers.EGNN_finetune_mdn(in_node_nf=config.model.max_atom_type * (config.model.charge_power + 1),
                                  in_edge_nf=0 if config.model.no_edge_types else 4, hidden_nf=config.model.hidden_dim,
                                  n_layers=config.model.n_layers,
                                  attention=config.model.attention, use_layer_norm=config.model.layernorm).to(device)


path = config.train.save_path + "/" + config.model.name + "/" + config.train.property
checkpoint_path = path + "/pred_checkpoint_best.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint['model'])
model.to(device)

optimizer = optim.Adam(
    [param for name, param in model.named_parameters()],
    lr=config.train.lr,
    weight_decay=float(config.train.weight_decay)
)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.train.epochs,
    eta_min=float(config.train.min_lr)
)


loss_l1 = nn.L1Loss()


def process_input(atom_type, max_atom_type=100, charge_power=2):
    one_hot = nn.functional.one_hot(atom_type, max_atom_type)
    charge_tensor = (atom_type.unsqueeze(-1) / max_atom_type).pow(
        torch.arange(charge_power + 1., dtype=torch.float32).to(atom_type))
    charge_tensor = charge_tensor.view(atom_type.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(atom_type.shape + (-1,))
    return atom_scalars


def binarize(x):
    return torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))


def get_higher_order_adj_matrix(adj, order):
    """
    Args:
        adj:        (N, N)
        type_mat:   (N, N)
    """

    adj_mats = [torch.eye(adj.size(0), device=adj.device), \
                binarize(adj + torch.eye(adj.size(0), device=adj.device))]
    for i in range(2, order + 1):
        adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
    order_mat = torch.zeros_like(adj).float()

    for i in range(1, order + 1):
        order_mat += (adj_mats[i] - adj_mats[i - 1]) * i
    return order_mat.long()


def add_high_order_edges(adj_mat):
    adj_order = get_higher_order_adj_matrix(adj_mat, 3)
    type_highorder = torch.where(adj_order > 1, adj_order, torch.zeros_like(adj_order))
    type_mat = adj_mat
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder
    edge_index, edge_type = dense_to_sparse(type_new)
    return edge_index, edge_type


def gen_adj_matrix(pos, mask):
    # pos : batch * n_nodes * 3
    batch, nodes = mask.shape
    batch_adj = torch.norm(pos.unsqueeze(1) - pos.unsqueeze(2), p=2, dim=-1)  # batch * n * n
    batch_mask = mask[:, :, None] * mask[:, None, :]  # batch * n * n
    batch_mask = batch_mask.bool() & (batch_adj <= 1.6) & (~torch.eye(nodes).to(mask).bool())  # batch * n * n
    return torch.block_diag(*batch_mask)


def gen_fully_connected(pos, mask):
    batch, nodes = mask.shape
    batch_mask = mask[:, :, None] * mask[:, None, :]
    batch_mask = batch_mask.bool() & (~torch.eye(nodes).to(mask).bool())
    batch_mask = torch.block_diag(*batch_mask)
    edge_index, edge_type = dense_to_sparse(batch_mask)
    return edge_index, edge_type

def gen_fully_connected_with_hop(pos, mask):
    batch, nodes = mask.shape
    batch_adj = torch.norm(pos.unsqueeze(1) - pos.unsqueeze(2), p=2, dim=-1)  # batch * n * n
    batch_mask_fc = mask[:, :, None] * mask[:, None, :]  # batch * n * n
    # 1.6 is an empirically reasonable cutoff to distinguish the existence of bonds for stable small molecules
    batch_mask = batch_mask_fc.bool() & (batch_adj <= 1.6) & (~torch.eye(nodes).to(mask).bool())  # batch * n * n
    batch_mask = torch.block_diag(*batch_mask)
    adj_order = get_higher_order_adj_matrix(batch_mask, 3)
    type_highorder = torch.where(adj_order > 1, adj_order, torch.zeros_like(adj_order))
    fc_mask = batch_mask_fc.bool() & (~torch.eye(nodes).to(mask).bool())
    fc_mask = torch.block_diag(*fc_mask)
    type_new = batch_mask + type_highorder + fc_mask
    edge_index, edge_type = dense_to_sparse(type_new)
    return edge_index, edge_type - 1


def pred(loader, config):
    predictions = []
    uncertainties = []
    indices = []

    for i, data in enumerate(loader):

        model.eval()

        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size, n_nodes, -1).to(device, dtype)
        atom_mask = data['atom_mask'].view(batch_size, n_nodes).to(device)
        edge_mask = data['edge_mask'].to(device)
        charges = data['charges'].to(device, torch.long)
        charges = charges.view(batch_size * n_nodes, -1)

        nodes = process_input(
            charges,
            max_atom_type=config.model.max_atom_type,
            charge_power=config.model.charge_power
        )
        nodes = nodes.view(batch_size * n_nodes, -1)

        if config.model.no_edge_types:
            edges, edge_types = gen_fully_connected(atom_positions, atom_mask)
            edge_attr = None
        else:
            edges, edge_types = gen_fully_connected_with_hop(atom_positions, atom_mask)
            edge_attr = nn.functional.one_hot(edge_types, 4)

        index = data[config.train.property].to(device, dtype)
        atom_positions = atom_positions.view(batch_size * n_nodes, -1)
        atom_mask = atom_mask.view(batch_size * n_nodes, -1)

        pi, sigma, mu = model(
            h=nodes,
            x=atom_positions,
            edges=edges,
            edge_attr=edge_attr,
            node_mask=atom_mask,
            n_nodes=n_nodes,
            adapter=config.train.property
        )

        mu_sq = mu ** 2
        sigma_sq = sigma ** 2
        pred_var = torch.sum(pi * (sigma_sq + mu_sq), dim=1) - torch.sum(pi * mu, dim=1) ** 2
        pred_mean = torch.sum(pi * mu, dim=1)
        pred_std = torch.sqrt(pred_var + 1e-6)

        # 反归一化
        pred_mean = pred_mean * mad + meann
        pred_std = pred_std * mad
        predict = pred_mean.detach().cpu().numpy()
        uncertainty = pred_std.detach().cpu().numpy()
        index = index.detach().cpu().numpy()

        predictions.extend(predict)
        uncertainties.extend(uncertainty)
        indices.extend(index)

    df = pd.DataFrame({
        'index': indices,
        'Prediction': predictions,
    })

    print("[DEBUG] prediction:", predictions)
    print("[DEBUG] index:", indices)
    print("[DEBUG] index:", uncertainties)
    df.to_csv('mdn_test_predictions_with_uncertainty.csv', index=False)
    print("[INFO] Prediction and uncertainty saved to mdn_test_predictions_with_uncertainty.csv")

    return 0


if __name__ == "__main__":
    res = {'epochs': [], 'losess': [], 'best_val': 0, 'best_test': 0, 'best_epoch': 0}
    meann = torch.tensor(1288.7224, dtype=torch.float64)
    mad = torch.tensor(195.8859, dtype=torch.float64)

    test_param = {'rmse': [], 'r2': [], 'mae': []}
    all_test_loss = []
    print(f"[DEBUG] Testing samples: {len(dataloaders['test'].dataset)}")

    pred(dataloaders['test'], config)





