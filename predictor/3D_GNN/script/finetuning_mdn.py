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
import torch.nn.functional as F

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
    #pi = F.softmax(pi, dim=1)
    distribution = torch.distributions.normal.Normal(mu, sigma)
    likelihood = distribution.log_prob(y)
    weight_likelihood = torch.log(pi + 1e-8) + likelihood
    log_likelihood = torch.logsumexp(weight_likelihood, dim=1)
    loss = -torch.mean(log_likelihood)

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
print('meann:', meann)
print('mad:', mad)

model = layers.EGNN_finetune_mdn(in_node_nf=config.model.max_atom_type * (config.model.charge_power + 1),
                                  in_edge_nf=0 if config.model.no_edge_types else 4, hidden_nf=config.model.hidden_dim,
                                  n_layers=config.model.n_layers,
                                  attention=config.model.attention, use_layer_norm=config.model.layernorm).to(device)

print(model)
print(sum(p.numel() for p in model.parameters()))

if config.train.restore_path:
    encoder_param = load_model(model, config.train.restore_path)
    print('load model from', config.train.restore_path)

optimizer = optim.Adam([param for name, param in model.named_parameters()], lr=config.train.lr,
                       weight_decay=float(config.train.weight_decay))
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.train.epochs,
                                                          eta_min=float(config.train.min_lr))
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


def train(epoch, loader, config, partition='train', loss_type='mdn'):
    res = {'loss': 0, 'counter': 0, 'loss_arr': []}
    train_npred = []
    train_ntrue = []
    valid_npred = []
    valid_ntrue = []
    test_npred = []
    test_ntrue = []
    predictions = []
    labels = []
    uncertainties = []

    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size, n_nodes, -1).to(device, dtype)
        atom_mask = data['atom_mask'].view(batch_size, n_nodes).to(device)
        edge_mask = data['edge_mask'].to(device)
        charges = data['charges'].to(device, torch.long)
        charges = charges.view(batch_size * n_nodes, -1)
        nodes = process_input(charges, max_atom_type=config.model.max_atom_type, charge_power=config.model.charge_power)
        nodes = nodes.view(batch_size * n_nodes, -1)

        if config.model.no_edge_types:
            edges, edge_types = gen_fully_connected(atom_positions, atom_mask)
            edge_attr = None
        else:
            edges, edge_types = gen_fully_connected_with_hop(atom_positions, atom_mask)
            edge_attr = nn.functional.one_hot(edge_types, 4)
        label = data[config.train.property].to(device, dtype)
        atom_positions = atom_positions.view(batch_size * n_nodes, -1)
        atom_mask = atom_mask.view(batch_size * n_nodes, -1)

        pi, sigma, mu = model(h=nodes, x=atom_positions, edges=edges, edge_attr=edge_attr, node_mask=atom_mask, n_nodes=n_nodes,
                     adapter=config.train.property)

        if loss_type=='mdn':
            loss, pred = mdn_loss_fn(pi, sigma, mu, (label.unsqueeze(-1) - meann) / mad)#(label.unsqueeze(-1) - meann) / mad归一化，meann是训练集的均值，mad是训练值的中位数绝对偏差
            loss = loss.mean()
        else:
            loss = loss_l1(mu.squeeze(-1), (label - meann) / mad)
            pred = mu

        if partition == 'train':
            # loss = loss_l1(pred, (label - meann) / mad)
            loss.backward()
            optimizer.step()
            train_npred.append(pred.detach().cpu().numpy())
            train_ntrue.append(label.cpu().numpy())


        elif partition == 'valid':
            # loss = loss_l1(pred, (label - meann) / mad)
            valid_npred.append(pred.detach().cpu().numpy())
            valid_ntrue.append(label.cpu().numpy())

        else: #test
            # loss = loss_l1(pred, (label - meann) / mad)
            test_npred.append(pred.detach().cpu().numpy())
            test_ntrue.append(label.cpu().numpy())

            #pi = F.softmax(pi, dim=1)
            mu_sq = mu ** 2
            sigma_sq = sigma ** 2
            pred_var = torch.sum(pi * (sigma_sq + mu_sq), dim=1) - torch.sum(pi * mu, dim=1) ** 2#计算混合高斯分布的总方差，方差包含偶然不确定度和认知不确定度
            pred_std = torch.sqrt(pred_var + 1e-6)#标准差
            pred_std = pred_std * mad
            uncertainty = pred_std.detach().cpu().numpy()
            predict = mad * pred + meann
            predict = predict.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            predictions.extend(predict)
            labels.extend(label)
            uncertainties.extend(uncertainty)


        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

        

    if partition == 'train':
        train_true = np.concatenate(np.array(train_ntrue), 0)
        train_pred = np.concatenate(np.array(train_npred), 0)
        train_rmse = np.sqrt(mean_squared_error(mad * train_pred + meann, train_true))
        train_mae = mean_absolute_error(mad * train_pred + meann, train_true)
        r2_train = r2_score(train_true, mad * train_pred + meann)
        rmse = train_rmse
        r2 = r2_train
        mae = train_mae

    elif partition == 'valid':
        valid_true = np.concatenate(np.array(valid_ntrue), 0)
        valid_pred = np.concatenate(np.array(valid_npred), 0)
        valid_rmse = np.sqrt(mean_squared_error(mad * valid_pred + meann, valid_true))
        valid_mae = mean_absolute_error(mad * valid_pred + meann, valid_true)
        r2_valid = r2_score(valid_true, mad * valid_pred + meann)
        rmse = valid_rmse
        r2 = r2_valid
        mae = valid_mae

    else: #test
        test_true = np.concatenate(np.array(test_ntrue), 0)
        test_pred = np.concatenate(np.array(test_npred), 0)
        test_rmse = np.sqrt(mean_squared_error(mad * test_pred + meann, test_true))
        test_mae = mean_absolute_error(mad * test_pred + meann, test_true)
        r2_test = r2_score(test_true, mad * test_pred + meann)
        rmse = test_rmse
        r2 = r2_test
        mae = test_mae

        predictions = np.array(predictions).reshape(-1)
        labels = np.array(labels).reshape(-1)
        uncertainties = np.array(uncertainties).reshape(-1)

    return res['loss'] / res['counter'], rmse, mae, r2, predictions, labels, uncertainties

if __name__ == "__main__":
    res = {'epochs': [], 'losess': [], 'best_val': 0, 'best_test': -5, 'best_epoch': 0}
    test_predictions = None
    test_labels = None
    test_uncertainty = None

    train_param = {'rmse': [], 'r2': [], 'mae': []}
    valid_param = {'rmse': [], 'r2': [], 'mae': []}
    test_param = {'rmse': [], 'r2': [], 'mae': []}

    all_train_loss, all_val_loss, all_test_loss = [], [], []
    path = config.train.save_path + "/" + config.model.name + "/" + config.train.property
    os.makedirs(path, exist_ok=True)
    for epoch in range(0, config.train.epochs):
        train_loss, train_rmse, train_mae, train_r2,_,_,_ = train(epoch, dataloaders['train'], config, partition='train')

        print('train_rmse:', train_rmse, 'train_mae:', train_mae, 'r2_train:', train_r2, 'epoch:', epoch)

        train_param['rmse'].append(train_rmse)
        train_param['r2'].append(train_r2)
        train_param['mae'].append(train_mae)

        all_train_loss.append(train_loss)
        lr_scheduler.step()
        if epoch % config.test.test_interval == 0:
            val_loss, val_rmse, val_mae, val_r2,_,_,_ = train(epoch, dataloaders['valid'], config, partition='valid')

            print('val_rmse:', val_rmse, 'val_mae:', val_mae, 'r2_val:', val_r2)

            valid_param['rmse'].append(val_rmse)
            valid_param['r2'].append(val_r2)
            valid_param['mae'].append(val_mae)

            test_loss, test_rmse, test_mae, test_r2,test_predictions, test_labels, test_uncertainties = train(epoch, dataloaders['test'], config, partition='test')

            print('test_rmse:', test_rmse, 'test_mae:', test_mae, 'r2_test:', test_r2)

            test_param['rmse'].append(test_rmse)
            test_param['r2'].append(test_r2)
            test_param['mae'].append(test_mae)

            res['epochs'].append(epoch)
            # res['losess'].append(test_loss)
            print('res_best:', res['best_test'])
            if test_r2 > res['best_test']:
                res['best_test'] = test_r2
                # res['best_test'] = test_loss
                res['best_epoch'] = epoch
                best_test_predictions = test_predictions
                best_test_labels = test_labels
                best_test_uncertainties = test_uncertainties
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch
                }
                torch.save(state, path + "/checkpoint6.pth")
                xdata = pd.DataFrame({
                    'predictions': best_test_predictions, 
                    'labels': best_test_labels, 
                    'uncertainty': best_test_uncertainties
                })
                xdata.to_csv('./pred6.csv', index=None)

            all_val_loss.append(val_loss)
            all_test_loss.append(test_loss)
        # save current loss
            xdata = pd.DataFrame(
                {'train_rmse': train_param['rmse'], 'r2_train': train_param['r2'], 'train_mae': train_param['mae'],
                 'val_rmse': valid_param['rmse'],  'r2_val': valid_param['r2'], 'val_mae': valid_param['mae'],
                 'test_rmse:': test_param['rmse'], 'test_mae:': test_param['mae'], 'r2_test:': test_param['r2']})
            xdata.to_csv('./results6.csv', index=None)

        loss_file = path + '/loss6.pkl'
        with open(loss_file, 'wb') as f:
            pkl.dump((all_train_loss, all_val_loss, all_test_loss), f)

