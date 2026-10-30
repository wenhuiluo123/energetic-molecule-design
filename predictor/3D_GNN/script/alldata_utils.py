import logging
import os
import random
import shutil
import rdkit
from rdkit import Chem
 # import rdkit.Chem.rdmolops as rd
import torch
import tarfile
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def process_sdf_e4(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    mol = Chem.MolFromMolFile(datafile, removeHs=False)
    mol_atom_prop = []
    mol_bond_prop = []
    charges = []

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 67:
        print('a:'+datafile)

    for atom in mol.GetAtoms():
        atom_prop = atom_features(atom)
        mol_atom_prop.append(atom_prop)
        atom_charge = atom.GetAtomicNum()
        charges.append(atom_charge)
    for bond in mol.GetBonds():
        bond_prop = bond_features(bond)
        mol_bond_prop.append(bond_prop)
    atom_position = read_coors(datafile)
    atom_ddec_charge = read_ddec_charge(datafile)
    atom_ddec_charge = atom_ddec_charge.squeeze(-1)
    # adj_matrix = rd.GetAdjacencyMatrix(mol)
    # adj_matrix = torch.tensor(adj_matrix)

    molecule = {'mol_atom_prop': mol_atom_prop, 'charges': charges, 'mol_bond_prop': mol_bond_prop}

    mol_props = {'num_atoms': num_atoms, 'positions': atom_position, 'ddec_charges': atom_ddec_charge}
    molecule.update(mol_props)
    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    return molecule

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H',
                                           'Unknown']) +  # H?
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()] +
                    [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
                    one_of_k_encoding_unk(atom.GetHybridization(), [
                        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                          SP3D, Chem.rdchem.HybridizationType.SP3D2]))


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([int(bt == Chem.rdchem.BondType.SINGLE),
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a, 0.0))


def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))

def read_coors(datafile):
    T = False
    with open(datafile, "rb") as f:
    # lines下标从0开始
        lines = f.readlines()
    for i, row_line in enumerate(lines):
        row_line = row_line.decode('utf-8').replace("\n", "")
        if i == 3:
            number = int(lines[i].split()[0])
            start_line = i + 1
            end_line = i + number
            T = True
            break
    if T:
        cb = np.zeros((number, 3))
        j = 0
        for i in range(start_line , end_line + 1):
            lines[i] = lines[i].decode('utf-8').replace("\n", "")
            lines[i] = [lines[i].split()[0] , lines[i].split()[1] , lines[i].split()[2]]
            ca = np.array(lines[i])
            cb[j] = ca
            j = j + 1
        cc = np.average(cb, axis=0)
        for i in range(0, 3):
            cb[:, i] = cb[:, i] - cc[i]
        cb = torch.tensor(cb)
        return cb

def read_ddec_charge(datafile):
    resp_charges = []
    T = True
    ca = None
    with open(datafile, "rb") as f:
        lines = f.readlines()
    for i, row_line in enumerate(lines):
        row_line = row_line.decode('utf-8').replace("\n", "")
        if row_line == '>  <Label>  (1) ':
            line = lines[i + 1].decode('utf-8').replace("\n", "")
            ca = np.array(float(line))

    if ca is None:
        raise ValueError(f"Charge value not found in {datafile}")

    cb = torch.tensor(ca)
    return cb


def copyFile(fileDir, save_dir):
    train_rate = 1
    valid_rate = 0.0

    image_list = os.listdir(fileDir)
    image_number = len(image_list)
    train_number = int(image_number * train_rate)
    valid_number = int(image_number * valid_rate)
    train_sample = random.sample(image_list, train_number)
    valid_sample = random.sample(list(set(image_list) - set(train_sample)), valid_number)
    test_sample = list(set(image_list) - set(train_sample) - set(valid_sample))
    sample = [train_sample, valid_sample, test_sample]

    for k in range(len(save_dir)):
        # os.makedirs(save_dir[k])
        # for name in sample[k]:
        #     shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k], name))

        if not os.path.isdir(save_dir[k]):
            os.makedirs(save_dir[k])

        for name in sample[k]:
            shutil.copy(os.path.join(fileDir, name),
                        os.path.join(save_dir[k] + '/', name))


def convert(T):
    # props = {'mol_atom_prop', 'charges', 'mol_bond_prop', 'num_atoms', 'positions', 'ddec_charges'}
    props = T[0].keys()
    assert all(props == mol.keys() for mol in T), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists
    T = {prop: [mol[prop] for mol in T] for prop in props}

    T = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in
         T.items()}

    return T


def prepare_dataset(datadir, dataset, subset=None, splits=None, copy=True):
    """
    Download and process dataset.

    Parameters
    ----------
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : dict, optional
        Dataset splits to use.
    cleanup : bool, optional
        Clean up files created while preparing the data.
    force_download : bool, optional
        If true, forces a fresh download of the dataset.

    Returns
    -------
    datafiles : dict of strings
        Dictionary of strings pointing to the files containing the data.

    Notes
    -----
    TODO: Delete the splits argument?
    """

    # If datasets have subsets,
    if subset:
        dataset_dir = [datadir, dataset, subset]
    else:
        dataset_dir = [datadir, dataset]

    # Names of splits, based upon keys if split dictionary exists, elsewise default to train/valid/test.
    split_names = splits.keys() if splits is not None else [
        'train', 'valid', 'test']


    # Assume one data file for each split
    data_splits = {split: os.path.join(
        datadir + '/', split) for split in split_names}
    datafiles = {split: os.path.join(datadir, split + '.npz') for split in split_names}

    # Check data_splits exist
    # data_splits_checks = [os.path.exists(datafile) for datafile in data_splits.values()]
    #
    # # Check if prepared dataset exists, and if not set flag to download below.
    # # Probably should add more consistency checks, such as number of datapoints, etc...
    # new_download = False
    # if all(data_splits_checks):
    #     logging.info('Dataset exists and is processed.')
    # elif all([not x for x in data_splits_checks]):
    #     # If checks are failed.
    #     new_download = True
    # else:
    #     raise ValueError(
    #         'Dataset only partially processed. Try deleting {} and running again to download/process.'.format(os.path.join(dataset_dir)))

    save_train_dir = data_splits['train']
    save_valid_dir = data_splits['valid']
    save_test_dir = data_splits['test']
    save_dir = [save_train_dir, save_valid_dir, save_test_dir]
    path = os.path.join(datadir, dataset)
    if copy == True:
        copyFile(path, save_dir)
    else:
        pass

    e4_data = {}
    train = []
    valid = []
    test = []
    for split, split_path in data_splits.items():
        for dirpath, dirnames, filenames in os.walk(split_path):
            for filepath in filenames:
                data_path = os.path.join(dirpath, filepath)
                print(data_path)
                if split == 'train':
                    train.append(process_sdf_e4(data_path))
                elif split == 'valid':
                    valid.append(process_sdf_e4(data_path))
                else:
                    test.append(process_sdf_e4(data_path))
    train = convert(train)
    test = convert(test)
    valid = convert(valid)


    e4_data['train'] = train
    e4_data['test'] = test
    e4_data['valid'] = valid

    for split, data in e4_data.items():
        savedir = os.path.join(datadir, split + '.npz')
        np.savez_compressed(savedir, **data)
    print('successful')

    return datafiles



