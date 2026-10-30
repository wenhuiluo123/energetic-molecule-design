from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd


def generate_3d_coordinates(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    return mol


def generate_sdf_from_csv(csv_file):

    data = pd.read_excel(csv_file)

    smiles = data['smiles']
    labels = data['Q']

    df = pd.DataFrame({'SMILES': smiles, 'Label': labels})

    for index, row in df.iterrows():

        smiles = row[0]
        label = row[1]

        mol = generate_3d_coordinates(smiles)
        mol.SetProp('Label', str(label))

        writer = Chem.SDWriter(f"predictor/3D_GNN/data/778Q/Q/{index}.sdf")
        writer.write(mol)
        writer.close()

csv_file = 'predictor/3D_GNN/data/778Q/Q_data.xlsx.xlsx'

generate_sdf_from_csv(csv_file)
