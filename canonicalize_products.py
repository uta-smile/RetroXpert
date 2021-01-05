import pandas as pd
from rdkit import Chem
from tqdm import tqdm

input_csv_file = ''
output_csv_file = ''

csv = pd.read_csv(input_csv_file)
reaction_list = csv['rxn_smiles']
reactant_smiles_list = list(map(lambda x: x.split('>>')[0], reaction_list))
product_smiles_list = list(map(lambda x: x.split('>>')[1], reaction_list))
reaction_list_new = []

for idx, product in enumerate(tqdm(product_smiles_list)):
    mol = Chem.MolFromSmiles(product)
    index2mapnums = {}
    for atom in mol.GetAtoms():
        index2mapnums[atom.GetIdx()] = atom.GetAtomMapNum()

    # canonicalize the product smiles
    mol_cano = Chem.RWMol(mol)
    [atom.SetAtomMapNum(0) for atom in mol_cano.GetAtoms()]
    smi_cano = Chem.MolToSmiles(mol_cano)
    mol_cano = Chem.MolFromSmiles(smi_cano)

    matches = mol.GetSubstructMatches(mol_cano)
    if matches:
        for atom, mat in zip(mol_cano.GetAtoms(), matches[0]):
            atom.SetAtomMapNum(index2mapnums[mat])
        product = Chem.MolToSmiles(mol_cano, canonical=False)
    reaction_list_new.append(reactant_smiles_list[idx] + '>>' + product)

csv['rxn_smiles'] = reaction_list_new
csv.to_csv(output_csv_file, index=False)
