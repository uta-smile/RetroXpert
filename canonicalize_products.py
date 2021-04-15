'''
Canonicalize product SMILES by re-assigning atom mapping numbers according to the canonical atom order.
Since the original mapping numbers may indicate the reaction atoms, which results in information leak.
'''

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
        mapnums_old2new = {}
        for atom, mat in zip(mol_cano.GetAtoms(), matches[0]):
            mapnums_old2new[index2mapnums[mat]] = 1 + atom.GetIdx()
            # update product mapping numbers according to canonical atom order
            # to completely remove potential information leak
            atom.SetAtomMapNum(1 + atom.GetIdx())
        product = Chem.MolToSmiles(mol_cano)
        # update reactant mapping numbers accordingly
        mol_react = Chem.MolFromSmiles(reactant_smiles_list[idx])
        for atom in mol_react.GetAtoms():
            if atom.GetAtomMapNum() > 0:
                atom.SetAtomMapNum(mapnums_old2new[atom.GetAtomMapNum()])
        reactant = Chem.MolToSmiles(mol_react)

    reaction_list_new.append(reactant + '>>' + product)

csv['rxn_smiles'] = reaction_list_new
csv.to_csv(output_csv_file, index=False)
