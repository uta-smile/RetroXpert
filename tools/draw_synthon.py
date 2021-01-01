import matplotlib.pyplot as plt
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdDepictor, Draw
from rdkit.Chem.Draw import rdMolDraw2D, DrawingOptions
DrawingOptions.atomLabelFontSize = 15
DrawingOptions.dotsPerAngstrom = 100
DrawingOptions.bondLineWidth = 3.0

file = r'C:\Users\chaochaoyan\Documents\retrosynthesis\retrosim\retrosim\data\data_split.csv'

data = pd.read_csv(file)
rxn = data['rxn_smiles'].tolist()



def get_tagged_atoms_from_mol(mol):
    '''Takes an RDKit molecule and returns list of tagged atoms and their
    corresponding numbers'''
    atoms = []
    atom_tags = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atoms.append(atom)
            atom_tags.append(str(atom.GetProp('molAtomMapNumber')))
    return atoms, atom_tags


for i, r in enumerate(rxn[:2]):
    s, t = r.split('>>')
    print(t)

    mol = Chem.MolFromSmiles(t)
    if mol is None:
        print('None mol:', t)

    prod_atoms, prod_atom_tags = get_tagged_atoms_from_mol(mol)
    reactants = s.split('.')
    react_mols, react_smis = [], []
    for j, reactant in enumerate(reactants):
        mol_react = Chem.MolFromSmiles(reactant)
        if mol_react is None:
            print('None mol:', reactant)
            continue

        react_atoms, react_atom_tags = get_tagged_atoms_from_mol(mol_react)
        react_atoms_index = [atom.GetIdx() for atom in react_atoms]
        react_atom_index_all = [atom.GetIdx() for atom in mol_react.GetAtoms()]
        atoms_to_remove = [idx for idx in react_atom_index_all if idx not in react_atoms_index]
        atoms_to_remove.sort(reverse=True)

        emol = Chem.EditableMol(mol_react)
        for atom in atoms_to_remove:
            emol.RemoveAtom(atom)

        mol_new = emol.GetMol()
        Chem.SanitizeMol(mol_new)
        smi_react = Chem.MolToSmiles(mol_new)
        print('smi_react:', smi_react)

        react_mols.append(mol_new)
        react_smis.append(smi_react)

    img = Draw.MolsToGridImage(
        react_mols,
        molsPerRow=1,
        subImgSize=(200, 200),
        legends=react_smis
    )

    plt.imshow(img)
    plt.tight_layout()
    plt.axis('off')
    plt.show()

