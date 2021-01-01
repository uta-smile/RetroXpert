
import pandas as pd

file = r'C:\Users\chaochaoyan\Documents\retrosynthesis\retrosim\retrosim\data\data_split.csv'
data = pd.read_csv(file) 

import cairosvg
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdDepictor, Draw
from rdkit.Chem.Draw import rdMolDraw2D, DrawingOptions
from IPython.display import SVG

DrawingOptions.atomLabelFontSize = 55
DrawingOptions.dotsPerAngstrom = 100
DrawingOptions.bondLineWidth = 3.0

rxn = data['rxn_smiles'].tolist()



def moltosvg(mol, molSize=(450, 150), kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return svg.replace('svg:','')



for i, r in enumerate(rxn[:5]):
    s, t = r.split('>>')
    for idx, smi in enumerate([s, t]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print('None Smiles:', smi)
        
        fig = Draw.MolToMPL(mol, size=(1000, 1000))
        fig.savefig('plots/{}_{}.png'.format(i, idx), bbox_inches='tight')
        
