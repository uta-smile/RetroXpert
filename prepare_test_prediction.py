import argparse
import re
import os
import pickle
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    default='USPTO50K',
                    help='dataset: USPTO50K')
parser.add_argument('--typed',
                    action='store_true',
                    default=False,
                    help='if given reaction types')
args = parser.parse_args()

assert args.dataset in ['USPTO50K', 'USPTO-full']

if args.typed:
    args.typed = 'typed'
    args.output_suffix = '-aug-typed'
else:
    args.typed = 'untyped'
    args.output_suffix = '-aug-untyped'
print(args)



# Get the mapping numbers in a SMILES.
def get_idx(smarts):
    item = re.findall('(?<=:)\d+', smarts)
    item = list(map(int, item))
    return item

def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

# Convert smarts to smiles by remove mapping numbers
def smarts2smiles(smarts, canonical=True):
    t = re.sub(':\d*', '', smarts)
    mol = Chem.MolFromSmiles(t, sanitize=False)
    return Chem.MolToSmiles(mol, canonical=canonical)

def get_smarts_pieces(mol, src_adj, target_adj, add_bond=False):
    m, n = src_adj.shape
    emol = Chem.EditableMol(mol)
    for j in range(m):
        for k in range(j + 1, n):
            if target_adj[j][k] == src_adj[j][k]:
                continue
            if 0 == target_adj[j][k]:
                emol.RemoveBond(j, k)
            elif add_bond:
                emol.AddBond(j, k, Chem.rdchem.BondType.SINGLE)
    synthon_smiles = Chem.MolToSmiles(emol.GetMol(), isomericSmiles=True)
    return synthon_smiles



pred_results = 'logs/test_result_mol_{}_{}.txt'.format(args.dataset, args.typed)
with open(pred_results) as f:
    pred_results = f.readlines()

bond_pred_results = 'logs/test_disconnection_{}_{}.txt'.format(args.dataset, args.typed)
with open(bond_pred_results) as f:
    bond_pred_results = f.readlines()

dataset_dir = 'data/{}/test'.format(args.dataset)
reaction_data_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pkl')]
reaction_data_files.sort()

product_adjs = []
product_mols = []
product_smiles = []
for data_file in tqdm(reaction_data_files):
    with open(os.path.join(dataset_dir, data_file), 'rb') as f:
        reaction_data = pickle.load(f)
    product_adjs.append(reaction_data['product_adj'])
    product_mols.append(reaction_data['product_mol'])
    product_smiles.append(Chem.MolToSmiles(reaction_data['product_mol'], canonical=False))

assert len(product_smiles) == len(bond_pred_results)



cnt = 0
guided_pred_results = []
bond_disconnection = []
bond_disconnection_gt = []
for i in range(len(bond_pred_results)):
    bond_pred_items = bond_pred_results[i].strip().split()
    bond_change_num = int(bond_pred_items[1]) * 2
    bond_change_num_gt = int(bond_pred_items[0]) * 2

    gt_adj_list = pred_results[3 * i + 1].strip().split(' ')
    gt_adj_list = [int(k) for k in gt_adj_list]
    gt_adj_index = np.argsort(gt_adj_list)
    gt_adj_index = gt_adj_index[:bond_change_num_gt]

    pred_adj_list = pred_results[3 * i + 2].strip().split(' ')
    pred_adj_list = [float(k) for k in pred_adj_list]
    pred_adj_index = np.argsort(pred_adj_list)
    pred_adj_index = pred_adj_index[:bond_change_num]

    bond_disconnection.append(pred_adj_index)
    bond_disconnection_gt.append(gt_adj_index)
    res = set(gt_adj_index) == set(pred_adj_index)
    guided_pred_results.append(int(res))
    cnt += res


print('guided bond_disconnection prediction cnt and acc:', cnt, cnt / len(bond_pred_results))
print('bond_disconnection len:', len(bond_disconnection))


# Generate synthons from bond disconnection prediction
synthons = []
for i, prod_adj in enumerate(product_adjs):
    x_adj = np.array(prod_adj)
    # find 1 index
    idxes = np.argwhere(x_adj > 0)
    pred_adj = prod_adj.copy()
    for k in bond_disconnection[i]:
        idx = idxes[k]
        assert pred_adj[idx[0], idx[1]] == 1
        pred_adj[idx[0], idx[1]] = 0

    pred_synthon = get_smarts_pieces(product_mols[i], prod_adj, pred_adj)
    synthons.append(pred_synthon)



with open('opennmt_data/src-test.txt') as f:
    srcs = f.readlines()
assert len(synthons) == len(srcs)



savedir = 'OpenNMT-py/data/{}{}'.format(args.dataset, args.output_suffix)
src_test_prediction = os.path.join(savedir, 'src-test-prediction.txt')
print('save src_test_prediction:', src_test_prediction)
cnt = 0
with open(src_test_prediction, 'w') as f:
    for src, synthon in zip(srcs, synthons):
        src_items = src.split(' ')
        src_items[2] = smi_tokenizer(smarts2smiles(src_items[2]))
        if args.typed == 'untyped':
            src_items[1] = '[RXN_0]'

        syns = synthon.split('.')
        syns = [smarts2smiles(s, canonical=False) for s in syns]

        # Double check the synthon prediction accuracy
        syns_gt = [smarts2smiles(s, canonical=False) for s in src_items[4:] if s != '.']
        cnt += set(syns_gt) == set(syns)

        syns = [smi_tokenizer(s) for s in syns]
        src_line = ' '.join(src_items[1:4]) + ' ' + ' . '.join(syns) + '\n'
        f.write(src_line)

print('double check guided synthon prediction acc:', cnt / len(synthons))
