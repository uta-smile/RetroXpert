import argparse
import json
import numpy as np
import os
import multiprocessing
import pickle
import sys

from collections import Counter
from rdkit import Chem
from tqdm import tqdm

sys.path.append('./util')
from rdchiral.template_extractor import extract_from_reaction

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    default='USPTO50K',
                    help='dataset: USPTO50K')
parser.add_argument('--extract_pattern',
                    action='store_true',
                    default=False,
                    help='if extract pattern from training data')
parser.add_argument('--min_freq',
                    type=int,
                    default=2,
                    help='minimum frequency for patterns to be kept')

args = parser.parse_args()
print('extract semi templates for dataset {}...'.format(args.dataset))
assert args.dataset in ['USPTO50K', 'USPTO-full']

patterns_filtered = []
pattern_file = os.path.join('./data', args.dataset, 'product_patterns.txt')
if not args.extract_pattern and os.path.exists(pattern_file):
    print('load semi template patterns from file:', pattern_file)
    with open(pattern_file) as f:
        patterns = f.readlines()
    for p in patterns:
        pa, cnt = p.strip().split(': ')
        if int(cnt) >= args.min_freq:
            patterns_filtered.append(pa)
    print('total number of semi template patterns:', len(patterns_filtered))


def get_tpl(task):
    idx, react, prod = task
    reaction = {'_id': idx, 'reactants': react, 'products': prod}
    template = extract_from_reaction(reaction, super_general=True)
    return idx, template


def cano_smarts(smarts):
    tmp = Chem.MolFromSmarts(smarts)
    if tmp is None:
        return None, smarts
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    cano = Chem.MolToSmarts(tmp)
    if '[[se]]' in cano:  # strange parse error
        cano = smarts
    return cano


def find_all_patterns(task):
    k, product = task
    product_mol = Chem.MolFromSmiles(product)
    [a.SetAtomMapNum(0) for a in product_mol.GetAtoms()]
    matches_all = {}
    for idx, pattern in enumerate(patterns_filtered):
        pattern_mol = Chem.MolFromSmarts(pattern)
        if pattern_mol is None:
            print('error: pattern_mol is None')
        try:
            matches = product_mol.GetSubstructMatches(pattern_mol,
                                                      useChirality=False)
        except:
            continue
        else:
            if len(matches) > 0 and len(matches[0]) > 0:
                matches_all[idx] = matches
    if len(matches_all) == 0:
        print(product)
    num_atoms = product_mol.GetNumAtoms()
    pattern_feature = np.zeros((len(patterns_filtered), num_atoms))
    for idx, matches in matches_all.items():
        if len(matches) > 1 and isinstance(matches[0], tuple):
            for match in matches:
                np.put(pattern_feature[idx], match, 1)
        else:
            np.put(pattern_feature[idx], matches, 1)
    pattern_feature = pattern_feature.transpose().astype('bool_')
    return k, pattern_feature


for data_set in ['train', 'valid', 'test']:
    data_dir = os.path.join('./data', args.dataset, data_set)
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    data_files.sort()
    products = []
    reactants = []
    for data_file in data_files:
        with open(os.path.join(data_dir, data_file), 'rb') as f:
            reaction_data = pickle.load(f)
        products.append(
            Chem.MolToSmiles(reaction_data['product_mol'], canonical=False))
        reactants.append(
            Chem.MolToSmiles(reaction_data['reactant_mol'], canonical=False))

    if data_set == 'train' and len(patterns_filtered) == 0:
        patterns = {}
        rxns = []
        for idx, r in enumerate(reactants):
            rxns.append((idx, r, products[idx]))
        print('total training rxns:', len(rxns))

        pool = multiprocessing.Pool(12)
        for result in tqdm(pool.imap_unordered(get_tpl, rxns),
                           total=len(rxns)):
            idx, template = result
            if 'reaction_smarts' not in template:
                continue
            product_pattern = cano_smarts(template['products'])
            if product_pattern not in patterns:
                patterns[product_pattern] = 1
            else:
                patterns[product_pattern] += 1

        patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        patterns = ['{}: {}\n'.format(p[0], p[1]) for p in patterns]
        with open(pattern_file, 'w') as f:
            f.writelines(patterns)

        exit(1)

    product_pattern_feat_list = [None] * len(data_files)
    tasks = [(idx, smi) for idx, smi in enumerate(products)]

    counter = []
    pool = multiprocessing.Pool(12)
    for result in tqdm(pool.imap_unordered(find_all_patterns, tasks),
                       total=len(tasks)):
        k, pattern_feature = result
        with open(os.path.join(data_dir, data_files[k]), 'rb') as f:
            reaction_data = pickle.load(f)
        reaction_data['pattern_feat'] = pattern_feature.astype(np.bool)
        with open(os.path.join(data_dir, data_files[k]), 'wb') as f:
            pickle.dump(reaction_data, f)

        pa = np.sum(pattern_feature, axis=0)
        counter.append(np.sum(pa > 0))

    print('# ave center per mol:', np.mean(counter))

'''
pattern_feat feature dim:  646
# ave center per mol: 36
'''
