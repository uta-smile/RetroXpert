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
    args.output_suffix = '-aug-typed'
else:
    args.output_suffix = '-aug-untyped'
print(args)


savedir = 'OpenNMT-py/data/{}{}'.format(args.dataset, args.output_suffix)
if not os.path.exists(savedir):
    os.mkdir(savedir)


src = {
    'train': 'src-train-aug.txt',
    'test': 'src-test.txt',
    'valid': 'src-valid.txt',
}
tgt = {
    'train': 'tgt-train-aug.txt',
    'test': 'tgt-test.txt',
    'valid': 'tgt-valid.txt',
}


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


tokens = Counter()
reaction_atoms = {}
reaction_atoms_list = []
for data_set in ['valid', 'train', 'test']:
    with open(os.path.join('opennmt_data', src[data_set])) as f:
        srcs = f.readlines()
    with open(os.path.join('opennmt_data', tgt[data_set])) as f:
        tgts = f.readlines()

    src_lines = []
    tgt_lines = []
    reaction_atoms_lists = []
    unknown = set()
    for s, t in tqdm(list(zip(srcs, tgts))):
        tgt_items = t.strip().split()
        src_items = s.strip().split()
        src_items[2] = smi_tokenizer(smarts2smiles(src_items[2]))
        tokens.update(src_items[2].split(' '))
        for idx in range(4, len(src_items)):
            if src_items[idx] == '.':
                continue
            src_items[idx] = smi_tokenizer(smarts2smiles(src_items[idx], canonical=False))
            tokens.update(src_items[idx].split(' '))
        for idx in range(len(tgt_items)):
            if tgt_items[idx] == '.':
                continue
            tgt_items[idx] = smi_tokenizer(smarts2smiles(tgt_items[idx]))
            tokens.update(tgt_items[idx].split(' '))

        if not args.typed:
            src_items[1] = '[RXN_0]'

        src_line = ' '.join(src_items[1:])
        tgt_line = ' '.join(tgt_items)
        src_lines.append(src_line + '\n')
        tgt_lines.append(tgt_line + '\n')

    src_file = os.path.join(savedir, src[data_set])
    print('src_file:', src_file)
    with open(src_file, 'w') as f:
        f.writelines(src_lines)

    tgt_file = os.path.join(savedir, tgt[data_set])
    print('tgt_file:', tgt_file)
    with open(tgt_file, 'w') as f:
        f.writelines(tgt_lines)
