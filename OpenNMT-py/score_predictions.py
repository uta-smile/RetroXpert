#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import configargparse
import pandas as pd
import numpy as np
from collections import Counter
from rdkit import Chem


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''

def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        preds = row['{}{}'.format(base, i)].split('.')
        preds.sort()
        targets = row['target'].split('.')
        targets.sort()
        if preds == targets:
            return i
    return 0

def main(opt):
    with open(opt.targets, 'r') as f:
        targets = [''.join(line.strip().split(' ')) for line in f.readlines()]
    with open(opt.sources, 'r') as f:
        rxn_cls = []
        products = []
        sources = []
        for line in f.readlines():
            items = line.strip().split(' ')
            rxn_cls.append(items[0])
            product, source = "".join(items[1:]).split('[PREDICT]')
            products.append(product)
            sources.append(source)

    test_df = pd.DataFrame(rxn_cls)
    test_df.columns = ['rxn_cls']
    test_df['product'] = products
    test_df['synthon'] = sources
    test_df['target'] = targets

    total = len(test_df)
    predictions = [[] for i in range(opt.beam_size)]
    with open(opt.predictions, 'r') as f:
        for i, line in enumerate(f.readlines()):
            predictions[i % opt.beam_size].append(''.join(line.strip().split(' ')))

    for i, preds in enumerate(predictions):
        test_df['prediction_{}'.format(i + 1)] = preds
        test_df['canonical_prediction_{}'.format(i + 1)] = test_df['prediction_{}'.format(i + 1)].apply(
            lambda x: canonicalize_smiles(x))

    test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'canonical_prediction_', opt.beam_size), axis=1)
    correct = 0
    for i in range(1, opt.beam_size+1):
        correct += (test_df['rank'] == i).sum()
        invalid_smiles = (test_df['canonical_prediction_{}'.format(i)] == '').sum()
        if opt.invalid_smiles:
            print('Top-{}: {:.1f}% || Invalid SMILES {:.2f}%'.format(i, correct/total*100,
                                                                     invalid_smiles/total*100))
        else:
            print('Top-{}: {:.1f}%'.format(i, correct / total * 100))
    
    test_df.to_csv(opt.predictions[:-4] + '.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    parser = configargparse.ArgParser(default_config_files=['/etc/app/conf.d/*.conf', '~/.my_settings'])

    parser.add('--beam_size', type=int, default=5, help='Beam size')
    parser.add('--invalid_smiles', action="store_true", help='Show % of invalid SMILES')
    parser.add('--predictions', type=str, default="", help="Path to file containing the predictions")
    parser.add('--targets', type=str, default="", help="Path to file containing targets")
    parser.add('--sources', type=str, default="", help="Path to file containing sources with rxn index")
    parser.add('--reaction_center_preds', type=str, default="", help="Path of reaction center prediction results")

    opt = parser.parse_args()
    
    main(opt)

    df = pd.read_csv(opt.predictions[:-4] + '.csv', encoding='utf-8')
    targets = df['target'].tolist()
    targets = [canonicalize_smiles(smi) for smi in targets]
    predictions = df['canonical_prediction_1'].tolist()

    pred_results_2th = []
    for tgt, pred in zip(targets, predictions):
        pred_results_2th.append(tgt == pred)
    
    print('--Second phase Top1 acc = ', sum(pred_results_2th) / len(pred_results_2th))

    if opt.reaction_center_preds:
        rc_pred_mask = np.loadtxt(opt.reaction_center_preds).astype(np.bool)
        print('First phase acc = ', sum(rc_pred_mask) / len(rc_pred_mask))
        cnt = 0
        for pred_1th, pred_2th in zip(rc_pred_mask, pred_results_2th):
            if pred_1th == True and pred_2th == True:
                cnt += 1
        print('Two phases Top1 acc = ', cnt / len(rc_pred_mask))

        rank = df['rank'].tolist()
        counter = Counter()
        for pred_1th, r in zip(rc_pred_mask, rank):
            if pred_1th:
                counter[r] += 1
        cnt = 0
        for idx in range(1, 1+opt.beam_size):
            cnt += counter[idx]
            print('Top-{} : {:.1f}'.format(idx, cnt / len(rank) * 100))