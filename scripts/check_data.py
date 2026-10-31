"""Inspect the authorized CMDPAD feature files before running costly training."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

EXPECTED = {
    'single': {'train': 2734, 'valid': 585, 'test': 587},
    'paired': {'train': 2608, 'valid': 554, 'test': 564},
}
REQUIRED_LABELS = {
    'single': ['speaker_valence', 'valence_prediction',
               'speaker_openness', 'speaker_neuroticism',
               'speaker_extraversion', 'speaker_agreeableness',
               'speaker_conscientiousness'],
    'paired': ['valence_prediction'],
}
MODALITIES = ['bert-base-chinese', 'wav2vec2-large-robust-emotion',
              'wav2vec2-large-xlsr-chinese', 'convnext-base']


def check_one(kind, features, labels):
    print(f'\n[{kind}] features={features} labels={labels}')
    if not features.exists() or not labels.exists():
        print('  MISSING: files not found (request access on Hugging Face first).')
        return False
    feat = pd.read_pickle(features)
    lab = pd.read_csv(labels)
    if not isinstance(feat, pd.DataFrame):
        raise TypeError(f'Expected pandas DataFrame in {features}')
    for table, name in ((feat, 'features'), (lab, 'labels')):
        if 'name' not in table.columns:
            raise ValueError(f'{kind} {name} missing name column')
        if table['name'].duplicated().any():
            raise ValueError(f'{kind} {name} contains duplicate name keys')
    if 'mode' not in lab and 'mode' not in feat:
        raise ValueError(f'{kind} missing mode (train/valid/test)')
    if 'mode' in lab and 'mode' in feat:
        compared = lab[['name', 'mode']].merge(feat[['name', 'mode']], on='name', suffixes=('_lab', '_feat'))
        if (compared.mode_lab != compared.mode_feat).any():
            raise ValueError(f'{kind}: feature and label modes disagree')
    missing_labels = sorted(set(REQUIRED_LABELS[kind]) - set(lab.columns))
    if missing_labels:
        raise ValueError(f'{kind}: missing label columns {missing_labels}')
    merged = lab.merge(feat.drop(columns=['mode'], errors='ignore'), on='name', how='inner') if 'mode' in lab else feat.merge(lab, on='name', how='inner')
    if len(merged) != len(lab) or len(merged) != len(feat):
        raise ValueError(f'{kind}: name mismatch: labels={len(lab)}, features={len(feat)}, joined={len(merged)}')
    counts = merged['mode'].value_counts().to_dict()
    print('  matched:', len(merged), 'split counts:', counts)
    for split, expected in EXPECTED[kind].items():
        actual = int(counts.get(split, 0))
        if actual != expected:
            print(f'  NOTE: {split} has {actual} entries vs paper {expected}' + (' (test split may be withheld)' if split == 'test' else ''))
    for modality in MODALITIES:
        for prefix in ([''] if kind == 'single' else ['A_', 'B_']):
            key = prefix + modality
            if key not in merged:
                raise ValueError(f'{kind}: missing required modality column {key}')
            sample = np.asarray(merged.iloc[0][key])
            if sample.size == 0:
                raise ValueError(f'{kind}: empty feature for {key}')
            print(f'  {key}: {sample.size} dimensions')
    if merged[list(REQUIRED_LABELS[kind])].isna().any().any():
        raise ValueError(f'{kind}: missing numeric target labels')
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path('dataset'))
    args = p.parse_args()
    root = args.data_dir
    single = check_one('single', root / 'multimodal_features_single.pkl', root / 'label_single.csv')
    paired = check_one('paired', root / 'multimodal_features_paired.pkl', root / 'label_paired.csv')
    if not (single and paired):
        print('\nCannot run the full pipeline until both file pairs are present.')
        sys.exit(1)
    print('\nPreflight checks passed (this does not verify the unpublished test labels).')


if __name__ == '__main__':
    main()
