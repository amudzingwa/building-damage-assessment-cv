"""Split by disaster event to prevent spatial leakage (Step 5)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def split_by_event(metadata_csv: str, out_dir: str, seed: int = 42):
    df = pd.read_csv(metadata_csv)
    events = df.event.dropna().unique().tolist()
    if len(events) < 3:
        raise ValueError('At least three disaster events required for train/val/test split')
    rng = np.random.default_rng(seed)
    rng.shuffle(events)
    n_test = max(1, round(len(events) * .15))
    n_val = max(1, round(len(events) * .15))
    assignment = dict.fromkeys(events[:n_test], 'test')
    assignment.update(dict.fromkeys(events[n_test:n_test + n_val], 'val'))
    assignment.update(dict.fromkeys(events[n_test + n_val:], 'train'))
    if not any(v == 'train' for v in assignment.values()):
        raise ValueError('No train events remain; add more events')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for part in ('train', 'val', 'test'):
        subset = df[df.event.map(assignment).eq(part)]
        subset.to_csv(out / f'{part}_patches.csv', index=False)
        print(f'{part}: {len(subset)} patches from {subset.event.nunique()} events')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', required=True)
    p.add_argument('--out-dir', required=True)
    a = p.parse_args()
    split_by_event(a.metadata, a.out_dir)
