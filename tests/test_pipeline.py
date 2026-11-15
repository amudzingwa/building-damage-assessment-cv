"""Lightweight regression tests; use pytest -q."""
import numpy as np
import pandas as pd

from extract_patches import crop_padded
from split_dataset import split_by_event
from metrics import compute_metrics


def test_crop_padded_keeps_size():
    image = np.ones((80, 55, 3), dtype=np.float32)
    crop = crop_padded(image, cx=0, cy=0, size=128)
    assert crop.shape == (128, 128, 3)
    assert crop[0, 0, 0] == 1
    assert crop[-1, -1, 0] == 0


def test_split_never_leaks_event(tmp_path):
    rows = [{'file': f'{event}_{i}.npy', 'label': i % 4, 'event': event}
            for event in ('alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot')
            for i in range(4)]
    metadata = tmp_path / 'patch_metadata.csv'
    pd.DataFrame(rows).to_csv(metadata, index=False)
    split_by_event(str(metadata), str(tmp_path))
    sets = [set(pd.read_csv(tmp_path / f'{p}_patches.csv').event) for p in ('train', 'val', 'test')]
    assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
    assert len(set.union(*sets)) == 6


def test_metrics_perfect_predictions():
    scores = compute_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    assert scores['OA'] == scores['wF1'] == scores['mIoU'] == 1
