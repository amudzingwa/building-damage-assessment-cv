"""Extract aligned pre/post building-centered NumPy patches (Step 3)."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from shapely import wkt
from tqdm.auto import tqdm
from read_geotiff import read_geotiff

LABEL_MAP = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}


def crop_padded(image: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    """Crop an exactly size x size view; pad small source images at edges."""
    h, w = image.shape[:2]
    x = max(0, min(cx - size // 2, max(0, w - size)))
    y = max(0, min(cy - size // 2, max(0, h - size)))
    arr = image[y:y + size, x:x + size]
    return np.pad(arr, ((0, size - arr.shape[0]), (0, size - arr.shape[1]), (0, 0)))


def extract_patches(pairs_csv: str, patch_dir: str, metadata_csv: str, size: int = 512,
                    limit_pairs: int | None = None) -> pd.DataFrame:
    pairs = pd.read_csv(pairs_csv)
    if limit_pairs is not None:
        pairs = pairs.head(limit_pairs)
    root = Path(patch_dir)
    records = []
    for row in tqdm(pairs.itertuples(index=False), total=len(pairs), desc='Pairs'):
        pre = read_geotiff(row.pre_img)
        post = read_geotiff(row.post_img)
        if pre.shape != post.shape:
            raise ValueError(f'Pre/post image shapes differ for {row.post_img}')
        with open(row.post_lbl, encoding='utf-8') as handle:
            label_data = json.load(handle)
        for idx, feature in enumerate(label_data.get('features', {}).get('xy', [])):
            damage = feature.get('properties', {}).get('subtype', 'un-classified')
            if damage not in LABEL_MAP:
                continue
            try:
                centroid = wkt.loads(feature['wkt']).centroid
                cx, cy = round(centroid.x), round(centroid.y)
            except (KeyError, ValueError, TypeError):
                continue
            cls = LABEL_MAP[damage]
            fname = f'{row.pair_id}_{idx}_{cx}_{cy}.npy'
            post_dest = root / str(cls) / fname
            pre_dest = root / 'pre' / str(cls) / fname
            post_dest.parent.mkdir(parents=True, exist_ok=True)
            pre_dest.parent.mkdir(parents=True, exist_ok=True)
            if not post_dest.exists():
                np.save(post_dest, crop_padded(post, cx, cy, size).astype(np.float32))
            if not pre_dest.exists():
                np.save(pre_dest, crop_padded(pre, cx, cy, size).astype(np.float32))
            records.append(dict(file=fname, pre_file=fname, label=cls,
                                event=row.event, pair_id=row.pair_id))
    df = pd.DataFrame(records, columns=['file', 'pre_file', 'label', 'event', 'pair_id'])
    df.to_csv(metadata_csv, index=False)
    return df


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', required=True)
    p.add_argument('--patch-dir', required=True)
    p.add_argument('--out', default='patch_metadata.csv')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--limit-pairs', type=int, default=None,
                   help='Useful for validating the pipeline before processing all scenes')
    a = p.parse_args()
    result = extract_patches(a.pairs, a.patch_dir, a.out, a.size, a.limit_pairs)
    print(f'Saved {len(result)} paired patches and metadata to {a.out}')
