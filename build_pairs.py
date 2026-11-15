"""Build an index of correctly matched pre/post image/label pairs (Step 2)."""
import argparse
from pathlib import Path
import pandas as pd


def build_pair_index(data_dir: str | Path) -> pd.DataFrame:
    base = Path(data_dir) / 'train'
    imgs = base / 'images'
    labels = base / 'labels'
    records = []
    for pre in sorted([*imgs.glob('*pre_disaster*.tif'), *imgs.glob('*pre_disaster*.tiff')]):
        post = pre.with_name(pre.name.replace('_pre_disaster', '_post_disaster'))
        pre_lbl = labels / f'{pre.stem}.json'
        post_lbl = labels / f'{post.stem}.json'
        if not all(p.exists() for p in (post, pre_lbl, post_lbl)):
            continue  # Incomplete pairs should never be aligned by sorting and zipping.
        key = pre.stem.replace('_pre_disaster', '')
        event = key.rsplit('_', 1)[0] if '_' in key else key
        records.append(dict(pre_img=str(pre.resolve()), post_img=str(post.resolve()),
                            pre_lbl=str(pre_lbl.resolve()), post_lbl=str(post_lbl.resolve()),
                            event=event, pair_id=key))
    if not records:
        raise FileNotFoundError(f'No complete pre/post .tif pairs under {imgs}')
    return pd.DataFrame(records)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out', default='pairs_index.csv')
    args = p.parse_args()
    df = build_pair_index(args.data_dir)
    df.to_csv(args.out, index=False)
    print(f'Indexed {len(df)} pairs; wrote {args.out}')
