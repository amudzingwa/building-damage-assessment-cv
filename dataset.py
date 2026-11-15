"""PyTorch DataLoader for 4-class building-level classification (Step 5)."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from augmentation import get_train_transform, get_val_transform


class XBDDataset(Dataset):
    def __init__(self, csv_path, patch_base_dir, transform=None, require_pre=False):
        self.df = pd.read_csv(csv_path)
        self.patch_dir = Path(patch_base_dir)
        self.transform = transform
        self.require_pre = require_pre
        if require_pre and 'pre_file' not in self.df:
            raise ValueError('Siamese training requires pre_file metadata; re-extract paired patches')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cls = int(row['label'])
        post = np.load(self.patch_dir / str(cls) / row['file']).astype(np.float32)
        pre_path = self.patch_dir / 'pre' / str(cls) / str(row.get('pre_file', ''))
        if pre_path.is_file():
            pre = np.load(pre_path).astype(np.float32)
        elif self.require_pre:
            raise FileNotFoundError(f'Missing true pre-disaster patch: {pre_path}')
        else:
            pre = post.copy()  # Baseline/ViT never inspect this input.
        if self.transform is not None:
            transformed = self.transform(image=pre, post=post)
            pre, post = transformed['image'], transformed['post']
        else:
            pre = torch.from_numpy(np.moveaxis(pre, -1, 0))
            post = torch.from_numpy(np.moveaxis(post, -1, 0))
        return {'pre': pre.float(), 'post': post.float(),
                'label': torch.tensor(cls, dtype=torch.long)}


def get_loaders(metadata_dir, patch_dir, batch_size=16, size=None, require_pre=False, num_workers=2):
    root = Path(metadata_dir)
    data = [XBDDataset(root / f'{part}_patches.csv', patch_dir,
                       get_train_transform(size) if part == 'train' else get_val_transform(size),
                       require_pre=require_pre)
            for part in ('train', 'val', 'test')]
    return tuple(DataLoader(ds, batch_size=batch_size if i == 0 else 32,
                            shuffle=i == 0, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available())
                 for i, ds in enumerate(data))
