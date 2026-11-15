"""Train a 4-class model (UNet classifier, Siamese ResNet, or ViT)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataset import get_loaders
from metrics import evaluate
from models import build_model


def class_weights(csv_file):
    labels = pd.read_csv(csv_file)['label'].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=4)
    if np.any(counts == 0):
        raise ValueError(f'Every class needs training examples; counts: {counts}')
    return torch.tensor(counts.sum() / (4.0 * counts), dtype=torch.float32)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size = 224 if args.arch == 'vit' else args.size
    train_loader, val_loader, _ = get_loaders(
        args.metadata_dir, args.patch_dir, batch_size=args.batch_size,
        size=size, require_pre=args.arch == 'siamese', num_workers=args.workers)
    model = build_model(args.arch, encoder_name=args.encoder, pretrained=args.pretrained).to(device)
    weight = class_weights(Path(args.metadata_dir) / 'train_patches.csv').to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or (1e-5 if args.arch == 'vit' else 1e-4),
                                  weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for batch in train_loader:
            post = batch['post'].to(device)
            y = batch['label'].to(device)
            logits = model(batch['pre'].to(device), post) if args.arch == 'siamese' else model(post)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_n += len(y)
        scheduler.step()
        scores = evaluate(model, val_loader, device, siamese=args.arch == 'siamese')
        row = {'epoch': epoch, 'train_loss': total_loss / max(total_n, 1), **scores}
        history.append(row)
        print(json.dumps(row))
        if scores['wF1'] > best:
            best = scores['wF1']
            torch.save({'architecture': args.arch, 'encoder': args.encoder, 'size': size,
                        'state_dict': model.state_dict(), 'val_metrics': scores},
                       output / f'best_{args.arch}.pt')
        (output / f'history_{args.arch}.json').write_text(json.dumps(history, indent=2))
    return best


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata-dir', required=True)
    ap.add_argument('--patch-dir', required=True)
    ap.add_argument('--output-dir', default='results')
    ap.add_argument('--arch', choices=['unet', 'siamese', 'vit'], default='unet')
    ap.add_argument('--encoder', default='resnet50')
    ap.add_argument('--pretrained', action='store_true', help='Download pretrained weights (internet required)')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--size', type=int, default=256, help='UNet/Siamese size; ViT uses 224')
    ap.add_argument('--lr', type=float)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    train(ap.parse_args())
