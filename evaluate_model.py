"""Evaluate a trained checkpoint on event-held-out test data; exports class metrics and matrix."""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix

from dataset import get_loaders
from metrics import CLASS_NAMES, compute_metrics, predict_labels
from models import build_model


def evaluate_checkpoint(args):
    # Use a checkpoint only from a trusted source; torch.load uses pickle internally.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    arch = checkpoint['architecture']
    size = checkpoint['size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(arch, checkpoint['encoder'], pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    _, _, loader = get_loaders(args.metadata_dir, args.patch_dir, size=size,
                              require_pre=arch == 'siamese', num_workers=args.workers)
    true, pred = predict_labels(model, loader, device, siamese=arch == 'siamese')
    metrics = compute_metrics(true, pred)
    output = Path(args.output_dir)
    output.mkdir(exist_ok=True, parents=True)
    (output / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
    matrix = confusion_matrix(true, pred, labels=[0, 1, 2, 3])
    (output / 'confusion_matrix.json').write_text(json.dumps({'classes': CLASS_NAMES,
                                                                'counts': matrix.tolist()}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--metadata-dir', required=True)
    ap.add_argument('--patch-dir', required=True)
    ap.add_argument('--output-dir', default='results/evaluation')
    ap.add_argument('--workers', type=int, default=2)
    evaluate_checkpoint(ap.parse_args())
