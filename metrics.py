"""Overall accuracy, weighted F1, class F1, and macro IoU (Step 11)."""
from typing import Sequence
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import torch

CLASS_NAMES = ('no_damage', 'minor', 'major', 'destroyed')


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    if not len(truth):
        raise ValueError('Cannot evaluate an empty set')
    metrics = {
        'OA': float(accuracy_score(truth, pred)),
        'wF1': float(f1_score(truth, pred, average='weighted', labels=list(range(4)),
                              zero_division=0)),
        'mIoU': float(jaccard_score(truth, pred, average='macro', labels=list(range(4)),
                                   zero_division=0)),
    }
    f1 = f1_score(truth, pred, average=None, labels=list(range(4)), zero_division=0)
    metrics.update({f'f1_{name}': float(score) for name, score in zip(CLASS_NAMES, f1)})
    metrics['f1_no_dmg'] = metrics['f1_no_damage']
    return {key: round(value, 4) for key, value in metrics.items()}


def predict_labels(model, loader, device, siamese=False):
    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for batch in loader:
            post = batch['post'].to(device)
            if siamese:
                logits = model(batch['pre'].to(device), post)
            else:
                logits = model(post)
            if logits.ndim != 2 or logits.shape[1] != 4:
                raise ValueError(f'Expected [batch, 4] class logits; got {tuple(logits.shape)}')
            predictions.extend(logits.argmax(1).cpu().tolist())
            truths.extend(batch['label'].tolist())
    return truths, predictions


def evaluate(model, loader, device, siamese=False, model_type='cnn'):
    """The model_type argument is accepted for compatibility with the guide."""
    truth, predictions = predict_labels(model, loader, device, siamese)
    return compute_metrics(truth, predictions)
