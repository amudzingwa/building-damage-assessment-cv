"""Synchronized pre/post geometric augmentations (Step 4)."""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _pipeline(train: bool, size: int | None = None):
    steps = []
    if size is not None:
        steps.append(A.Resize(size, size))
    if train:
        steps += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                  A.RandomRotate90(p=0.5), A.RandomBrightnessContrast(p=0.4)]
    steps.append(ToTensorV2())
    return A.Compose(steps, additional_targets={'post': 'image'})


def get_train_transform(size: int | None = None):
    return _pipeline(True, size)


def get_val_transform(size: int | None = None):
    return _pipeline(False, size)
