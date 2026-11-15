"""Predict building *image-patch* damage class using an exported ONNX model.

These predictions are experimental, NOT official ATC-20 placards or safety inspections.
"""
import io
import json
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

CLASS_NAMES = ('No Damage', 'Minor Damage', 'Major Damage', 'Destroyed')
SCREENING = ('Low visible damage', 'Review recommended', 'Urgent review', 'Urgent review')


def _image_from_bytes(data: bytes, suffix: str) -> np.ndarray:
    if suffix.lower() in ('.tif', '.tiff'):
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(data) as mem:
            with mem.open() as src:
                bands = [5, 3, 2] if src.count >= 5 else [1, 2, 3]
                if src.count < 3:
                    raise ValueError('GeoTIFF must contain at least three bands')
                image = src.read(bands)
                if np.issubdtype(image.dtype, np.integer):
                    image = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
                image = np.moveaxis(image, 0, -1)
    else:
        try:
            image = np.array(Image.open(io.BytesIO(data)).convert('RGB')).astype(np.float32) / 255.0
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError('Not a supported image') from error
    return np.clip(image.astype(np.float32), 0, 1)


def preprocess(image: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    uint8 = np.uint8(np.round(np.clip(image, 0, 1) * 255))
    resized = np.asarray(Image.fromarray(uint8).resize((size, size), Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0
    return np.moveaxis(resized, 2, 0)[None].astype(np.float32)


class Predictor:
    def __init__(self, model_path='results/model.onnx'):
        import onnxruntime as ort
        self.path = Path(model_path)
        self.session = ort.InferenceSession(str(self.path), providers=['CPUExecutionProvider'])
        config_path = self.path.with_suffix('.json')
        if not config_path.exists():
            raise FileNotFoundError(f'Missing exported model config: {config_path}')
        self.config = json.loads(config_path.read_text())

    def predict(self, post_image: bytes, suffix: str, pre_image: bytes | None = None,
                pre_suffix: str | None = None):
        size = int(self.config['input_size'])
        post = preprocess(_image_from_bytes(post_image, suffix), size)
        arch = self.config['architecture']
        if arch == 'siamese':
            if pre_image is None:
                raise ValueError('This Siamese model requires a matching pre-disaster image')
            pre = preprocess(_image_from_bytes(pre_image, pre_suffix or suffix), size)
            inputs = {'pre': pre, 'post': post}
        else:
            inputs = {'post': post}
        logits = self.session.run(['logits'], inputs)[0][0]
        shifted = logits - np.max(logits)
        probs = np.exp(shifted) / np.exp(shifted).sum()
        cls = int(np.argmax(probs))
        return {'class_id': cls, 'damage_class': CLASS_NAMES[cls],
                'screening_priority': SCREENING[cls],
                'model_probability': round(float(probs[cls]), 4),
                'disclaimer': 'Research screening only. Not an ATC-20 placard, structural inspection, or safety clearance.'}
