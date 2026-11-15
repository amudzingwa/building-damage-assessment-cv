"""Read RGB-compatible views of GeoTIFF satellite images (guide Steps 1-2).

The guide assumes all files have eight uint16 bands. Actual files can differ;
inspect the raster rather than relying on that assumption.
"""
from pathlib import Path
import numpy as np
import rasterio


def read_geotiff(file_path: str | Path, bands: tuple[int, int, int] | None = None) -> np.ndarray:
    """Return an H x W x 3 float32 RGB image scaled into [0, 1].

    For 8+ band rasters use the guide's (5, 3, 2) red/green/blue indices;
    for 3-band rasters use (1, 2, 3). All indices are Rasterio's 1-based IDs.
    """
    with rasterio.open(file_path) as src:
        selected = bands or ((5, 3, 2) if src.count >= 5 else (1, 2, 3))
        if src.count < max(selected):
            raise ValueError(f"{file_path} has only {src.count} bands; requested {selected}")
        data = src.read(list(selected))
        if np.issubdtype(data.dtype, np.integer):
            scale = float(np.iinfo(data.dtype).max)
            data = data.astype(np.float32) / scale
        else:
            data = data.astype(np.float32)
    return np.clip(np.moveaxis(data, 0, -1), 0, 1)
