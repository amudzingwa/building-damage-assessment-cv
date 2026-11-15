"""Experimental FastAPI classifier. Start with uvicorn app.main:app --reload."""
import os
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

from inference import Predictor

app = FastAPI(title='CEAMLS Damage Screening Research App')
MAX_BYTES = 25 * 1024 * 1024
ALLOWED = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


@lru_cache(maxsize=1)
def model():
    return Predictor(os.environ.get('CEAMLS_MODEL_PATH', 'results/model.onnx'))


async def get_image(upload: UploadFile | None):
    if upload is None:
        return None, None
    suffix = Path(upload.filename or '').suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, 'Use JPG, PNG or GeoTIFF files')
    payload = await upload.read(MAX_BYTES + 1)
    if not payload or len(payload) > MAX_BYTES:
        raise HTTPException(400, 'File must be nonempty and at most 25 MB')
    return payload, suffix


@app.get('/', response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / 'index.html').read_text(encoding='utf-8')


@app.post('/classify')
async def classify(file: UploadFile = File(...), pre_file: UploadFile | None = File(None)):
    post, post_suffix = await get_image(file)
    pre, pre_suffix = await get_image(pre_file)
    try:
        return model().predict(post, post_suffix, pre, pre_suffix)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        # Error should be logged in a production application, without leaking internals.
        raise HTTPException(503, 'Model unavailable; verify ONNX model/config files') from exc
