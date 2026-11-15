# CEAMLS | Post-Disaster Building Damage Assessment AI

Research code reconstructed and organized from the **CEAMLS Complete Student Research Guide** (Morgan State University, June–August 2025). This is a **GitHub-ready starter repository**, not a trained model or a validated assessment system.

## What is included

| Stage | Files | Purpose |
|---|---|---|
| S1 — data | `read_geotiff.py`, `build_pairs.py`, `extract_patches.py`, `split_dataset.py`, `augmentation.py`, `dataset.py` | Create matched pre/post image crops, event-separated dataset |
| S2/S3 — models | `models.py`, `train.py` | Train UNet-derived patch classifier, Siamese ResNet or ViT |
| S5 — evaluation | `metrics.py`, `evaluate_model.py` | Evaluate accuracy, weighted F1, class F1, IoU, confusion matrix |
| S4 — app | `export_onnx.py`, `inference.py`, `app/main.py`, `app/index.html` | Export a trained model and run a simple FastAPI demo |

**Important:** The guide has snippets spread over pages and some non-running examples. The scripts here are **adapted**, not a verbatim transcription. See `docs/ADAPTATIONS.md` for changes and known limitations.

## Setup

Use Python 3.11 (or a compatible Colab runtime), preferably with a GPU for training:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The original guide assumes `ceamls-project/` in Google Drive. In Colab, mount Drive first and set paths below to `/content/drive/MyDrive/ceamls-project/...`. Model training requires xBD data downloaded separately under the applicable dataset terms; **the dataset is not included in this repository**. Your inputs should be a pair of `train/images/*pre_disaster*.tif` and corresponding post images and matching JSON label files under `train/labels/`.

### 1. Build pre/post pairs

```bash
python build_pairs.py --data-dir /path/to/ceamls-project/data --out /path/to/ceamls-project/notebooks/pairs_index.csv
```

### 2. Extract **matched** building patches

```bash
python extract_patches.py --pairs /path/to/ceamls-project/notebooks/pairs_index.csv --patch-dir /path/to/ceamls-project/data/patches --out /path/to/ceamls-project/notebooks/patch_metadata.csv --limit-pairs 3
```

Remove `--limit-pairs 3` once you've verified the crop/labels using a small subset. Full patch extraction can require substantial disk space.

### 3. Split by disaster event

```bash
python split_dataset.py --metadata /path/to/ceamls-project/notebooks/patch_metadata.csv --out-dir /path/to/ceamls-project/notebooks
```

### 4. Train on *training* and evaluate on *validation*

```bash
python train.py --arch unet --metadata-dir /path/to/ceamls-project/notebooks --patch-dir /path/to/ceamls-project/data/patches --output-dir /path/to/ceamls-project/results --epochs 30 --pretrained
# alternative architectures: --arch siamese, --arch vit
```

Siamese requires real pre-disaster patch files. Reduce `--batch-size` for small GPUs. `--arch vit` uses 224 × 224 images; others default to 256 × 256 to limit GPU memory.

### 5. Evaluate once on held-out events and export

```bash
python evaluate_model.py --checkpoint /path/to/ceamls-project/results/best_unet.pt --metadata-dir /path/to/ceamls-project/notebooks --patch-dir /path/to/ceamls-project/data/patches
python export_onnx.py --checkpoint /path/to/ceamls-project/results/best_unet.pt --output results/model.onnx
```

For rigorous research, select hyperparameters on validation events, retrain on the planned split, then evaluate on *test* events once. These scripts do not automate the final retraining or a complete ablation study.

### 6. Run the app (only after export)

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Set `CEAMLS_MODEL_PATH=/path/to/model.onnx` to use a model stored elsewhere. Keep `model.json` beside it. The model is NOT included, and the app cannot classify until you train and export one. Uploaded images should be geographically aligned satellite building crops from the same domain as training; ordinary photos are accepted as files but **not validated for reliable predictions**.

## Tests and caveats

You can check syntax without installing heavy ML dependencies:

```bash
python -m compileall -q .
```

`tests/test_pipeline.py` exercises crop, split and metric behavior when the corresponding dependencies are available. The model training, ONNX export and app have **not** been validated against the complete xBD dataset or a trained checkpoint.

**Life-safety warning:** This research prototype does not replace field inspections or engineering judgment. Its displayed labels are not official ATC-20 placards, clearance decisions, or guarantees of safety.
