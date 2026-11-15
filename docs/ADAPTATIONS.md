# Notes on reconstructing code from the CEAMLS guide

The source is a 46-page instructional PDF with snippets embedded in narrative text. These files implement its broad 15-step workflow but deliberately correct several issues:

- **UNet output mismatch (pages 20–21):** A UNet returns segmentation logits `[N,4,H,W]`, whereas the DataLoader supplies a single class label per building `[N]`. The packaged `UNetClassifier` globally pools pixel logits to `[N,4]`. This is a *building patch classifier*, **not** validated semantic segmentation.
- **Pairs can be misaligned (page 9):** Sorting two lists and zipping them can incorrectly match files. `build_pairs.py` matches by paired filenames and verifies matching label JSONs.
- **Siamese input bug (pages 17–18, 23–24):** The PDF DataLoader copies the post-disaster image into `pre`. That destroys the before/after signal. Here, `extract_patches.py` saves corresponding real pre/post patches and Siamese training requires both.
- **Metadata/cropping bug (page 12):** Output records are written for each successfully saved feature, with filenames unique to the pair and feature. Patches are padded where needed.
- **ViT interface (pages 25–27):** Training and evaluation now call one wrapper that returns a `[N,4]` logits tensor. Inputs are sized to 224. Input normalization must also match the pretrained processor; the model wrapper applies ViT's standard 0.5 mean/std transform.
- **ONNX input signature (pages 31–32):** Model exports have architecture-specific input names; Siamese needs both images. The companion JSON records architecture and shape so the service can preprocess consistently.
- **Probabilities and ATC-20 (pages 32, 39–40):** Raw maximum logits are *not* confidence. `inference.py` applies softmax to present model probabilities (which are still uncalibrated). ATC-20 colors cannot be assigned as real safety placards based solely on this prediction, so the app shows nonofficial screening priorities and a safety disclaimer.
- **Dataset format:** The PDF asserts all satellite images have 8 uint16 bands. Real data products vary. The GeoTIFF reader checks band count/dtype and supports 3-band rasters; preprocessing assumptions still require per-dataset verification.
- **Time and compute:** The guide's large patch target, Colab GPU assumptions, and training durations should be treated as estimates, not measured results.

## Limitations

This is a research starter implementation, **not independently field-validated code**. We have not downloaded or redistributed xBD, trained weights, or obtained measured performance. Before publishing scientific claims: confirm data rights; inspect ground-truth alignment and exact dataset band order; run unit/integration tests, train with fixed seeds, compare against meaningful baselines; evaluate by unseen events; report uncertainty and failure modes; obtain expert review for any safety-related deployment.
