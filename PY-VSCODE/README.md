# 🫁 Pneumo-Scan AI — Pneumonia Chest X-Ray Classification

A complete, VS Code–ready Python project for training CNN-based pneumonia
classifiers on chest X-ray images, with a dark-themed **Streamlit** web app
(styled like a radiology lightbox) for interactive diagnosis, including
**Grad-CAM** attention-map visualisation.

```
pneumonia_classifier/
├── app.py                # Streamlit frontend (run this to launch the UI)
├── train.py               # CLI training script
├── requirements.txt
├── README.md
├── .gitignore
├── data/                  # <- put your dataset here (see data/README.md)
│   └── README.md
├── models/                # trained .h5 checkpoints get saved here
├── reports/               # metrics.json + confusion matrix / ROC / training-curve plots
└── src/
    ├── config.py          # paths, image size, class names, constants
    ├── data_pipeline.py   # dataset auto-discovery + Keras ImageDataGenerators
    ├── models.py          # Custom CNN, VGG16, MobileNetV2 architectures
    ├── evaluate.py        # metrics, confusion matrix, ROC curve, classification report
    ├── gradcam.py          # Grad-CAM explainability (works for all 3 architectures)
    └── predict.py          # single-image preprocessing + inference
```

---

## 1. Setup

```bash
# From inside the pneumonia_classifier/ folder
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> A GPU is strongly recommended for training (CPU training will work but is slow).

## 2. Add your dataset

Extract/upload your chest X-ray dataset into the `data/` folder so it looks like:

```
data/
  train/
    NORMAL/
    PNEUMONIA/
  val/            (optional — auto-created from train/ if missing or too small)
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

See `data/README.md` for details — the loader auto-detects folders that
contain `NORMAL/` + `PNEUMONIA/` subfolders, so minor naming/nesting
differences are tolerated.

## 3. Train the models

```bash
# Train all three models (Custom CNN, VGG16, MobileNetV2)
python train.py

# Train specific models only
python train.py --models custom
python train.py --models vgg16,mobilenet

# Train + fine-tune the best transfer-learning model afterwards
python train.py --models vgg16,mobilenet --fine-tune

# Common tuning flags
python train.py --epochs 20 --fine-tune-epochs 10 --batch-size 16 --data-dir /path/to/dataset
```

This saves:
- `models/<name>.h5` — trained checkpoints (best epoch, via `ModelCheckpoint`)
- `reports/<name>_confusion_matrix.png`, `_roc_curve.png`, `_training_curves.png`
- `reports/<name>_classification_report.txt`
- `reports/metrics.json` — consolidated comparison table used by the Streamlit app

## 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Open the printed local URL (typically `http://localhost:8501`) in your browser.

**Diagnose tab:** pick a trained model from the sidebar, upload an X-ray
image (JPG/PNG), and get an instant NORMAL / PNEUMONIA verdict with a
confidence meter and an optional Grad-CAM attention-map overlay showing
which region of the image most influenced the prediction.

**Model Performance tab:** a live comparison table (Accuracy / Precision /
Recall / F1 / ROC-AUC) across every model you've trained, plus their
confusion matrix, ROC curve, and training-curve plots.

## 5. Notes / Tips

- Re-running `python train.py` overwrites `reports/metrics.json` and any
  matching `models/<name>.h5` files — rename or back up checkpoints you
  want to keep.
- The app's model dropdown is populated automatically from whatever
  `.h5` files exist in `models/`, so training additional architectures
  (or re-running with different hyperparameters under a new name) will
  show up next time you refresh the app.
- Because pneumonia screening is a medical use case, the model comparison
  is sorted by **Recall** (catching true positive cases) rather than raw
  accuracy — missing a real pneumonia case is far costlier than a false
  alarm.
- This project is for educational/demonstration purposes only and is
  **not** a certified medical device.
