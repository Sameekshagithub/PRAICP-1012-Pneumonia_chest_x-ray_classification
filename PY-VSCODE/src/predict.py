"""
Inference helpers used by both the Streamlit app and any ad-hoc scripts.
"""
import numpy as np

from . import config


def preprocess_image(pil_image, img_size=config.IMG_SIZE):
    """Resize/convert a PIL image into a model-ready batch of shape (1, H, W, 3)."""
    img = pil_image.convert("RGB").resize(img_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, pil_image, img_size=config.IMG_SIZE):
    """Run the model on a single PIL image.

    Returns:
        label (str): "NORMAL" or "PNEUMONIA"
        confidence (float): probability of the predicted label, in [0.5, 1.0]
        prob (float): raw model output = P(PNEUMONIA)
        batch (np.ndarray): the preprocessed input batch (reusable for Grad-CAM)
    """
    batch = preprocess_image(pil_image, img_size)
    prob = float(model.predict(batch, verbose=0)[0][0])
    label = config.CLASS_NAMES[1] if prob >= 0.5 else config.CLASS_NAMES[0]
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, confidence, prob, batch
