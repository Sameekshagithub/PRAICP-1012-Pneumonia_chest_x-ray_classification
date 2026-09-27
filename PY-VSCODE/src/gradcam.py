"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for explainability.

Works for both:
  - simple Sequential models (the custom CNN), and
  - Functional models that wrap a nested pretrained backbone
    (VGG16 / MobileNetV2 transfer-learning models), where the backbone
    is itself a Keras Model rather than a single layer.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm


def find_last_conv_layer(m):
    """Return the last Conv2D layer found in a model, searching from the end."""
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in the given model.")


def find_nested_base_model(model):
    """Return the first nested Keras Model found among a model's layers
    (this is how our transfer-learning models wrap VGG16 / MobileNetV2),
    or None if the model has no nested sub-model (e.g. the custom CNN)."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None


def make_gradcam_heatmap(img_array, model, pred_index=None):
    """Compute a Grad-CAM heatmap (values in [0, 1]) for a single preprocessed
    image batch of shape (1, H, W, 3)."""
    base_model = find_nested_base_model(model)

    if base_model is None:
        # --- Simple model path (e.g. the custom CNN) ---
        last_conv_layer = find_last_conv_layer(model)
        grad_model = tf.keras.models.Model(model.inputs, [last_conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img_array)
            idx = 0 if pred_index is None else pred_index
            class_channel = preds[:, idx]
        grads = tape.gradient(class_channel, conv_output)
    else:
        # --- Nested transfer-learning model path ---
        last_conv_layer = find_last_conv_layer(base_model)
        conv_model = tf.keras.models.Model(base_model.input, last_conv_layer.output)
        with tf.GradientTape() as tape:
            conv_output = conv_model(img_array)
            tape.watch(conv_output)
            x = conv_output
            # Manually replay the remaining head layers (GAP -> Dense -> Dropout -> Dense)
            for layer in model.layers:
                if layer is base_model or isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                x = layer(x)
            preds = x
            idx = 0 if pred_index is None else pred_index
            class_channel = preds[:, idx]
        grads = tape.gradient(class_channel, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(pil_image, heatmap, alpha=0.45, colormap="jet"):
    """Blend a Grad-CAM heatmap on top of the original PIL image."""
    img = pil_image.convert("RGB")
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(img.size)
    colored = cm.get_cmap(colormap)(np.asarray(heatmap_resized) / 255.0)[:, :, :3]
    colored_img = Image.fromarray(np.uint8(colored * 255))
    blended = Image.blend(img, colored_img, alpha=alpha)
    return blended
