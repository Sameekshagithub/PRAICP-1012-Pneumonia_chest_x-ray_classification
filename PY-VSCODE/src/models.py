"""
Model architectures: a CNN built from scratch, plus two transfer-learning
models (VGG16 and MobileNetV2) used as frozen feature extractors with a
custom classification head.
"""
import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                                      Flatten, GlobalAveragePooling2D, Input,
                                      MaxPooling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from . import config


def build_custom_cnn(input_shape=config.INPUT_SHAPE):
    """A CNN trained entirely from scratch on the chest X-ray dataset."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Conv2D(64, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Conv2D(128, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Conv2D(128, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ],
        name="custom_cnn",
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _build_transfer(base_fn, name, input_shape=config.INPUT_SHAPE):
    """Wrap a frozen ImageNet-pretrained backbone with a small custom head."""
    base_model = base_fn(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name=name)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model, base_model


def build_vgg16(input_shape=config.INPUT_SHAPE):
    return _build_transfer(VGG16, "vgg16", input_shape)


def build_mobilenet(input_shape=config.INPUT_SHAPE):
    return _build_transfer(MobileNetV2, "mobilenet", input_shape)


# Registry used by train.py's --models CLI flag
MODEL_BUILDERS = {
    "custom": build_custom_cnn,
    "vgg16": build_vgg16,
    "mobilenet": build_mobilenet,
}

# Human-friendly display names, used by the Streamlit app
MODEL_DISPLAY_NAMES = {
    "custom": "Custom CNN (from scratch)",
    "vgg16": "VGG16 (Transfer Learning)",
    "mobilenet": "MobileNetV2 (Transfer Learning)",
    "vgg16_finetuned": "VGG16 (Fine-Tuned)",
    "mobilenet_finetuned": "MobileNetV2 (Fine-Tuned)",
}
