"""
Dataset discovery and Keras ImageDataGenerator setup.

Expects a dataset root (default: ./data) containing at least a `train/`
and a `test/` folder, each with `NORMAL/` and `PNEUMONIA/` sub-folders,
e.g.:

    data/
      train/
        NORMAL/
        PNEUMONIA/
      val/                <- optional, will be created from train/ if missing/too small
        NORMAL/
        PNEUMONIA/
      test/
        NORMAL/
        PNEUMONIA/
"""
import os
import random
import shutil

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from . import config


def find_split_dirs(root):
    """Walk `root` and return {split_name: path} for every directory that
    directly contains both a NORMAL and a PNEUMONIA sub-folder."""
    found = {}
    for dirpath, dirnames, _ in os.walk(root):
        upper = [d.upper() for d in dirnames]
        if "NORMAL" in upper and "PNEUMONIA" in upper:
            split_name = os.path.basename(dirpath).lower()
            if split_name not in found:
                found[split_name] = dirpath
    return found


def _count_images(split_dir):
    return sum(len(os.listdir(os.path.join(split_dir, c))) for c in config.CLASS_NAMES)


def resolve_split_dirs(data_dir):
    """Locate train/val/test folders under `data_dir`, auto-detecting the layout
    and carving out a validation split from training data if none is found
    (or the provided one is too small to be useful)."""
    splits = find_split_dirs(data_dir)

    train_dir = splits.get("train")
    val_dir = splits.get("val") or splits.get("valid") or splits.get("validation")
    test_dir = splits.get("test")

    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            f"Could not find train/test folders under '{data_dir}'.\n"
            "Expected a structure like:\n"
            "  data/train/NORMAL, data/train/PNEUMONIA\n"
            "  data/test/NORMAL,  data/test/PNEUMONIA\n"
            "  data/val/NORMAL,   data/val/PNEUMONIA   (optional)\n"
            "Upload/extract your dataset into the 'data/' folder with this layout."
        )

    if val_dir is None or _count_images(val_dir) < 100:
        print("No usable validation folder found — creating one from 10% of the training data.")
        val_dir = os.path.join(data_dir, "_val_split")
        if os.path.exists(val_dir):
            shutil.rmtree(val_dir)
        for cls in config.CLASS_NAMES:
            src_dir = os.path.join(train_dir, cls)
            dst_dir = os.path.join(val_dir, cls)
            os.makedirs(dst_dir, exist_ok=True)
            files = sorted(os.listdir(src_dir))
            random.Random(config.SEED).shuffle(files)
            n_val = max(1, int(0.10 * len(files)))
            for fname in files[:n_val]:
                shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    return train_dir, val_dir, test_dir


def get_generators(data_dir=config.DATA_DIR, img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE):
    """Build train/val/test Keras generators plus class weights for the imbalance."""
    train_dir, val_dir, test_dir = resolve_split_dirs(data_dir)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode="binary", classes=config.CLASS_NAMES, shuffle=True, seed=config.SEED,
    )
    val_gen = eval_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size,
        class_mode="binary", classes=config.CLASS_NAMES, shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size,
        class_mode="binary", classes=config.CLASS_NAMES, shuffle=False,
    )

    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_gen.classes), y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights_arr))

    return train_gen, val_gen, test_gen, class_weights
