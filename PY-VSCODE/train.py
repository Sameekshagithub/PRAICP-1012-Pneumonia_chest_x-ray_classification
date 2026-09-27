#!/usr/bin/env python3
"""
Command-line training script for the Pneumonia Chest X-Ray Classification project.

Usage examples
--------------
Train every model on data/ and save results:
    python train.py

Train only the custom CNN:
    python train.py --models custom

Train the transfer-learning models and fine-tune the best one afterwards:
    python train.py --models vgg16,mobilenet --fine-tune

Point at a dataset uploaded somewhere else on disk:
    python train.py --data-dir /path/to/your/chest_xray
"""
import argparse
import datetime
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from src import config
from src.data_pipeline import get_generators
from src.evaluate import (evaluate_model, plot_confusion_matrix, plot_roc_curve,
                           save_classification_report)
from src.models import MODEL_BUILDERS


def parse_args():
    p = argparse.ArgumentParser(description="Train pneumonia chest X-ray classification models.")
    p.add_argument("--data-dir", default=config.DATA_DIR,
                   help="Path to dataset root (must contain train/ and test/ folders "
                        "with NORMAL/PNEUMONIA subfolders).")
    p.add_argument("--models", default="all",
                   help="Comma-separated list of models to train: custom,vgg16,mobilenet, or 'all'.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--fine-tune-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--fine-tune", action="store_true",
                   help="After initial training, fine-tune the best transfer-learning model.")
    return p.parse_args()


def _save_training_curves(history, name):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title(f"{name} — Accuracy"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title(f"{name} — Loss"); plt.xlabel("Epoch"); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, f"{name}_training_curves.png"))
    plt.close()


def train_one(name, builder_fn, train_gen, val_gen, test_gen, class_weights, args):
    print(f"\n=== Training: {name} ===")
    build_result = builder_fn()
    model, base_model = build_result if isinstance(build_result, tuple) else (build_result, None)

    steps_per_epoch = max(1, train_gen.samples // args.batch_size)
    validation_steps = max(1, val_gen.samples // args.batch_size)
    test_steps = -(-test_gen.samples // args.batch_size)  # ceiling division

    ckpt_path = os.path.join(config.MODEL_DIR, f"{name}.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        train_gen, steps_per_epoch=steps_per_epoch,
        validation_data=val_gen, validation_steps=validation_steps,
        epochs=args.epochs, class_weight=class_weights, callbacks=callbacks,
    )

    metrics, y_true, y_pred, y_prob = evaluate_model(model, test_gen, test_steps)
    print(f"{name} test metrics: {metrics}")

    plot_confusion_matrix(y_true, y_pred, os.path.join(config.REPORTS_DIR, f"{name}_confusion_matrix.png"))
    plot_roc_curve(y_true, y_prob, os.path.join(config.REPORTS_DIR, f"{name}_roc_curve.png"))
    save_classification_report(y_true, y_pred, os.path.join(config.REPORTS_DIR, f"{name}_classification_report.txt"))
    _save_training_curves(history, name)

    return model, base_model, metrics, ckpt_path


def fine_tune(name, model, base_model, train_gen, val_gen, test_gen, args):
    print(f"\n=== Fine-tuning: {name} ===")
    base_model.trainable = True
    fine_tune_at = max(0, len(base_model.layers) - 30)  # keep earlier, generic layers frozen
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss="binary_crossentropy", metrics=["accuracy"])

    steps_per_epoch = max(1, train_gen.samples // args.batch_size)
    validation_steps = max(1, val_gen.samples // args.batch_size)
    test_steps = -(-test_gen.samples // args.batch_size)

    ckpt_path = os.path.join(config.MODEL_DIR, f"{name}_finetuned.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    ]
    model.fit(
        train_gen, steps_per_epoch=steps_per_epoch,
        validation_data=val_gen, validation_steps=validation_steps,
        epochs=args.fine_tune_epochs, callbacks=callbacks,
    )

    metrics, y_true, y_pred, y_prob = evaluate_model(model, test_gen, test_steps)
    print(f"{name}_finetuned test metrics: {metrics}")

    fname = f"{name}_finetuned"
    plot_confusion_matrix(y_true, y_pred, os.path.join(config.REPORTS_DIR, f"{fname}_confusion_matrix.png"))
    plot_roc_curve(y_true, y_prob, os.path.join(config.REPORTS_DIR, f"{fname}_roc_curve.png"))
    save_classification_report(y_true, y_pred, os.path.join(config.REPORTS_DIR, f"{fname}_classification_report.txt"))

    return metrics, ckpt_path


def main():
    args = parse_args()
    requested = list(MODEL_BUILDERS.keys()) if args.models == "all" else [m.strip() for m in args.models.split(",")]

    print("Loading data from:", args.data_dir)
    train_gen, val_gen, test_gen, class_weights = get_generators(args.data_dir, batch_size=args.batch_size)
    print("Detected classes:", train_gen.class_indices)
    print("Computed class weights:", class_weights)

    all_metrics = {}
    trained_models = {}

    for name in requested:
        if name not in MODEL_BUILDERS:
            print(f"Skipping unknown model name: '{name}' (choices: {list(MODEL_BUILDERS)})")
            continue
        model, base_model, metrics, ckpt_path = train_one(
            name, MODEL_BUILDERS[name], train_gen, val_gen, test_gen, class_weights, args
        )
        all_metrics[name] = {**metrics, "model_path": ckpt_path}
        trained_models[name] = (model, base_model)

    if args.fine_tune:
        transfer_names = [n for n in requested if n in ("vgg16", "mobilenet") and n in trained_models]
        if transfer_names:
            best_name = max(transfer_names, key=lambda n: all_metrics[n]["recall"])
            model, base_model = trained_models[best_name]
            ft_metrics, ft_path = fine_tune(best_name, model, base_model, train_gen, val_gen, test_gen, args)
            all_metrics[f"{best_name}_finetuned"] = {**ft_metrics, "model_path": ft_path}
        else:
            print("No transfer-learning model was trained in this run — skipping --fine-tune.")

    with open(config.METRICS_PATH, "w") as f:
        json.dump({"trained_at": datetime.datetime.now().isoformat(), "models": all_metrics}, f, indent=2)

    print("\n=== Final Comparison (sorted by Recall — most important for pneumonia screening) ===")
    for name, m in sorted(all_metrics.items(), key=lambda kv: kv[1]["recall"], reverse=True):
        print(f"{name:24s} | acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
              f"rec={m['recall']:.4f}  f1={m['f1_score']:.4f}  auc={m['roc_auc']:.4f}")

    print(f"\nMetrics saved to : {config.METRICS_PATH}")
    print(f"Models saved to  : {config.MODEL_DIR}")
    print(f"Plots saved to   : {config.REPORTS_DIR}")
    print("\nNext step: run the Streamlit app ->  streamlit run app.py")


if __name__ == "__main__":
    main()
