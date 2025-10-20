#!/usr/bin/env python3
"""
Advanced Plant Disease Classifier with:
- Multi-model training (6 architectures)
- Quantization (TFLite + ONNX)
- Full evaluation: curves, confusion matrix, reports
- Gradio UI with shareable link
- ONNX export
"""

import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tf2onnx
import onnx
import gradio as gr

from training_monitor import LiveTrainingMonitor, PerformanceAnalyzer

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def configure_compute_device() -> None:
    """Ensure TensorFlow uses CUDA GPUs when available."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as exc:
        print(f"⚠️  Unable to query GPU devices: {exc}")
        return

    if not gpus:
        print("🟡 No NVIDIA GPU detected. Running on CPU.")
        return

    print(f"🟢 Detected {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"⚠️  Could not set memory growth for {gpu.name}: {exc}")
    logical_gpus = tf.config.list_logical_devices("GPU")
    print(f"✅ TensorFlow will run on GPU (logical devices: {len(logical_gpus)})")


def create_alexnet(input_shape, num_classes):
    return Sequential([
        Input(input_shape),
        Conv2D(96, 11, strides=4, activation='relu', padding='same'),
        MaxPooling2D(3, 2),
        Conv2D(256, 5, padding='same', activation='relu'),
        MaxPooling2D(3, 2),
        Conv2D(384, 3, padding='same', activation='relu'),
        Conv2D(384, 3, padding='same', activation='relu'),
        Conv2D(256, 3, padding='same', activation='relu'),
        MaxPooling2D(3, 2),
        GlobalAveragePooling2D(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name="AlexNet")

def create_squeezenet(input_shape, num_classes):
    def fire(x, s, e):
        sq = Conv2D(s, 1, activation='relu', padding='same')(x)
        exp1 = Conv2D(e // 2, 1, activation='relu', padding='same')(sq)
        exp3 = Conv2D(e // 2, 3, activation='relu', padding='same')(sq)
        return concatenate([exp1, exp3])

    inp = Input(input_shape)
    x = Conv2D(64, 3, strides=2, activation='relu', padding='same')(inp)
    x = MaxPooling2D(3, 2)(x)
    x = fire(x, 16, 64)
    x = fire(x, 16, 64)
    x = MaxPooling2D(3, 2)(x)
    x = fire(x, 32, 128)
    x = fire(x, 32, 128)
    x = MaxPooling2D(3, 2)(x)
    x = fire(x, 48, 192)
    x = fire(x, 48, 192)
    x = fire(x, 64, 256)
    x = fire(x, 64, 256)
    x = Dropout(0.5)(x)
    x = Conv2D(num_classes, 1, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)

def create_hybrid_mergenet(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    skip = Conv2D(32, 3, padding='same', activation='relu')(x)
    skip = BatchNormalization()(skip)
    x = Add()([x, skip])
    reduce = Conv2D(16, 1, padding='same', activation='relu')(x)
    x = Concatenate()([x, reduce])
    x = MaxPooling2D(2)(x)

    y = Conv2D(64, 3, padding='same', activation='relu')(x)
    y = BatchNormalization()(y)
    skip2 = Conv2D(64, 3, padding='same', activation='relu')(y)
    skip2 = BatchNormalization()(skip2)
    y = Add()([y, skip2])
    reduce2 = Conv2D(32, 1, padding='same', activation='relu')(y)
    y = Concatenate()([y, reduce2])
    y = MaxPooling2D(2)(y)

    y = GlobalAveragePooling2D()(y)
    y = Dense(128, activation='relu')(y)
    y = Dropout(0.4)(y)
    out = Dense(num_classes, activation='softmax')(y)
    return Model(inp, out, name="HybridMergeNetKeras")

def create_alexnet_lite(input_shape, num_classes):
    return Sequential([
        Input(input_shape),
        Conv2D(64, 7, strides=2, activation='relu', padding='same'),
        MaxPooling2D(3, 2),
        Conv2D(96, 3, padding='same', activation='relu'),
        MaxPooling2D(3, 2),
        Conv2D(128, 3, padding='same', activation='relu'),
        Conv2D(128, 3, padding='same', activation='relu'),
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ], name="AlexNetLite")

def create_squeezenet_lite(input_shape, num_classes):
    def fire(x, squeeze, expand):
        sq = Conv2D(squeeze, 1, activation='relu', padding='same')(x)
        exp1 = Conv2D(expand // 2, 1, activation='relu', padding='same')(sq)
        exp3 = Conv2D(expand // 2, 3, activation='relu', padding='same')(sq)
        return concatenate([exp1, exp3])
    inp = Input(shape=input_shape)
    x = Conv2D(32, 3, strides=2, activation='relu', padding='same')(inp)
    x = MaxPooling2D(3, 2)(x)
    x = fire(x, 8, 64)
    x = fire(x, 8, 64)
    x = MaxPooling2D(3, 2)(x)
    x = fire(x, 16, 96)
    x = fire(x, 16, 96)
    x = Dropout(0.4)(x)
    x = Conv2D(num_classes, 1, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out, name="SqueezeNetLite")

# ----------------------------
# Model Registry
# ----------------------------

MODEL_BUILDERS = {
    'MobileNetV2': lambda ishape, ncls: MobileNetV2(input_shape=ishape, include_top=False, weights='imagenet'),
    'ResNet50': lambda ishape, ncls: ResNet50(input_shape=ishape, include_top=False, weights='imagenet'),
    'EfficientNetB0': lambda ishape, ncls: EfficientNetB0(input_shape=ishape, include_top=False, weights='imagenet'),
    'DenseNet121': lambda ishape, ncls: DenseNet121(input_shape=ishape, include_top=False, weights='imagenet'),
    'AlexNet': create_alexnet,
    'SqueezeNet': create_squeezenet,
    'HybridMergeNet': create_hybrid_mergenet,
    'AlexNetLite': create_alexnet_lite,
    'SqueezeNetLite': create_squeezenet_lite,
}

MODEL_CONFIGS = {
    "default": {"size": (224, 224)},
    "resnet": {"size": (224, 224)},
    "googlenet": {"size": (299, 299)},
    "alexnet": {"size": (227, 227)},
    "densenet": {"size": (224, 224)},
    "mobilenetv2": {"size": (224, 224)},
    "efficientnetb0": {"size": (224, 224)},
    "densenet121": {"size": (224, 224)},
    "alexnetlite": {"size": (224, 224)},
    "squeezenet": {"size": (224, 224)},
    "squeezenetlite": {"size": (224, 224)},
    "hybridmergenet": {"size": (224, 224)},
}

def get_image_size(model_name):
    return MODEL_CONFIGS.get(model_name.lower(), MODEL_CONFIGS['default'])['size']

def get_available_models():
    models = {}
    models_dir = Path("models").resolve()
    if not models_dir.exists():
        print(f"⚠️  Models directory not found at {models_dir}")
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created models directory at {models_dir}")
        return models
    
    h5_files = list(models_dir.glob("*.h5"))
    if not h5_files:
        print(f"⚠️  No .h5 model files found in {models_dir}")
    
    for model_path in h5_files:
        key = model_path.stem.lower()
        display = model_path.stem
        models[display] = {
            "path": model_path.resolve(),
            "size": MODEL_CONFIGS.get(key, MODEL_CONFIGS["default"])["size"],
        }
        print(f"✅ Found model: {display} at {model_path}")
    
    return models


MODEL_CACHE = {}
NO_MODEL_PLACEHOLDER = "❌ No trained models available"


def get_class_names():
    images_dir = Path("images").resolve()
    if not images_dir.exists():
        print(f"⚠️  Images directory not found at {images_dir}")
        return []
    
    class_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"⚠️  No class subdirectories found in {images_dir}")
    
    return sorted([d.name for d in class_dirs])


def load_model_for_inference(model_name, models):
    info = models.get(model_name)
    if info is None:
        return None
    model_path = str(info["path"])
    if model_path not in MODEL_CACHE:
        MODEL_CACHE[model_path] = tf.keras.models.load_model(model_path)
    return MODEL_CACHE[model_path]

def build_model(name, input_shape, num_classes):
    custom_heads = {'AlexNet', 'SqueezeNet', 'HybridMergeNet', 'AlexNetLite', 'SqueezeNetLite'}
    if name in custom_heads:
        return MODEL_BUILDERS[name](input_shape, num_classes)

    base = MODEL_BUILDERS[name](input_shape, num_classes)
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(base.input, out)

# ----------------------------
# Quantization & ONNX
# ----------------------------

def quantize_and_export(model_path, save_dir, model_name):
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    with open(os.path.join(save_dir, f"{model_name}_quant.tflite"), 'wb') as f:
        f.write(tflite_quant_model)

    spec = (tf.TensorSpec((None, *model.input_shape[1:]), tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(onnx_model, os.path.join(save_dir, f"{model_name}.onnx"))

# ----------------------------
# Plotting & Evaluation
# ----------------------------

def plot_training_history(history, model_name, results_dir):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'], label='Train Acc')
    ax[0].plot(history.history['val_accuracy'], label='Val Acc')
    ax[0].set_title(f'{model_name} - Accuracy')
    ax[0].legend()

    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Val Loss')
    ax[1].set_title(f'{model_name} - Loss')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_training.png'))
    plt.close()

def evaluate_model(model, val_ds, class_names, model_name, results_dir):
    y_true, y_pred = [], []
    for x, y in val_ds:
        preds = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        if len(y_true) >= len(val_ds) * val_ds.batch_size:
            break

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    with open(os.path.join(results_dir, f'{model_name}_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(results_dir, f'{model_name}_confusion.png'))
    plt.close()

    return acc

# ----------------------------
# Training Pipeline
# ----------------------------

def train_all_models(img_size=224, epochs=10):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    data_dir = 'images'
    if not os.path.exists(data_dir):
        raise FileNotFoundError("📁 'images' folder not found!")

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_ds = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=32,
        subset='training',
        class_mode='categorical'
    )
    val_ds = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=32,
        subset='validation',
        class_mode='categorical',
        shuffle=False
    )

    class_names = list(train_ds.class_indices.keys())
    num_classes = len(class_names)
    input_shape = (img_size, img_size, 3)

    analyzer = PerformanceAnalyzer("results")
    results = {}
    for name in MODEL_BUILDERS:
        print(f"\n🚀 Training {name}...")
        try:
            model = build_model(name, input_shape, num_classes)
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

            live_monitor = LiveTrainingMonitor(model_name=name, output_dir="results")
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
                live_monitor,
            ]

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )

            # Ensure models directory exists
            models_dir = Path('models').resolve()
            models_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = models_dir / f'{name}.h5'
            model.save(str(model_path))
            print(f"✅ Saved model to {model_path}")
            
            plot_training_history(history, name, 'results')
            acc = evaluate_model(model, val_ds, class_names, name, 'results')
            results[name] = acc
            analyzer.record_model(
                model_name=name,
                history=live_monitor.get_history(),
                val_accuracy=acc,
                epoch_durations=live_monitor.epoch_durations,
            )
            quantize_and_export(str(model_path), str(models_dir), name)

            print(f"✅ {name} - Val Accuracy: {acc:.4f}")
        except Exception as exc:
            print(f"❌ Failed {name}: {exc}")
            results[name] = 0.0

    best = max(results, key=results.get)
    print(f"\n🏆 Best Model: {best} ({results[best]:.4f})")
    
    models_dir = Path('models').resolve()
    best_h5 = models_dir / f'{best}.h5'
    best_tflite = models_dir / f'{best}_quant.tflite'
    best_onnx = models_dir / f'{best}.onnx'
    
    if best_h5.exists():
        shutil.copy(str(best_h5), 'best_model.h5')
        print(f"✅ Copied best model: {best_h5} -> best_model.h5")
    if best_tflite.exists():
        shutil.copy(str(best_tflite), 'best_model_quant.tflite')
    if best_onnx.exists():
        shutil.copy(str(best_onnx), 'best_model.onnx')

    with open('results/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    analyzer.generate_overview_plot()

# ----------------------------
# Gradio UI
# ----------------------------

def predict(image, model_choice=None):
    models = get_available_models()
    if not models:
        return {"Error": "No trained models found. Train models first."}

    if model_choice in (None, "", NO_MODEL_PLACEHOLDER) or model_choice not in models:
        # Fall back to first available model
        model_choice = next(iter(models))

    model = load_model_for_inference(model_choice, models)
    if model is None:
        return {"Error": f"Unable to load model '{model_choice}'."}

    class_names = get_class_names()
    if not class_names:
        return {"Error": "No class folders found in 'images/'."}

    if image is None:
        return {"Error": "Please upload an image."}

    img = tf.convert_to_tensor(image)
    target_size = models[model_choice]["size"]
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, 0)

    preds = model.predict(img, verbose=0)
    if preds.ndim == 2:
        preds = preds[0]
    preds = np.asarray(preds)

    num_classes = min(len(class_names), len(preds))
    if num_classes == 0:
        return {"Error": "Model output does not match available class labels."}

    return {class_names[i]: float(preds[i]) for i in range(num_classes)}


def launch_gradio():
    models = get_available_models()
    model_choices = list(models.keys())
    dropdown_choices = model_choices if model_choices else [NO_MODEL_PLACEHOLDER]
    initial_value = model_choices[0] if model_choices else NO_MODEL_PLACEHOLDER

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="numpy", label="🌿 Upload Plant Leaf Image"),
            gr.Dropdown(
                choices=dropdown_choices,
                value=initial_value,
                label="Select Model",
                interactive=bool(model_choices),
                info="Place trained .h5 models in models/ and restart this app to refresh the list."
            ),
        ],
        outputs=gr.Label(num_top_classes=5, label="Disease Prediction"),
        title="🌱 Plant Disease Classifier (Multi-Model AI)",
        description="Trained on 13 disease classes. Choose any available CNN for inference.",
        examples=None,
        live=False
    )
    print("\n🌐 Launching Gradio with PUBLIC SHARE LINK...")
    demo.launch(share=True)

# ----------------------------
# Main Menu
# ----------------------------

def main():
    configure_compute_device()
    while True:
        print("\n" + "=" * 70)
        print("🌿 Advanced Plant Disease AI: Multi-Model + Quantization + ONNX + Gradio")
        print("=" * 70)
        print("1. Train all models (with full evaluation, quantization, ONNX)")
        print("2. Launch Gradio UI (with public shareable link)")
        print("3. Exit")
        print("-" * 70)
        choice = input("👉 Choose (1-3): ").strip()

        if choice == '1':
            img_size = int(input("🖼️  Image size (224 recommended) [default=224]: ") or "224")
            epochs = int(input("⏳ Epochs [default=10]: ") or "10")
            train_all_models(img_size=img_size, epochs=epochs)
        elif choice == '2':
            launch_gradio()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice.")

if __name__ == '__main__':
    main()