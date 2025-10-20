import os
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Model configurations
MODEL_CONFIGS = {
    "resnet": {"size": (224, 224)},
    "googlenet": {"size": (299, 299)},
    "alexnet": {"size": (227, 227)},
    "densenet": {"size": (224, 224)},
}

DEFAULT_IMAGE_SIZE = (224, 224)
MODELS_DIR = Path("models").resolve()
IMAGES_DIR = Path("images").resolve()

def get_available_models() -> dict[str, dict]:
    """Return discovered models keyed by display name with metadata."""
    models: dict[str, dict] = {}
    if not MODELS_DIR.exists():
        print(f"⚠️  Models directory not found at {MODELS_DIR}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return models

    h5_files = list(MODELS_DIR.glob("*.h5"))
    if not h5_files:
        print(f"⚠️  No .h5 files in {MODELS_DIR}")
    
    for path in h5_files:
        display_name = path.stem
        models[display_name] = {
            "path": path.resolve(),
            "size": get_image_size(display_name),
        }
        print(f"✅ Found model: {display_name} at {path}")
    
    return dict(sorted(models.items()))

# Get class labels from training
classes = []
if IMAGES_DIR.exists():
    classes = sorted([d for d in os.listdir(IMAGES_DIR) if os.path.isdir(IMAGES_DIR / d)])
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

# Load available models
available_models = get_available_models()
model_cache: dict[str, tf.keras.Model] = {}

def resolve_config_key(model_name: str) -> str | None:
    model_name = model_name.lower()
    for key in MODEL_CONFIGS:
        if key in model_name:
            return key
    return None


def get_image_size(model_name: str) -> tuple[int, int]:
    """Get the appropriate image size for the selected model."""
    config_key = resolve_config_key(model_name)
    if config_key:
        return MODEL_CONFIGS[config_key]["size"]
    return DEFAULT_IMAGE_SIZE

def load_model(model_name: str) -> tf.keras.Model | None:
    """Load the requested model from disk if not already cached."""
    info = available_models.get(model_name)
    if info is None:
        return None

    model_path = str(info["path"])
    cache_key = model_path
    if cache_key not in model_cache:
        model_cache[cache_key] = tf.keras.models.load_model(model_path)
    return model_cache.get(cache_key)

def predict(image, model_choice):
    """Predict using the selected model"""
    try:
        if model_choice not in available_models:
            return "No valid model selected. Add .h5 files to the models/ directory and refresh."

        # Load the selected model
        model = load_model(model_choice)
        if model is None:
            return "Error loading model"

        # Process image
        img = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image)
        img = img.convert('RGB')
        img = img.resize(available_models[model_choice]["size"])
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        # Make prediction
        preds = model.predict(arr)

        # Get top 3 predictions
        top_3_idx = np.argsort(preds[0])[-3:][::-1]
        results = []

        for idx in top_3_idx:
            class_name = idx_to_class.get(idx, f"class_{idx}")
            confidence = float(preds[0][idx])
            results.append(f"{class_name} (Confidence: {confidence:.2f})")
        
        return "\n".join(results)
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Create examples directory if it doesn't exist
model_choices = list(available_models.keys())
placeholder_choice = "⚠️ No models found in models/"
dropdown_choices = model_choices if model_choices else [placeholder_choice]
initial_value = model_choices[0] if model_choices else placeholder_choice

# Only add examples if valid example images exist
example_path = "examples/healthy_leaf.jpg"
examples = None
if os.path.exists(example_path) and model_choices:
    examples = [[example_path, model_choices[0]]]

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Upload Plant Image"),
        gr.Dropdown(
            choices=dropdown_choices,
            value=initial_value,
            label="Select Model",
            interactive=bool(model_choices),
            info="Place Keras .h5 models in the models/ directory and restart to refresh list."
        )
    ],
    outputs=gr.Textbox(label="Predictions"),
    title="Plant Disease Detection",
    description="Upload a plant image and select a model to detect plant diseases. The system will show the top 3 predictions.",
    examples=examples
)

if __name__ == "__main__":
    if not available_models:
        print("Warning: No model files found. Please ensure the trained models are in the current directory.")
    iface.launch()
