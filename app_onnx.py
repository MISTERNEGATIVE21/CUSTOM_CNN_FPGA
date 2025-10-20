import os
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
    ORT_IMPORT_ERROR: Optional[str] = None
except (ImportError, OSError) as exc:  # OSError catches execstack restrictions
    ort = None  # type: ignore[assignment]
    ORT_AVAILABLE = False
    ORT_IMPORT_ERROR = str(exc)

# --- Model & Preprocessing Configurations ---

# Define image sizes for different architectures
MODEL_CONFIGS = {
    'resnet': {'size': (224, 224)},
    'googlenet': {'size': (299, 299)},
    'alexnet': {'size': (227, 227)},
    'densenet': {'size': (224, 224)}
}

MODELS_DIR = Path("models").resolve()
SUPPORTED_MODEL_EXTENSIONS = {".onnx", ".pt", ".pth"}

# --- Utility Functions ---

def get_available_models() -> dict[str, str]:
    """Discover supported model artifacts inside the models/ directory."""
    models: dict[str, str] = {}
    if not MODELS_DIR.exists():
        print(f"⚠️  Models directory not found at {MODELS_DIR}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return models

    model_files = [p for p in MODELS_DIR.rglob("*") 
                   if p.is_file() and p.suffix.lower() in SUPPORTED_MODEL_EXTENSIONS]
    
    if not model_files:
        print(f"⚠️  No model files with extensions {SUPPORTED_MODEL_EXTENSIONS} in {MODELS_DIR}")
    
    for path in model_files:
        display_name = str(path.relative_to(MODELS_DIR))
        models[display_name] = str(path.resolve())
        print(f"✅ Found model: {display_name} at {path}")

    return dict(sorted(models.items(), key=lambda item: item[0].lower()))


def get_image_size(model_identifier: str) -> tuple[int, int]:
    """Infer expected input size based on filename stem."""
    stem = Path(model_identifier).stem.lower()
    model_name_key = stem.split('_')[0] if '_' in stem else stem
    return MODEL_CONFIGS.get(model_name_key, {'size': (224, 224)})['size']

# --- Class Label Setup ---

# Assumes your class labels are the names of subdirectories in 'images/'
IMAGE_DIR = Path('images').resolve()
try:
    if IMAGE_DIR.exists():
        classes = sorted([d.name for d in IMAGE_DIR.iterdir() if d.is_dir()])
        if not classes:
            print(f"⚠️  No class subdirectories found in {IMAGE_DIR}")
            classes = [f"Class_{i}" for i in range(38)]
        else:
            print(f"✅ Found {len(classes)} classes in {IMAGE_DIR}")
    else:
        print(f"⚠️  Images directory not found at {IMAGE_DIR}. Using placeholder classes.")
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        classes = [f"Class_{i}" for i in range(38)]
    
    idx_to_class = {i: cls for i, cls in enumerate(classes)}
except Exception as e:
    print(f"⚠️  Error loading classes: {e}. Using placeholder classes.")
    classes = [f"Class_{i}" for i in range(38)]
    idx_to_class = {i: cls for i, cls in enumerate(classes)}


# --- Model Loading & Caching ---

# Global cache for models and ONNX sessions
model_cache: dict[str, object] = {}
# Set device for PyTorch models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model_or_session(model_identifier: str, model_path: str):
    """Load a PyTorch model or an ONNX session if not already in cache."""
    cache_key = model_path
    if cache_key not in model_cache:
        print(f"Loading {model_identifier} from {model_path}...")
        if model_path.endswith(('.pt', '.pth')):
            try:
                model = torch.load(model_path, map_location=device)
                model.eval()  # Set model to evaluation mode
                model_cache[cache_key] = model
            except Exception as e:
                print(f"Error loading PyTorch model {model_path}: {e}")
                return None
        elif model_path.endswith('.onnx'):
            if not ORT_AVAILABLE:
                print(f"ONNX Runtime unavailable: {ORT_IMPORT_ERROR}")
                return None
            try:
                session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                model_cache[cache_key] = session
            except Exception as e:
                print(f"Error loading ONNX session {model_path}: {e}")
                return None
    return model_cache.get(cache_key)

# --- Prediction Function ---

def predict(image, model_choice):
    """Predict the class of a plant image using the selected model."""
    available_models = get_available_models()
    if model_choice not in available_models:
        return "Selected model is not available. Please check the file names."

    model_path = available_models[model_choice]
    model_or_session = load_model_or_session(model_choice, model_path)

    if model_or_session is None:
        if model_path.endswith('.onnx') and not ORT_AVAILABLE:
            guidance = (
                "ONNX Runtime isn't available in this environment.\n"
                f"Import error: {ORT_IMPORT_ERROR}\n\n"
                "Try one of the following:\n"
                "• Reinstall ONNX Runtime: `pip install --force-reinstall onnxruntime==1.17.0`\n"
                "• On hardened Linux systems, clear the executable-stack flag on the shared library using `execstack -c <path-to-onnxruntime_pybind11_state.so>`\n"
                "• Alternatively, export the model to PyTorch (.pt/.pth) and load that format."
            )
            return guidance
        return f"Error: Failed to load the model '{model_choice}'."

    try:
        # Define image transformations
        img_size = get_image_size(model_choice)
        preprocess = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Preprocess the input image
        img = Image.fromarray(image).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0) # Add batch dimension

        # --- Inference ---
        if isinstance(model_or_session, torch.nn.Module): # PyTorch model
            with torch.no_grad():
                outputs = model_or_session(img_tensor.to(device))
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                preds = probabilities.cpu().numpy()
        else: # ONNX session
            input_name = model_or_session.get_inputs()[0].name
            ort_input = img_tensor.numpy()
            input_meta = model_or_session.get_inputs()[0]
            input_shape = list(input_meta.shape) if hasattr(input_meta, "shape") else []
            if len(input_shape) == 4:
                def _resolve_dim(val):
                    if isinstance(val, (int, float)):
                        return int(val)
                    if isinstance(val, str) and val.isdigit():
                        return int(val)
                    return None

                dim1 = _resolve_dim(input_shape[1])
                dim3 = _resolve_dim(input_shape[3])
                if dim1 == 3:
                    pass  # channels-first already
                elif dim3 == 3:
                    ort_input = np.transpose(ort_input, (0, 2, 3, 1))
            ort_inputs = {input_name: ort_input}
            ort_outs = model_or_session.run(None, ort_inputs)
            # Apply softmax to logits from ONNX model
            raw_preds = ort_outs[0][0]
            exp_preds = np.exp(raw_preds - np.max(raw_preds))
            preds = exp_preds / exp_preds.sum()

        # Get top 3 predictions
        top_3_idx = np.argsort(preds)[-3:][::-1]

        results = []
        for idx in top_3_idx:
            class_name = idx_to_class.get(idx, "Unknown Class")
            confidence = float(preds[idx])
            results.append(f"{class_name} (Confidence: {confidence:.2f})")

        return "\n".join(results)

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

# --- Gradio UI Setup ---

# Discover models at startup
model_choices = list(get_available_models().keys())
if not model_choices:
    print("Warning: No model files (.pt, .pth, .onnx) found.")
    model_choices = ["No models available"]

# Set up examples if the path exists
example_path = "examples/healthy_leaf.jpg"
examples = [[example_path, model_choices[0]]] if os.path.exists(example_path) and model_choices[0] != "No models available" else None

description = (
    "Upload a plant image and select a model to detect diseases. "
    "The system supports PyTorch (.pt) and ONNX (.onnx) models."
)
if not ORT_AVAILABLE:
    description += (
        "\n\n⚠️ ONNX Runtime couldn't be imported, so ONNX models will be unavailable until "
        "you reinstall the runtime or clear the executable stack restriction."
    )

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Upload Plant Image"),
        gr.Dropdown(choices=model_choices, value=model_choices[0] if model_choices else None, label="Select Model")
    ],
    outputs=gr.Textbox(label="Top 3 Predictions"),
    title="🌿 Plant Disease Detection (PyTorch/ONNX)",
    description=description,
    examples=examples,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    iface.launch()
