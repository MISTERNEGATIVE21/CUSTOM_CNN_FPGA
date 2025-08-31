import gradio as gr
import torch
import torchvision.transforms as transforms
import onnxruntime
import numpy as np
from PIL import Image
import os
import operator

# --- Model & Preprocessing Configurations ---

# Define image sizes for different architectures
MODEL_CONFIGS = {
    'resnet': {'size': (224, 224)},
    'googlenet': {'size': (299, 299)},
    'alexnet': {'size': (227, 227)},
    'densenet': {'size': (224, 224)}
}

# --- Utility Functions ---

def get_available_models():
    """Find all available .pt, .pth, and .onnx model files."""
    models = {}
    for file in os.listdir('.'):
        if file.endswith(('.pt', '.pth', '.onnx')) and 'classification' in file:
            # Extract a clean model name from the filename
            model_name = file.replace('plant_disease_classification_', '').rsplit('.', 1)[0].lower()
            models[model_name] = file
    return models

def get_image_size(model_name):
    """Get the appropriate image size for the selected model from its architecture name."""
    model_name_key = model_name.split('_')[0].lower() # e.g., 'resnet_v1' -> 'resnet'
    return MODEL_CONFIGS.get(model_name_key, {'size': (224, 224)})['size']

# --- Class Label Setup ---

# Assumes your class labels are the names of subdirectories in 'images/images'
try:
    IMAGE_DIR = 'images/images'
    classes = sorted([d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))])
    idx_to_class = {i: cls for i, cls in enumerate(classes)}
except FileNotFoundError:
    print(f"Warning: Directory '{IMAGE_DIR}' not found. Using placeholder classes.")
    # Create placeholder classes if the directory doesn't exist
    classes = [f"Class_{i}" for i in range(38)] # Assuming 38 classes like the PlantVillage dataset
    idx_to_class = {i: cls for i, cls in enumerate(classes)}


# --- Model Loading & Caching ---

# Global cache for models and ONNX sessions
model_cache = {}
# Set device for PyTorch models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model_or_session(model_name, model_path):
    """Load a PyTorch model or an ONNX session if not already in cache."""
    if model_name not in model_cache:
        print(f"Loading {model_name}...")
        if model_path.endswith(('.pt', '.pth')):
            try:
                model = torch.load(model_path, map_location=device)
                model.eval()  # Set model to evaluation mode
                model_cache[model_name] = model
            except Exception as e:
                print(f"Error loading PyTorch model {model_path}: {e}")
                return None
        elif model_path.endswith('.onnx'):
            try:
                session = onnxruntime.InferenceSession(model_path)
                model_cache[model_name] = session
            except Exception as e:
                print(f"Error loading ONNX session {model_path}: {e}")
                return None
    return model_cache.get(model_name)

# --- Prediction Function ---

def predict(image, model_choice):
    """Predict the class of a plant image using the selected model."""
    available_models = get_available_models()
    if model_choice not in available_models:
        return "Selected model is not available. Please check the file names."

    model_path = available_models[model_choice]
    model_or_session = load_model_or_session(model_choice, model_path)

    if model_or_session is None:
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
            ort_inputs = {input_name: img_tensor.numpy()}
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

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Upload Plant Image"),
        gr.Dropdown(choices=model_choices, value=model_choices[0] if model_choices else None, label="Select Model")
    ],
    outputs=gr.Textbox(label="Top 3 Predictions"),
    title="🌿 Plant Disease Detection (PyTorch/ONNX)",
    description="Upload a plant image and select a model to detect diseases. The system supports PyTorch (.pt) and ONNX (.onnx) models.",
    examples=examples,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    iface.launch()
