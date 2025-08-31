import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Model configurations
MODEL_CONFIGS = {
    'resnet': {'size': (224, 224)},
    'googlenet': {'size': (299, 299)},
    'alexnet': {'size': (227, 227)},
    'densenet': {'size': (224, 224)}
}

def get_available_models():
    """Find all available .h5 model files"""
    models = {}
    for file in os.listdir('.'):
        if file.endswith('.h5') and 'classification' in file:
            model_name = file.replace('plant_disease_classification_', '').replace('.h5', '').lower()
            models[model_name] = file
    return models

# Get class labels from training
IMAGE_DIR = 'images/images'
classes = sorted([d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))])
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

# Load available models
available_models = get_available_models()
model_cache = {}

def get_image_size(model_name):
    """Get the appropriate image size for the selected model"""
    model_name = model_name.lower()
    return MODEL_CONFIGS.get(model_name, {'size': (224, 224)})['size']

def load_model(model_name):
    """Load model if not already in cache"""
    if model_name not in model_cache and model_name in available_models:
        model_path = available_models[model_name]
        model_cache[model_name] = tf.keras.models.load_model(model_path)
    return model_cache.get(model_name)

def predict(image, model_choice):
    """Predict using the selected model"""
    try:
        if model_choice not in available_models:
            return "Selected model not found"
        
        # Load the selected model
        model = load_model(model_choice)
        if model is None:
            return "Error loading model"
        
        # Process image
        img = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image)
        img = img.convert('RGB')
        img = img.resize(get_image_size(model_choice))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        # Make prediction
        preds = model.predict(arr)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(preds[0])[-3:][::-1]
        results = []
        
        for idx in top_3_idx:
            class_name = idx_to_class[idx]
            confidence = float(preds[0][idx])
            results.append(f"{class_name} (Confidence: {confidence:.2f})")
        
        return "\n".join(results)
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Create examples directory if it doesn't exist
model_choices = list(available_models.keys()) if available_models else ["No models available"]

# Only add examples if valid example images exist
example_path = "examples/healthy_leaf.jpg"
examples = None
if os.path.exists(example_path):
    examples = [[example_path, model_choices[0]]]

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Upload Plant Image"),
        gr.Dropdown(choices=model_choices, value=model_choices[0] if model_choices else None, label="Select Model")
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
