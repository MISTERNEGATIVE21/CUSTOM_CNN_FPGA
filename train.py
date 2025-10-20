import os
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
from tqdm import tqdm # Import tqdm for progress bars

# --- Helper Functions ---
def get_epochs():
    """Asks the user for a valid number of epochs."""
    while True:
        try:
            epochs = int(input("Please enter the number of epochs to train for: "))
            if epochs > 0:
                return epochs
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

def select_device():
    """Asks the user to select a device if CUDA is available."""
    import torch
    if not torch.cuda.is_available():
        print("\nCUDA not available. Defaulting to CPU.")
        return torch.device("cpu")

    while True:
        print("\n--- Select Training Device ---")
        print("1. GPU (cuda)")
        print("2. CPU")
        choice = input("Enter your choice (1 or 2): ")
        if choice == '1':
            return torch.device("cuda")
        elif choice == '2':
            return torch.device("cpu")
        else:
            print("Invalid choice. Please enter 1 or 2.")

def plot_and_save_history(history, epochs, framework="keras"):
    """Plots and saves the training history."""
    if framework.lower() == "keras":
        acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
        loss, val_loss = history.history['loss'], history.history['val_loss']
    else: # PyTorch
        acc, val_acc = history['accuracy'], history['val_accuracy']
        loss, val_loss = history['loss'], history['val_loss']

    epochs_range = range(epochs)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'ro-', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    output_filename = f'training_history_{framework}_{epochs}_epochs.png'
    plt.savefig(output_filename)
    plt.close()
    print(f"\n📈 Plot saved to {output_filename}")

# ==============================================================================
# KERAS / TENSORFLOW WORKFLOW
# ==============================================================================
def run_keras_workflow():
    # This workflow remains unchanged
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from qkeras import QConv2D, QDense, QActivation, quantized_bits

    print("\n--- Keras/TensorFlow Workflow Selected ---")
    IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE = 128, 128, 32
    from pathlib import Path
    DATA_DIR = str(Path('images').resolve())
    
    if not Path(DATA_DIR).exists():
        print(f"⚠️  Images directory not found at {DATA_DIR}")
        print("Please ensure the 'images' directory exists with class subdirectories.")
        return

    print(f"--- Loading Data from {DATA_DIR} ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
    )
    num_classes = len(train_ds.class_names)
    print(f"Found {num_classes} classes.")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    quantizer = quantized_bits(bits=8, integer=0, alpha=1)
    model = models.Sequential([
        tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Rescaling(1./255),
        QConv2D(32, (3, 3), padding='same', kernel_quantizer=quantizer, bias_quantizer=quantizer),
        QActivation('quantized_relu(8)'), layers.MaxPooling2D(),
        QConv2D(64, (3, 3), padding='same', kernel_quantizer=quantizer, bias_quantizer=quantizer),
        QActivation('quantized_relu(8)'), layers.MaxPooling2D(),
        QConv2D(128, (3, 3), padding='same', kernel_quantizer=quantizer, bias_quantizer=quantizer),
        QActivation('quantized_relu(8)'), layers.MaxPooling2D(),
        layers.Dropout(0.25), layers.Flatten(),
        QDense(256, kernel_quantizer=quantizer, bias_quantizer=quantizer),
        QActivation('quantized_relu(8)'), layers.Dropout(0.5),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.summary()
    epochs = get_epochs()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    model_filename = 'trained_keras_model.h5'
    model.save(model_filename)
    print(f"\n--- 💾 Model saved to {model_filename} ---")
    plot_and_save_history(history, epochs, framework="keras")

# ==============================================================================
# PYTORCH WORKFLOW
# ==============================================================================
def run_pytorch_workflow():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import brevitas.nn as qnn
    from brevitas.export import export_qonnx

    # --- MODIFICATION: Import QONNX libraries ---
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.gemm_to_matmul import GemmToMatMul
    from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
    import qonnx.util.cleanup

    # --- MODIFICATION: Import hls4ml for later use ---
    import hls4ml

    print("\n--- PyTorch/Brevitas Workflow Selected ---")

    device = select_device()
    print(f"Using device: {device}")

    IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE = 128, 128, 32
    DATA_DIR = './images'
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    num_classes = len(full_dataset.classes)
    print(f"Found {num_classes} classes.")

    class DeeperQuantCNN(nn.Module):
        # (Model definition remains the same)
        def __init__(self, num_classes):
            super(DeeperQuantCNN, self).__init__()
            self.quant_inp = qnn.QuantIdentity(bit_width=8)
            self.features = nn.Sequential(
                qnn.QuantConv2d(3, 32, kernel_size=3, padding=1, weight_bit_width=8),
                qnn.QuantReLU(bit_width=8), nn.MaxPool2d(2, 2),
                qnn.QuantConv2d(32, 64, kernel_size=3, padding=1, weight_bit_width=8),
                qnn.QuantReLU(bit_width=8), nn.MaxPool2d(2, 2),
                qnn.QuantConv2d(64, 128, kernel_size=3, padding=1, weight_bit_width=8),
                qnn.QuantReLU(bit_width=8), nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.25), nn.Flatten(),
                qnn.QuantLinear(self._get_flat_features(), 256, bias=True, weight_bit_width=8),
                qnn.QuantReLU(bit_width=8), nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        def _get_flat_features(self):
            with torch.no_grad():
                return self.features(torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)).flatten(1).shape[1]
        def forward(self, x):
            return self.classifier(self.features(self.quant_inp(x)))

    model = DeeperQuantCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = get_epochs()
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    print(f"--- Starting Training for {epochs} Epochs ---")
    # (Training loop remains the same)
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_pbar.set_postfix({'Loss': running_loss/total, 'Acc': 100*correct/total})
        history['loss'].append(running_loss / len(train_loader.dataset))
        history['accuracy'].append(correct / total)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'Val Loss': val_loss/val_total, 'Val Acc': 100*val_correct/val_total})
        history['val_loss'].append(val_loss / len(val_loader.dataset))
        history['val_accuracy'].append(val_correct / val_total)

    model_filename = 'trained_pytorch_model.onnx'
    dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
    print(f"\n--- Exporting model to {model_filename} ---")
    export_qonnx(model, dummy_input, model_filename)

    # --- MODIFICATION: Prepare the QONNX model for hls4ml using the qonnx package ---
    print("\n--- Preparing QONNX model with cleanup and transformations ---")
    qonnx_model = ModelWrapper(model_filename)
    qonnx_model = qonnx.util.cleanup.cleanup_model(qonnx_model)
    print("Step 1/4: Initial cleanup complete.")
    qonnx_model = qonnx_model.transform(ConvertToChannelsLastAndClean())
    print("Step 2/4: Channels-last conversion complete.")
    qonnx_model = qonnx_model.transform(GemmToMatMul())
    print("Step 3/4: Gemm to MatMul conversion complete.")
    qonnx_model = qonnx.util.cleanup.cleanup_model(qonnx_model)
    print("Step 4/4: Final cleanup complete.")

    cleaned_model_filename = model_filename.replace('.onnx', '_cleaned.onnx')
    qonnx_model.save(cleaned_model_filename)
    print(f"--- 💾 Cleaned and prepared QONNX model saved to: {cleaned_model_filename} ---")

    # --- MODIFICATION: Create hls4ml config from the prepared QONNX model ---
    print("\n--- Creating hls4ml configuration from the prepared model ---")
    config = hls4ml.utils.config_from_onnx_model(
        qonnx_model, granularity='name', default_precision='ap_fixed<16,6>'
    )
    print("hls4ml configuration created successfully.")
    # Here you could print or modify the config dictionary if needed
    # pprint.pprint(config)

    # --- MODIFICATION: Convert the prepared QONNX model to an HLS project ---
    print("\n--- Converting prepared QONNX model to HLS project ---")
    hls_model = hls4ml.converters.convert_from_onnx_model(
        qonnx_model,
        output_dir='hls4ml_pytorch_project',
        # You can specify a board here, e.g., part='xczu7ev-ffvc1156-2-e'
        hls_config=config
    )

    print("\n--- Compiling HLS project ---")
    hls_model.compile()
    print("--- HLS project compilation complete! ---")
    print(f"Find your project in the 'hls4ml_pytorch_project' directory.")

    plot_and_save_history(history, epochs, framework="pytorch")


# ==============================================================================
# MAIN MENU
# ==============================================================================
if __name__ == '__main__':
    while True:
        print("\n" + "="*30)
        print("    MODEL TRAINING FRAMEWORK    ")
        print("="*30)
        print("1. Train with Keras/TensorFlow")
        print("2. Train with PyTorch/Brevitas")
        print("3. Exit")
        print("="*30)
        choice = input("Enter your choice (1, 2, or 3): ")
        if choice == '1':
            run_keras_workflow()
            break
        elif choice == '2':
            run_pytorch_workflow()
            break
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
