# CUSTOM_CNN_FPGA

## 🚀 Quick start with Docker

You can run the full training/export menu from an isolated container. Make sure your dataset is available inside the container (bind-mount the `images/` directory or adjust the path).

```bash
docker build -t custom-cnn-fpga .
docker run --rm -it \
    -p 7860:7860 \
    -v "$PWD/images":/workspace/images \
    -v "$PWD/models":/workspace/models \
    -v "$PWD/results":/workspace/results \
    custom-cnn-fpga
```

The default command launches `python new_train.py`, which presents the interactive menu for Keras, PyTorch, and the advanced multi-model workflow. You can override the command, for example to launch the Gradio UI directly once a model has been trained:

```bash
docker run --rm -it -p 7860:7860 custom-cnn-fpga python advanced_train.py
```

### Script launcher

For a quick way to run any project script locally, use the interactive launcher:

```bash
python script_launcher.py
```

It scans the repository for `.py` files (excluding caches) and lets you pick which one to execute in a loop.

### Netron model viewer

Need a quick way to inspect a trained network? Launch the lightweight Flask helper:

```bash
python netron_viewer.py
```

Upload any supported model file (`.onnx`, `.h5`, `.tflite`, `.pt`, etc.). The helper copies the file into `netron_uploads/`, serves it from the Flask app, and produces a sharable `https://netron.app/?url=...` link so you can explore the model in the hosted Netron UI. If you want to share the link with others, make sure the machine running this script is reachable (or expose the `/models/<id>` route through your own tunnel).

### Optional: GPU acceleration

The provided Dockerfile targets CPU execution. If you have an NVIDIA GPU and want to enable CUDA, start from an `nvidia/cuda` base image and install the matching TensorFlow/PyTorch wheels, then run the container with `--gpus all`.

---

## 📊 Training analytics

`advanced_train.py` now streams live accuracy/loss charts and aggregates results for every architecture trained in one run:

- While each model trains, a live snapshot is written to `results/<ModelName>_live.png` so you can watch convergence without attaching a notebook.
- When training finishes, `results/model_performance_summary.json` captures final metrics and epoch timings for all models.
- A comparison bar chart at `results/model_accuracy_overview.png` highlights the best validation accuracy across the suite.

These files are regenerated on every run; clear the `results/` directory first if you want a clean slate.


import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, ReLU,
                                                                         Add, Concatenate, MaxPooling2D,
                                                                         GlobalAveragePooling2D, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


# 🔹 Define HybridMergeNet Architecture
def hybrid_mergenet(input_shape=(224, 224, 3), num_classes=2):
    inputs = Input(shape=input_shape)

    # First Conv Block
    x = Conv2D(32, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Skip Connection Block
    skip = Conv2D(32, (3, 3), padding="same")(x)
    skip = BatchNormalization()(skip)
    skip = ReLU()(skip)
    x = Add()([x, skip])  # Residual

    # DenseNet-style Concatenation
    reduce = Conv2D(16, (1, 1), padding="same")(x)  # channel reduction
    x = Concatenate()([x, reduce])

    # Downsampling
    x = MaxPooling2D((2, 2))(x)

    # Repeat Block with more filters
    y = Conv2D(64, (3, 3), padding="same")(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    skip2 = Conv2D(64, (3, 3), padding="same")(y)
    skip2 = BatchNormalization()(skip2)
    skip2 = ReLU()(skip2)
    y = Add()([y, skip2])

    reduce2 = Conv2D(32, (1, 1), padding="same")(y)
    y = Concatenate()([y, reduce2])

    y = MaxPooling2D((2, 2))(y)

    # Global Pool + Dense Softmax
    x = GlobalAveragePooling2D()(y)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="HybridMergeNet")


# 🔹 Load Dataset
def load_dataset(img_dir="images", img_size=(224, 224), batch_size=32):
    train_ds = image_dataset_from_directory(
        img_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = image_dataset_from_directory(
        img_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    # Data Augmentation (on-the-fly)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (train_ds
                .map(lambda x, y: (data_augmentation(x), y),
                     num_parallel_calls=AUTOTUNE)
                .cache()
                .shuffle(1000)
                .prefetch(buffer_size=AUTOTUNE))

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


# 🔹 Plot Training History
def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


# 🔹 Main Training Script
if __name__ == "__main__":
    IMG_DIR = "images"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20

    # Load dataset
    train_ds, val_ds = load_dataset(IMG_DIR, IMG_SIZE, BATCH_SIZE)

    # Build model
    model = hybrid_mergenet(input_shape=(224, 224, 3), num_classes=2)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks: save best model + stop early if no improvement
    checkpoint = ModelCheckpoint("best_hybrid_mergenet.h5",
                                 save_best_only=True,
                                 monitor="val_loss",
                                 mode="min")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )

    # Plot results
    plot_training(history)

    # Save final model
    model.save("final_hybrid_mergenet.h5")
