# CUSTOM_CNN_FPGA
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
