"""
Train a crop disease classification model using transfer learning with MobileNetV2.
Uses data augmentation to handle limited training data.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_disease_model.h5")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "class_labels.json")


def clean_dataset(dataset_dir):
    """Remove corrupted or non-image files from dataset."""
    print("Cleaning dataset...")
    removed = 0
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            if not os.path.isfile(filepath):
                continue

            # Check extension
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                os.remove(filepath)
                removed += 1
                continue

            # Try to open and verify the image
            try:
                img = tf.io.read_file(filepath)
                img = tf.image.decode_image(img, channels=3)
                if img.shape[0] < 10 or img.shape[1] < 10:
                    os.remove(filepath)
                    removed += 1
            except Exception:
                os.remove(filepath)
                removed += 1

    print(f"  Removed {removed} invalid files")


def create_model(num_classes):
    """Create MobileNetV2 transfer learning model."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze base model layers
    base_model.trainable = False

    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


def train():
    """Train the crop disease classification model."""
    print("=" * 60)
    print("  CROP DISEASE MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(DATASET_DIR):
        print("ERROR: Dataset directory not found. Run download_images.py first.")
        return

    # Clean dataset
    clean_dataset(DATASET_DIR)

    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    print("\nLoading training data...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    print("\nLoading validation data...")
    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    num_classes = len(train_generator.class_indices)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")

    # Save class labels
    class_labels = {str(v): k for k, v in train_generator.class_indices.items()}

    # Create human-readable labels
    readable_labels = {}
    for idx, class_name in class_labels.items():
        crop = class_name.split("_")[0]
        disease = " ".join(class_name.split("_")[1:])
        readable_labels[idx] = {
            "class_name": class_name,
            "crop": crop,
            "disease": disease if disease != "Healthy" else "Healthy",
        }

    with open(LABELS_PATH, 'w') as f:
        json.dump(readable_labels, f, indent=2)
    print(f"\nClass labels saved to: {LABELS_PATH}")

    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Create model
    print("\nCreating model...")
    model, base_model = create_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Phase 1: Train with frozen base
    print("\n" + "=" * 60)
    print("  Phase 1: Training classification head (base frozen)")
    print("=" * 60)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS // 2,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Phase 2: Fine-tune top layers of base model
    print("\n" + "=" * 60)
    print("  Phase 2: Fine-tuning top layers")
    print("=" * 60)

    base_model.trainable = True
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        initial_epoch=EPOCHS // 2,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    print(f"\n  Final Validation Accuracy: {val_acc:.4f}")
    print(f"  Final Validation Loss:     {val_loss:.4f}")
    print(f"\n  Model saved to: {MODEL_PATH}")
    print(f"  Labels saved to: {LABELS_PATH}")


if __name__ == "__main__":
    train()
