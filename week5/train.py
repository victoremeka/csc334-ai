import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse

def create_emotion_model(input_shape=(224, 224, 3), num_classes=7, fine_tune=False):
    """
    Create a MobileNet-based emotion detection model
    """
    # Load pre-trained MobileNet without top layers
    base_model = MobileNet(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    if fine_tune:
        # Unfreeze some layers for fine-tuning
        for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers
            layer.trainable = True
    else:
        # Freeze all base model layers
        for layer in base_model.layers:
            layer.trainable = False

    # Add custom classification head with dropout for better generalization
    x = Flatten()(base_model.output)
    x = Dense(units=128, activation='relu')(x)  # Add intermediate layer
    x = Dense(units=num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=base_model.input, outputs=x)

    return model, base_model

def setup_data_generators(train_dir, test_dir, batch_size=32):
    """
    Setup data generators for training and validation
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator

def plot_sample_images(generator, class_names, num_images=10):
    """
    Plot sample images from the generator
    """
    # Get a batch of images
    images, labels = next(generator)

    plt.figure(figsize=(15, 10))

    for i in range(min(num_images, len(images))):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i])

        # Get class name
        class_idx = np.argmax(labels[i])
        class_name = class_names[class_idx]

        plt.title(f'{class_name}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def train_emotion_model(train_dir, test_dir, epochs=50, batch_size=32, save_path="best_model.h5", fine_tune=False):
    """
    Train the emotion detection model
    """
    print("Setting up data generators...")
    train_generator, validation_generator = setup_data_generators(train_dir, test_dir, batch_size)

    # Get class names
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)

    print(f"Found {num_classes} classes: {class_names}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")

    # Create model
    print(f"Creating model (fine_tune={fine_tune})...")
    model, base_model = create_emotion_model(num_classes=num_classes, fine_tune=fine_tune)

    # Compile model with learning rate schedule
    initial_learning_rate = 0.001 if not fine_tune else 0.0001

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    print("Model summary:")
    model.summary()

    # Plot sample images
    print("Plotting sample training images...")
    plot_sample_images(train_generator, class_names)

    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.01,
        patience=10,  # Increased patience
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        filepath=save_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='auto'
    )

    # Add learning rate reduction on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # Calculate steps
    steps_per_epoch = max(1, train_generator.samples // batch_size)
    validation_steps = max(1, validation_generator.samples // batch_size)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    print(f"Training completed! Best model saved to {save_path}")

    # Plot training history
    plot_training_history(history)

    return model, history

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train emotion detection model')
    parser.add_argument('--train-dir', type=str, default='./data/train',
                       help='Path to training data directory')
    parser.add_argument('--test-dir', type=str, default='./data/test',
                       help='Path to test/validation data directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--output', type=str, default='best_model.h5',
                       help='Output path for trained model')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Fine-tune the base model layers (slower but potentially better)')

    args = parser.parse_args()

    # Check if data directories exist
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory '{args.train_dir}' does not exist!")
        print("Please create the directory structure:")
        print(f"  {args.train_dir}/")
        print("    angry/")
        print("    disgust/")
        print("    fear/")
        print("    happy/")
        print("    neutral/")
        print("    sad/")
        print("    surprise/")
        return

    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory '{args.test_dir}' does not exist!")
        return

    # Train the model
    model, history = train_emotion_model(
        args.train_dir,
        args.test_dir,
        args.epochs,
        args.batch_size,
        args.output,
        args.fine_tune
    )

if __name__ == "__main__":
    main()
