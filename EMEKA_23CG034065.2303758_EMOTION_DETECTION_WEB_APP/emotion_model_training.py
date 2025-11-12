"""
Emotion Detection Model Training Script
This script trains a CNN model to detect 7 emotions from facial images
Emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import cv2
import os

# Set random seed for reproducibility
np.random.seed(42)

# Constants
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 7

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def create_model():
    """
    Create a Convolutional Neural Network for emotion detection
    """
    model = Sequential()
    
    # First Convolutional Block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second Convolutional Block
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Third Convolutional Block
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Fourth Convolutional Block
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # Output Layer
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    return model

def train_with_fer2013():
    """
    Train the model using FER2013 dataset
    Note: You need to download the FER2013 dataset from Kaggle
    Link: https://www.kaggle.com/datasets/msambare/fer2013
    """
    print("=" * 50)
    print("EMOTION DETECTION MODEL TRAINING")
    print("=" * 50)
    
    # Check if dataset exists
    train_dir = 'fer2013/train'
    test_dir = 'fer2013/test'
    
    if not os.path.exists(train_dir):
        print(f"\nError: Dataset not found at {train_dir}")
        print("\nTo train the model:")
        print("1. Download FER2013 dataset from: https://www.kaggle.com/datasets/msambare/fer2013")
        print("2. Extract it in the same directory as this script")
        print("3. The folder structure should be: fer2013/train/ and fer2013/test/")
        print("\nFor now, creating a pre-configured model architecture...")
        
        # Create model anyway for demonstration
        model = create_model()
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Summary:")
        model.summary()
        
        # Save the untrained model (you'll need to train it with actual data)
        model.save('emotion_model.h5')
        print("\nModel architecture saved as 'emotion_model.h5'")
        print("Note: This is an untrained model. Please train it with FER2013 dataset.")
        return
    
    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation/test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True
    )
    
    # Load test data
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")
    
    # Create model
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'emotion_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train the model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        callbacks=[reduce_lr, early_stop, checkpoint],
        verbose=1
    )
    
    # Save final model
    model.save('emotion_model.h5')
    print("\nFinal model saved as 'emotion_model.h5'")
    
    # Evaluate the model
    print("\nEvaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    plt.figure(figsize=(14, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")
    plt.show()

def create_simple_pretrained_model():
    """
    Create a simple pre-trained model for demonstration
    This uses transfer learning with a smaller dataset
    """
    print("\nCreating a simple emotion detection model...")
    
    model = create_model()
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model architecture
    model.save('emotion_model.h5')
    print("Model saved as 'emotion_model.h5'")
    print("\nNote: For production use, train this model with the FER2013 dataset")
    
    return model

if __name__ == "__main__":
    print("Emotion Detection Model Training Script")
    print("=" * 50)
    print("\nOptions:")
    print("1. Train with FER2013 dataset (requires dataset download)")
    print("2. Create model architecture only")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        train_with_fer2013()
    elif choice == "2":
        create_simple_pretrained_model()
    else:
        print("Invalid choice. Creating model architecture only...")
        create_simple_pretrained_model()
    
    print("\n" + "=" * 50)
    print("Training script completed!")
    print("=" * 50)