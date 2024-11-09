import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define constants
IMG_WIDTH = 500
IMG_HEIGHT = 500
CHANNELS = 3
BATCH_SIZE = 32

# Create data generators
# Training generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Rescale pixel values to [0,1]
    shear_range=0.2,         # Random shear transformations
    zoom_range=0.2,          # Random zoom transformations
    horizontal_flip=True      # Random horizontal flips
)

# Validation generator with only rescaling
validation_datagen = ImageDataGenerator(
    rescale=1./255           # Only rescale for validation data
)

# Setup data generators
train_generator = train_datagen.flow_from_directory(
    'Data/train',            # Training directory
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'Data/valid',       # Validation directory
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Print some information about the generators
print(f"Training samples: {train_generator.samples}")        # Should be 1942
print(f"Validation samples: {validation_generator.samples}") # Should be 431
print(f"Number of classes: {train_generator.num_classes}")

# Define the CNN model
model = Sequential([
    # First Convolutional Layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)),
    layers.MaxPooling2D(2, 2),
    
    # Second Convolutional Layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Third Convolutional Layer
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Flatten the output for dense layers
    layers.Flatten(),
    
    # Dense layers
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model and store history
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    batch_size=BATCH_SIZE
)

# Plot training results
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Print final metrics
print("\nFinal Training Accuracy:", history.history['accuracy'][-1])
print("Final Validation Accuracy:", history.history['val_accuracy'][-1])
print("Final Training Loss:", history.history['loss'][-1])
print("Final Validation Loss:", history.history['val_loss'][-1])

def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array = img_array / 255.0  # Normalize
    
    # Get predictions
    predictions = model.predict(img_array)
    
    # Get class labels (assuming they're in alphabetical order as processed by flow_from_directory)
    class_labels = list(train_generator.class_indices.keys())
    
    # Get the predicted class and probability
    predicted_class = class_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    
    # Display results
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}%')
    plt.show()
    
    # Print all class probabilities
    print("\nClass Probabilities:")
    for label, prob in zip(class_labels, predictions[0]):
        print(f"{label}: {prob*100:.2f}%")

def test_multiple_images(image_paths):
    for img_path in image_paths:
        print(f"\nTesting image: {img_path}")
        predict_image(img_path)

# Test images:
test_images = [
    "Data/test/crack/test_crack.jpg",
    "Data/test/missing-head/test_missinghead.jpg",
    "Data/test/paint-off/test_paintoff.jpg"
]
test_multiple_images(test_images)