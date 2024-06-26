import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from unet_model import build_unet_model
import matplotlib.pyplot as plt

# Load preprocessed data
train_images = np.load('../data/processed/train_images.npy')
train_masks = np.load('../data/processed/train_masks')  # The generated split mask needs to be provided
test_images = np.load('../data/processed/test_images.npy')
test_masks = np.load('../data/processed/test_masks.npy')

# Divide the training set and validation set
X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)

# Build and train the model
input_shape = (224, 224, 3)
model = build_unet_model(input_shape)

# Setting training parameters
epochs = 10
batch_size = 32

# Train the model and save the history
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

# Save the trained model, using the Keras native format
model.save('models/unet/unet_model.keras')

# Preservation of training history
history_path = 'models/unet/unet_history.npy'
if not os.path.exists(os.path.dirname(history_path)):
    os.makedirs(os.path.dirname(history_path))
np.save(history_path, history.history)

# Visualization Training History
if not os.path.exists('models/unet'):
    os.makedirs('models/unet')

# Plot and save accuracy images
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('models/unet/accuracy.png')
plt.close()

# Draw and save loss images
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('models/unet/loss.png')
plt.close()
