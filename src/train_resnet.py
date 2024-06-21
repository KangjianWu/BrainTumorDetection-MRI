import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from resnet_model import build_resnet_model
import matplotlib.pyplot as plt

train_images = np.load('../data/processed/train_images.npy')
train_labels = np.load('../data/processed/train_labels.npy')
test_images = np.load('../data/processed/test_images.npy')
test_labels = np.load('../data/processed/test_labels.npy')

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Build and train the model
input_shape = (224, 224, 3)
num_classes = 4  # four types：glioma, meningioma, notumor, pituitary
model = build_resnet_model(input_shape, num_classes)

# Setting training parameters
epochs = 10
batch_size = 32

# Train the model and save the history
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

# Save the trained model
model.save('../models/resnet/resnet_model.h5')

# Preservation of training history
np.save('../models/resnet/resnet_history.npy', history.history)

# Visualization Training History
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('models/resnet/accuracy.png')

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('models/resnet/loss.png')
