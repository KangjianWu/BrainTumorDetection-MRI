import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2

app = Flask(__name__)
CORS(app)

try:
    resnet_model = tf.keras.models.load_model('models/resnet/resnet_model.keras')
    unet_model = tf.keras.models.load_model('models/unet/unet_model.keras')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

def prepare_image_for_resnet(image, target):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def prepare_image_for_unet(image, target):
    image = cv2.resize(image, target)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image = file.read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    print(f"Original image shape: {image.shape}")

    image_for_resnet = prepare_image_for_resnet(image, (224, 224))

    print(f"Preprocessed image shape: {image_for_resnet.shape}")

    preds = resnet_model.predict(image_for_resnet)
    label_map = ['glioma', 'meningioma', 'notumor', 'pituitary']
    classification = label_map[np.argmax(preds)]
    print("Predictions:", preds)
    print("ResNet Classification:", classification)

    if classification != 'notumor':
        image_for_unet = prepare_image_for_unet(image, (224, 224))
        mask = unet_model.predict(image_for_unet)
        mask = (mask > 0.5).astype(np.uint8)
        mask = mask[0, :, :, 0]

        mask_path = os.path.join('uploads', f'mask_{file.filename}.png')
        cv2.imwrite(mask_path, mask * 255)

        return jsonify(prediction=classification, mask_path=mask_path)
    else:
        return jsonify(prediction=classification)

@app.route('/')
def index():
    return "Welcome to the Brain Tumor MRI Segmentation API!"

@app.route('/uploads/<filename>')
def send_mask(filename):
    return send_from_directory('uploads', filename)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(host='0.0.0.0', port=5000)
