import os
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2

app = Flask(__name__)
CORS(app)

MODEL_URLS = {
    'resnet': 'https://539ml-model.s3.amazonaws.com/resnet_model.keras',
    'unet': 'https://539ml-model.s3.amazonaws.com/unet_model.keras'
}

MODEL_PATHS = {
    'resnet': 'models/resnet/resnet_model.keras',
    'unet': 'models/unet/unet_model.keras'
}

resnet_model = None
unet_model = None


def download_model(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading model from {url} to {path}...")
        response = requests.get(url)
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded model from {url} to {path}")
    else:
        print(f"Model already exists at {path}")

    # 检查文件是否存在并且大小合理
    if os.path.exists(path):
        file_size = os.path.getsize(path)
        print(f"File {path} exists, size: {file_size} bytes")
    else:
        print(f"File {path} does not exist after download")


try:
    print("Starting model download...")
    download_model(MODEL_URLS['resnet'], MODEL_PATHS['resnet'])
    download_model(MODEL_URLS['unet'], MODEL_PATHS['unet'])

    print("Loading models...")
    resnet_model = tf.keras.models.load_model(MODEL_PATHS['resnet'])
    unet_model = tf.keras.models.load_model(MODEL_PATHS['unet'])
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

label_map = ['glioma', 'meningioma', 'notumor', 'pituitary']


def prepare_image_for_resnet(image, target):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def prepare_image_for_unet(image, target):
    image = cv2.resize(image, target)
    if len(image.shape) == 2:  # 检查是否是单通道图像
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # 将灰度图像转换为RGB
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0
    return image


def calculate_tumor_area(mask):
    return np.sum(mask > 0)


def calculate_relative_size(mask, original_image):
    brain_area = original_image.size  # 假设大脑区域占据整个图像
    tumor_area = np.sum(mask > 0)
    return (tumor_area / brain_area) * 100


def overlay_tumor_contour(original_image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = original_image.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)


@app.route('/predict', methods=['POST'])
def predict():
    global resnet_model, unet_model

    if resnet_model is None or unet_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image = file.read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    image_for_resnet = prepare_image_for_resnet(image, (224, 224))
    preds = resnet_model.predict(image_for_resnet)
    classification = label_map[np.argmax(preds)]

    if classification != 'notumor':
        image_for_unet = prepare_image_for_unet(image, (224, 224))
        mask = unet_model.predict(image_for_unet)
        mask = (mask > 0.5).astype(np.uint8)
        mask = mask[0, :, :, 0]

        # Resize the mask to the original image size
        original_size_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask_path = os.path.join('uploads', f'mask_{file.filename}.png')
        cv2.imwrite(mask_path, original_size_mask * 255)

        # Calculate additional information using the original size mask
        tumor_area = calculate_tumor_area(original_size_mask)
        relative_size = calculate_relative_size(original_size_mask, image)

        return jsonify(
            prediction=classification,
            mask_path=mask_path,
            tumor_area=float(tumor_area),
            relative_size=float(relative_size)
        )
    else:
        return jsonify(prediction=classification)


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    global resnet_model, unet_model

    if resnet_model is None or unet_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400

    files = request.files.getlist('files')
    results = []

    for file in files:
        if file.filename == '':
            continue

        image = file.read()
        image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        if image is None:
            results.append({"filename": file.filename, "error": "Invalid image format"})
            continue

        original_path = os.path.join('uploads', file.filename)
        cv2.imwrite(original_path, image)

        image_for_resnet = prepare_image_for_resnet(image, (224, 224))
        preds = resnet_model.predict(image_for_resnet)
        classification = label_map[np.argmax(preds)]

        result = {"filename": file.filename, "prediction": classification}

        if classification != 'notumor':
            image_for_unet = prepare_image_for_unet(image, (224, 224))
            mask = unet_model.predict(image_for_unet)
            mask = (mask > 0.5).astype(np.uint8)
            mask = mask[0, :, :, 0]

            original_size_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            tumor_area = calculate_tumor_area(original_size_mask)
            relative_size = calculate_relative_size(original_size_mask, image)

            mask_path = os.path.join('uploads', f'mask_{file.filename}')
            cv2.imwrite(mask_path, original_size_mask * 255)

            contour_overlay = overlay_tumor_contour(image, original_size_mask)
            overlay_path = os.path.join('uploads', f'overlay_{file.filename}')
            cv2.imwrite(overlay_path, contour_overlay)

            result.update({
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "tumor_area": float(tumor_area),
                "relative_size": float(relative_size)
            })

        results.append(result)

    return jsonify(results=results)


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
