import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

def load_images(image_dir):
    images = []
    image_files = []
    for label_dir in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label_dir)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.resize(image, (224, 224))
                    images.append(image)
                    image_files.append(img_file)
    images = np.array(images)
    return images, image_files

def save_masks(masks, mask_dir, image_files):
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    for idx, mask in enumerate(masks):
        mask_path = os.path.join(mask_dir, f"{image_files[idx]}.png")
        cv2.imwrite(mask_path, mask * 255)  # Save as binary image

def visualize_mask(mask):
    plt.imshow(mask, cmap='gray')
    plt.show()

if __name__ == "__main__":
    image_dir = '../data/Training'
    mask_dir = '../data/processed/train_masks'

    # Loading pre-trained segmentation models
    model = tf.keras.models.load_model('models/pretrained_unet_model.keras')  # Make sure the path is correct

    # Load Image
    images, image_files = load_images(image_dir)

    # Generate Segmentation Mask
    images = images.astype('float32') / 255.0  # Preprocessed images
    masks = model.predict(images)
    masks = (masks > 0.5).astype(np.uint8)  # binarization mask

    # Visual inspection of some generated masks
    for i in range(5):
        visualize_mask(masks[i, :, :, 0])

    # The generated mask file name matches the image file name
    image_files = [f"img_{idx}" for idx in range(len(images))]

    # Save Mask
    save_masks(masks, mask_dir, image_files)
