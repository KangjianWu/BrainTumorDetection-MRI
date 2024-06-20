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
        cv2.imwrite(mask_path, mask * 255)  # 保存为二值图像

def visualize_mask(mask):
    plt.imshow(mask, cmap='gray')
    plt.show()

if __name__ == "__main__":
    image_dir = '../data/Training'
    mask_dir = '../data/processed/train_masks'

    # 加载预训练的分割模型
    model = tf.keras.models.load_model('models/pretrained_unet_model.keras')  # 确保路径正确

    # 加载图像
    images, image_files = load_images(image_dir)

    # 生成分割掩码
    images = images.astype('float32') / 255.0  # 预处理图像
    masks = model.predict(images)
    masks = (masks > 0.5).astype(np.uint8)  # 二值化掩码

    # 可视化检查一些生成的掩码
    for i in range(5):
        visualize_mask(masks[i, :, :, 0])

    # 生成的掩码文件名与图像文件名一致
    image_files = [f"img_{idx}" for idx in range(len(images))]

    # 保存掩码
    save_masks(masks, mask_dir, image_files)
