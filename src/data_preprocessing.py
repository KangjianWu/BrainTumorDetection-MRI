import os
import numpy as np
import cv2


def load_data(data_dir):
    labels = []
    images = []

    # 定义标签映射
    label_map = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

    # 遍历每个标签
    for label, value in label_map.items():
        path = os.path.join(data_dir, label)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (224, 224))  # 调整图像大小以适应模型输入
                images.append(image)
                labels.append(value)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def preprocess_data(images, labels):
    images = images.astype('float32') / 255.0
    labels = np.array(labels)

    return images, labels


if __name__ == "__main__":
    train_data_dir = '../data/Training'
    test_data_dir = '../data/Testing'

    train_images, train_labels = load_data(train_data_dir)
    test_images, test_labels = load_data(test_data_dir)

    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)

    # 保存处理后的数据
    if not os.path.exists('../data/processed'):
        os.makedirs('../data/processed')

    np.save('../data/processed/train_images.npy', train_images)
    np.save('../data/processed/train_labels.npy', train_labels)
    np.save('../data/processed/test_images.npy', test_images)
    np.save('../data/processed/test_labels.npy', test_labels)
