# Brain Tumor MRI Segmentation

This project aims to detect and segment brain tumors from MRI images using deep learning models. The project employs a ResNet model for tumor classification and a U-Net model for tumor segmentation.

Due to size limitations, the dataset is not included in this repository. You need to download the dataset manually and place it in the data directory.

Download the brain tumor MRI dataset from Kaggle.
Extract the dataset and place the Training and Testing directories in the data directory as follows:

data
├── processed
├── Testing
│   ├── ...
└── Training
    ├── ...

1. Data Preprocessing
Preprocess the data and generate .npy files for training and testing:
python src/data_preprocessing.py
2.Generate segmentation masks using the pretrained U-Net model:
python src/generate_masks.py
3. Train the ResNet model for classification:
python src/train_resnet.py
4. Train the U-Net model for segmentation:
python src/train_unet.py
Results
After training, the models and training history will be saved in the models directory. The training accuracy and loss plots can be found in models/unet.
