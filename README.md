# Brain Tumor MRI Segmentation Project

This project aims to develop a system for detecting and segmenting brain tumors from MRI images. The system uses a ResNet model to classify MRI images as having a tumor or not, and a U-Net model to segment the tumor if it is detected. The frontend is implemented using React.


## Download Data

1. Download the brain tumor MRI dataset from Kaggle 'https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data'
2. Create a new folder named `data` in the root directory of the project.
3. Extract the downloaded dataset into the `data` folder.

## Data Preprocessing

Run the `data_preprocessing.py` script to preprocess the data.

## Generate Masks

Run the `unet_model.py` script to build the U-Net model, then generate masks using the `masks.py` script.

## Train ResNet Model

Run the `train_resnet.py` script to train the ResNet model. This script trains the ResNet model on the preprocessed data and saves the trained model in the `models/resnet` folder.

## Train U-Net Model

Run the `train_unet.py` script to train the U-Net model. This script trains the U-Net model on the preprocessed data and saves the trained model in the `models/unet` folder.

## Inference

Run the `inference.py` script to start the inference server. This script starts a Flask server that uses the trained ResNet and U-Net models to classify and segment brain tumors in MRI images.

## Frontend Implementation

To start the React frontend:
1. Open a terminal and navigate to the `react_app` directory.
2. Start the React app using `npm start`.


Ensure that all necessary dependencies are installed, and follow the steps in each section to complete the project setup.
