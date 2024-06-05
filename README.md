# Automatic Defect Detection in Manufacturing Using VGG

## Authors: Yongchun Chen, Zhuochen Dai

## Overview

This project aims to develop a deep learning model using the VGG16 architecture to automatically detect defects in manufacturing components from images. By leveraging the MVTec Anomaly Detection (AD) dataset, the model is trained to identify various defects such as scratches, dents, and stains, thereby improving the quality control process and reducing the need for manual inspections.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Challenges and Future Work](#challenges-and-future-work)
- [Acknowledgments](#acknowledgments)

## Dataset

The MVTec AD dataset is used in this project. It includes 15 categories of objects and textures, each with defect-free and defective images. The categories are:

- Bottle
- Cable
- Capsule
- Carpet
- Grid
- Hazelnut
- Leather
- Metal Nut
- Pill
- Screw
- Tile
- Toothbrush
- Transistor
- Wood
- Zipper

Each category contains training images (defect-free) and test images (both defect-free and defective), with pixel-precise annotations of anomalous regions.

## Methodology

The VGG16 model pre-trained on ImageNet is fine-tuned for binary classification (defect-free vs. defective). The methodology includes the following steps:

1. **Data Preparation**: Images are resized to 224x224 pixels, normalized, and split into training and testing sets.
2. **Model Development**: The VGG16 model is loaded, and custom layers are added for binary classification.
3. **Model Training**: The model is trained using the training data with early stopping and model checkpointing to prevent overfitting.
4. **Model Evaluation**: The trained model is evaluated on the test set to assess its performance.

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yongchun780/ai_project_vgg.git
   cd ai_project_vgg
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Mount your Google Drive to access the dataset:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

Please note that the dataset and the trained model are too large to be uploaded to the repository. You will need to download the MVTec AD dataset from [MVTec's official website](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place it in your Google Drive.

## Usage

1. **Data Preparation**: Load and preprocess the dataset.
   ```python
   # Example code to load and preprocess the dataset
   base_path = '/content/drive/My Drive/your_project_folder'
   train_images, train_labels, test_images, test_labels = load_bottle_data(base_path)
   ```

2. **Train the Model**: Train the VGG16 model on the dataset.
   ```python
   # Example code to train the model
   history = model.fit(
       train_images, train_labels,
       epochs=50,
       validation_split=0.2,
       batch_size=16,
       callbacks=callbacks
   )
   ```

3. **Evaluate the Model**: Evaluate the trained model on the test set.
   ```python
   # Example code to evaluate the model
   test_loss, test_accuracy = model.evaluate(test_images, test_labels)
   print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
   ```

## Results

The model achieved high accuracy in distinguishing between defect-free and defective components. Detailed results and visualizations can be found in the `Ai_Project.ipynb` notebook and the project report `CyberSquad_Project_Report.pdf`.

## Challenges and Future Work

Despite the promising results, several challenges were encountered:
- **Parameter Tuning**: Finding the optimal hyperparameters remains challenging.
- **Dataset Size and Complexity**: Handling the large and complex dataset required significant computational resources.
- **Class Imbalance**: The imbalance between defect-free and defective samples affected model performance.

Future work will focus on addressing these challenges through hyperparameter optimization, enhanced data processing techniques, and exploring other deep learning architectures.

## Acknowledgments

We would like to thank the authors of the MVTec AD dataset for providing a comprehensive dataset for anomaly detection. We also acknowledge the support from Google Colab for providing the computational resources necessary for this project.
