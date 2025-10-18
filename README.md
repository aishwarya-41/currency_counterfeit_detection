# Advanced Counterfeit Currency Detection 
This repository contains the implementation for the project "Advanced Counterfeit Currency Detection," which explores and compares multiple deep learning and machine learning approaches for identifying counterfeit banknotes. The project is structured into three distinct "variants," each representing a unique methodology for tackling this challenge.

# Abstract
Counterfeit currency poses a severe threat to national economies and financial institutions. This project investigates the application of Artificial Intelligence to automate counterfeit detection with high accuracy. We implement and evaluate three different models: an end-to-end attention-enhanced CNN, a transfer learning pipeline with classical machine learning classifiers, and a novel hybrid variant that combines the strengths of both approaches. The models are designed to be robust, deployable, and effective at distinguishing genuine currency from sophisticated forgeries.

# The Three Variants
This project is organized into three separate directories, each containing a self-contained implementation of a specific variant:

- Variant 1: Attention-Enhanced CNN (CBAM): An end-to-end deep learning model that uses a custom Convolutional Neural Network (CNN) enhanced with a Convolutional Block Attention Module (CBAM). This model learns to identify key security features directly from images and is trained on a combination of genuine notes and synthetically generated counterfeits.

- Variant 2: Transfer Learning + ML Classifiers: A two-stage approach that leverages a powerful, pre-trained model (MobileNetV2) as an expert feature extractor. These high-quality features are then used to train a suite of fast and effective classical machine learning models, including SVM, Random Forest, and XGBoost.

- Variant 3: Hybrid Model (MobileNetV2 + CBAM + Ensemble): The most advanced variant, as proposed in our research report. This model combines the feature extraction power of MobileNetV2 with the refinement of a CBAM block. The resulting hybrid features are then classified by an ensemble of ML models to achieve maximum accuracy and robustness.

# Dataset Setup 
- Due to its large size, the image dataset used for this project is not included in this GitHub repository. You must download it manually. The scripts in each variant folder expect a local copy of the dataset.
- Dataset: Indian Currency Dataset (from Mendeley Data)
- Download Link: https://data.mendeley.com/datasets/8ckhkssyn3/1

# Setup Instructions:
- Click the download link above and download the dataset zip file.

- Extract the contents of the downloaded zip file. You should have a folder (likely named dataset or similar) containing subfolders for each denomination. Ensure the main folder containing the denomination subfolders is named dataset.

- Copy (or move) this entire dataset folder into each of the variant directories (Variant_1_CBAM_CNN, Variant_2_Transfer_Learning_ML, and Variant_3_Hybrid_Model).

- The final structure should look like this:

<img width="400" height="600" alt="image" src="https://github.com/user-attachments/assets/082a6eb8-e5ba-4560-a902-376cd7c84604" />



# How to Use

- Clone this repository.
- Follow the Dataset Setup instructions above.
- Navigate into the directory of the variant you wish to run.
- Follow the instructions in that variant's specific README.md file.


# Technologies Used

- Python
- PyTorch
- TensorFlow / Keras
- Scikit-learn
- OpenCV
- Pandas
- Matplotlib & Seaborn
- XGBoost & CatBoost
