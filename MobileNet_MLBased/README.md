# Transfer Learning + ML Classifiers
This directory contains the implementation of Variant 2, a high-performance counterfeit currency detection system that uses a two-stage, hybrid approach. This variant combines the power of a large, pre-trained deep learning model for feature extraction with the speed and efficiency of classical machine learning algorithms for classification.

# Overview
- The core strategy of this variant is to leverage transfer learning. Instead of training a deep learning model from scratch, we use MobileNetV2, a state-of-the-art model pre-trained on millions of diverse images from the ImageNet dataset. MobileNetV2 acts as an expert "feature extractor," converting each banknote image into a highly informative numerical signature (a feature vector) that captures its essential patterns and textures.
- These rich feature vectors are then used to train a suite of powerful and efficient machine learning classifiers:
    1. Random Forest
    2. Support Vector Machine (SVM)
    3. XGBoost
    4. CatBoost

- This two-stage process is highly effective because it separates the complex task of visual understanding from the final classification, often leading to superior accuracy and faster training times for the final classifiers.

# How it Works
- The pipeline is divided into two distinct scripts that must be run in order:
- 1_extract_features.py: This script first loads all the genuine currency images from the dataset/ folder. For each genuine note, it creates a corresponding synthetic counterfeit by applying augmentations. It then uses the pre-trained MobileNetV2 model to process every image (both real and fake) and extract a 1280-dimension feature vector. These features are saved to disk in a new features/ directory.
- 2_train_classifiers.py: This script loads the pre-computed features from the features/ directory. It then trains and evaluates all four machine learning classifiers on this data. Finally, it prints a performance comparison and generates a detailed report and confusion matrix for the best-performing model.

# File Structure
- 1_extract_features.py: The first script to run. It loads images, creates synthetic fakes, and uses MobileNetV2 to extract and save deep feature vectors.
- 2_train_classifiers.py: The second script to run. It loads the saved features and trains, evaluates, and compares the four ML classifiers.
- requirements.txt: A text file listing all the Python libraries and dependencies required to run the project.
- features/: A directory automatically created by the first script to store the .npy feature and label files.

# Setup and Execution
1. Prerequisites
- Before running, ensure you have set up the main dataset folder inside this directory as described in the root README.md of the project.

2. Install Dependencies
- Open a terminal in this directory and install the required Python libraries.
- pip install -r requirements.txt

3. Step 1: Extract Features
- Run the first script to process all images and generate the feature files. This step will take the most time as it involves the deep learning model.
- python 1_extract_features.py
- This will create a new folder named features/ in your directory.

4. Step 2: Train and Compare Classifiers
- Once the feature extraction is complete, run the second script. This step is much faster.
- python 2_train_classifiers.py


# Results and Evaluation
- This approach yielded outstanding results, with all classifiers achieving over 98% accuracy. The Support Vector Machine (SVM) was the top-performing model, achieving a near-perfect accuracy of 98.74%.
- Performance Comparison of Ensemble Classifiers: The table below shows the performance of each classifier on the unseen test set.

<img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/e58c785d-8c44-4c4c-b128-6d9f52d7bc3e" />

# Detailed Report for Best Model: SVM
The SVM model demonstrated exceptional performance, correctly classifying almost every note in the test set.
<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/40ffd0b5-c6e7-4c4a-b66d-098ee471d5c8" />

# Confusion Matrix for Best Model: SVM
The confusion matrix visually confirms the model's high accuracy. It shows that out of 358 genuine notes, all 358 were correctly identified. Out of 358 counterfeit notes, 349 were correctly identified, with only 9 misclassifications.
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/01737c4d-1cf2-46e4-b409-42926a69fe3b" />



