# Hybrid Attention Model + ML Ensemble
This directory contains the implementation of Variant 3, a sophisticated hybrid model that mirrors the architecture proposed in the research report. This approach combines a pre-trained deep learning backbone, an attention mechanism for feature refinement, and an ensemble of classical machine learning classifiers to achieve robust and balanced counterfeit detection.

# Overview
- This variant represents the most complex and nuanced approach of the three experiments. The core strategy is to create a powerful, two-stage feature extraction pipeline before feeding the results into a final ensemble of classifiers for a democratic decision.
- Pre-trained Backbone: We start with MobileNetV2 as an expert feature extractor, leveraging its vast knowledge from the ImageNet dataset to capture rich, high-level patterns from the banknote images.
- Attention-Based Refinement: The features from MobileNetV2 are then passed through a Convolutional Block Attention Module (CBAM). This step is crucial as it allows the system to refine the features by learning to focus on the most discriminative regions and channels, effectively mimicking how a human expert might pay closer attention to watermarks or security threads.
- Ensemble Classification: The final, attention-weighted feature vectors are used to train a suite of powerful machine learning classifiers (Random Forest, SVM, XGBoost, CatBoost). A Hard Voting Ensemble of these classifiers makes the final prediction, ensuring the decision is robust and less prone to the errors of any single model.

# How it Works
- The pipeline is divided into two distinct scripts that must be run in order:
- 1_extract_hybrid_features.py: This script orchestrates the entire feature extraction process. It loads genuine notes, creates synthetic fakes, passes them through the MobileNetV2 backbone, refines the output with the CBAM module, and saves the final, enhanced feature vectors to a new features/ directory.
- 2_train_ensemble_classifiers.py: This script loads the pre-computed hybrid features. It then trains and evaluates all individual ML classifiers and the final hard-voting ensemble. Finally, it prints a detailed performance comparison and saves a confusion matrix for the best model.

# File Structure
- 1_extract_hybrid_features.py: The first script to run. It uses MobileNetV2 and a CBAM block to extract and save sophisticated feature vectors from the images.
- 2_train_ensemble_classifiers.py: The second script to run. It loads the hybrid features and trains, evaluates, and compares the ML classifiers and the final hard-voting ensemble.
- requirements.txt: A text file listing all the Python libraries and dependencies required.
- features/: A directory automatically created by the first script to store the .npy feature and label files.

# Setup and Execution
1. Prerequisites
   - Before running, ensure you have set up the main dataset folder inside this directory as described in the root README.md of the project.

2. Install Dependencies
    - Open a terminal in this directory and install the required Python libraries.
    - pip install -r requirements.txt


3. Step 1: Extract Hybrid Features
    - Run the first script to process all images and generate the feature files. This is the most computationally intensive step.
    - python 1_extract_hybrid_features.py
    - This will create a new folder named features/ in your directory.

4. Step 2: Train and Compare Classifiers
    - Once the feature extraction is complete, run the second script. This step is much faster.
    - python 2_train_ensemble_classifiers.py

# Results and Evaluation
This hybrid approach produced a highly balanced model. The final Ensemble (Hard Vote) classifier was the top performer, achieving a solid overall accuracy of 70.11%.

<img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/8199e724-90bb-471d-8d96-18cccfc368e6" />

# Detailed Report for Best Model: Ensemble (Hard Vote)
The ensemble model provides the most balanced performance across both classes.

<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/ef8bac34-8228-4290-b3ad-36096923cad9" />

# Confusion Matrix for Best Model: Ensemble (Hard Vote)
The confusion matrix for the ensemble model visually demonstrates its balanced nature.

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/a86279d9-ab59-47aa-a28f-8fd440dfb3bf" />

# Advantages of this Hybrid Variant
- While Variant 2 (MobileNetV2 + SVM) achieved a higher raw accuracy, this Hybrid Variant presents several key advantages that make it a more robust and potentially more reliable real-world solution.
  
- More Balanced Performance: The most significant advantage is visible in the confusion matrix. Unlike the SVM in Variant 2 which was perfect at identifying genuine notes but made several mistakes on fakes, this ensemble model has a more fair and balanced distribution of errors. It makes mistakes on both classes, but it doesn't have a strong bias towards one, which is crucial for a trustworthy system.

- Robustness Through Democracy: The hard-voting mechanism makes the final decision more robust. An error or uncertainty from one model can be corrected by the majority vote of the others, leading to a more stable and generalizable system.

- Advanced Feature Representation: By combining a pre-trained backbone with an attention mechanism, this variant creates a feature representation that is theoretically more powerful and nuanced. While it didn't achieve the highest accuracy in this specific test, this architecture has a higher potential to perform well on more complex and varied datasets.
