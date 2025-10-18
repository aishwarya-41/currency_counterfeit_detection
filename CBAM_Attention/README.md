# Attention-Enhanced CNN (CBAM)
This directory contains the implementation of Variant 1, an end-to-end deep learning model for counterfeit currency detection. The model uses a custom Convolutional Neural Network (CNN) architecture enhanced with a Convolutional Block Attention Module (CBAM) to improve feature learning and classification accuracy.

# Overview
- The core strategy of this variant is to train a model that can effectively distinguish between genuine banknotes and synthetically generated counterfeits. By learning the intricate patterns of real currency, the model becomes adept at identifying fakes that lack these authentic features.
- The CBAM attention mechanism is a key component of this architecture. It allows the model to dynamically learn and focus on the most important parts of an image—both spatially (like a watermark) and across channels (like specific color patterns)—while suppressing irrelevant noise. This leads to a more robust and accurate classification.

# How it Works
- The main.py script orchestrates the entire pipeline:
- Data Loading: The script begins by scanning the dataset/ folder to locate all genuine currency images.
- Synthetic Data Generation: For each genuine image, a corresponding "synthetic counterfeit" is created on-the-fly. This is achieved by applying a series of aggressive image augmentations, such as blurring, noise injection, and severe contrast/brightness adjustments. This process doubles the size of our dataset, providing the model with both positive (Genuine) and negative (Fake) examples.
- End-to-End Training: The CNN+CBAM model is trained from scratch on these pairs of genuine and synthetic counterfeit images. It learns to solve a binary classification task: "Is this note Genuine or Fake?"
- Evaluation: After training, the model's performance is evaluated on a separate, unseen test set of genuine and synthetic fake images.

# File Structure
- main.py: The main executable script that runs the entire pipeline. It orchestrates the process by calling functions from the other scripts to load data, train the model, and evaluate the final results.
- data_loader.py: This script is responsible for all data-related tasks. It scans the dataset directory, splits the image paths into training, validation, and test sets, and defines the custom PyTorch Dataset class that generates synthetic counterfeit images on-the-fly.
- train.py: Contains the core training logic. The train_model function iterates through the specified number of epochs, performs the training and validation steps for each epoch, and saves the best-performing model based on validation accuracy.
- test.py: This script handles the final evaluation of the trained model. The test_model function takes the best saved model, runs it on the unseen test set, and generates the final classification report and confusion matrix.
- model/: This directory contains the Python files that define the neural network architecture.
  1. attention.py: Implements the ChannelAttention and SpatialAttention modules that form the CBAM block.
  2. cnn_with_cbam.py: Defines the main CNNWithCBAM class, which assembles the convolutional layers and integrates the attention modules.

- saved_models/: This directory is automatically created to store the weights of the best-performing model during training. The final model is saved as best_synthetic_model.pth.


# Setup and Execution
1. Prerequisites
- Before running, ensure you have set up the main dataset folder inside this directory as described in the root README.md of the project.

2. Install Dependencies
- Open a terminal in this directory and install the required Python libraries.

3. Run the Training and Evaluation
- Execute the main script from your terminal. This single command will handle data preparation, model training, and final testing.
- python main.py

# Results and Evaluation

The model was trained for 20 epochs on a CPU. It demonstrated strong learning capabilities, achieving a peak validation accuracy of 82.17% during training. The final evaluation on the test set yielded a robust overall accuracy of 83.24%.

# Classification Report

The detailed performance on the test set is shown below. The model shows a high precision in identifying fakes and a very high recall for genuine notes, indicating it is very good at correctly identifying real currency.

<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/9c79ebdc-9566-4052-a3d2-05efa6a84b3f" />


Confusion Matrix

The confusion matrix provides a visual representation of the model's performance. It shows that the model correctly identified 340 out of 358 genuine notes and 256 out of 358 synthetic fakes.

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/92c8ec4a-88e4-43d7-95bf-890a64b72593" />

