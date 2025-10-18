import torch
import torch.nn as nn
import torch.optim as optim
import os

from model.cnn_with_cbam import CNNWithCBAM
from data_loader import prepare_dataset
from train import train_model
from test import test_model

def main():
    # --- Configuration ---
    # This path should contain subfolders of ONLY GENUINE notes
    dataset_folder = 'dataset' 
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    
    os.makedirs('saved_models', exist_ok=True)
    model_save_path = 'saved_models/best_synthetic_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Step 1: Prepare Dataset ---
    print("\n--- Step 1: Preparing Dataset ---")
    # The new data loader creates synthetic fakes on the fly
    train_loader, val_loader, test_loader, class_names = prepare_dataset(
        data_dir=dataset_folder, batch_size=batch_size
    )
    print("Dataset prepared successfully with synthetic data generation.")

    # --- Step 2: Initialize Model, Loss, and Optimizer ---
    print("\n--- Step 2: Initializing Model ---")
    # num_classes is 2: 'Synthetic Fake' and 'Genuine'
    model = CNNWithCBAM(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Model initialized successfully.")

    # --- Step 3: Train the Model ---
    print("\n--- Step 3: Starting Training ---")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_save_path)
    
    # --- Step 4: Test the Best Model ---
    print("\n--- Step 4: Testing Final Model ---")
    # Load the best performing model for final testing
    model.load_state_dict(torch.load(model_save_path))
    test_model(model, test_loader, class_names, device)
    
    print("\n--- All Done! ---")
    print(f"The best model has been saved to {model_save_path}")
    print("To test a single image, run the command:")
    print(f"python predict.py --image_path \"path/to/your/test_note.jpg\" --model_path \"{model_save_path}\"")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

