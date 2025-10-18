import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import glob

# --- Custom Transforms to Create Synthetic Counterfeits ---

class GaussianBlur:
    """Applies Gaussian Blur to a PIL Image."""
    def __init__(self, kernel_size=(5, 5)):
        self.kernel_size = kernel_size

    def __call__(self, img):
        img_np = np.array(img)
        # Apply blur
        blurred = cv2.GaussianBlur(img_np, self.kernel_size, 0)
        return Image.fromarray(blurred)

def create_synthetic_counterfeit_transform():
    """Creates a transform pipeline to simulate counterfeit notes."""
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        GaussianBlur(kernel_size=(5, 5)),
    ])


# --- Custom PyTorch Dataset ---

class SyntheticCounterfeitDataset(Dataset):
    """
    A custom PyTorch Dataset that takes a list of genuine image paths
    and generates a dataset twice the size, containing both the original
    genuine images and synthetically created fakes.
    """
    def __init__(self, image_paths, is_training=True):
        self.image_paths = image_paths
        self.is_training = is_training
        
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
        
        self.counterfeit_transform = create_synthetic_counterfeit_transform()

    def __len__(self):
        return len(self.image_paths) * 2

    def __getitem__(self, idx):
        original_img_idx = idx // 2
        is_genuine = (idx % 2 == 0)
        
        img_path = self.image_paths[original_img_idx]
        image = Image.open(img_path).convert('RGB')
        
        label = 1 if is_genuine else 0

        if not is_genuine:
            image = self.counterfeit_transform(image)
        
        if self.is_training:
            image = self.train_augment_transform(image)
        
        image = self.base_transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


# --- Main Data Preparation Function ---

def prepare_dataset(data_dir='dataset', batch_size=32):
    """
    Loads all genuine image paths from the specified directory, splits them, 
    and creates the PyTorch DataLoaders using the synthetic generation dataset.
    """
    if not os.path.isdir(data_dir):
        print("="*60)
        print(f"!! ERROR: Dataset folder '{data_dir}' not found. !!")
        print("Please ensure your folder containing genuine currency is named 'dataset' or change the path in main.py.")
        print("="*60)
        exit()

    all_image_paths = glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True)
    if not all_image_paths:
        print(f"!! ERROR: No '.jpg' images found in '{data_dir}'. !!")
        exit()

    print(f"Found {len(all_image_paths)} genuine images.")

    train_paths, test_paths = train_test_split(all_image_paths, test_size=0.2, random_state=42)
    train_paths, val_paths = train_test_split(train_paths, test_size=0.2, random_state=42)

    print(f"Splitting into {len(train_paths)} train, {len(val_paths)} validation, and {len(test_paths)} test images.")
    print("Each set will be doubled to include synthetic fakes, creating balanced classes.")

    train_dataset = SyntheticCounterfeitDataset(train_paths, is_training=True)
    val_dataset = SyntheticCounterfeitDataset(val_paths, is_training=False)
    test_dataset = SyntheticCounterfeitDataset(test_paths, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    class_names = ['Synthetic Fake', 'Genuine']

    return train_loader, val_loader, test_loader, class_names

