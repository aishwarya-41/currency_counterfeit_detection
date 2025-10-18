import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import glob

# --- Configuration ---
DATASET_PATH = 'dataset/'
IMG_SIZE = (224, 224)
RANDOM_STATE = 42
AUGMENTATIONS_PER_IMAGE = 4 # Create 4 augmented versions for each training image

# --- CBAM Module (as described in your report) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# --- Data Loading & Augmentation ---
def load_image_paths(dataset_path):
    """Loads all image paths from the subdirectories."""
    image_paths = []
    print(f"Loading image paths from '{dataset_path}'...")
    # Using glob to recursively find all common image types
    for ext in ('*.jpg', '*.png', '*.jpeg'):
        image_paths.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
    print(f"Found {len(image_paths)} genuine images.")
    return image_paths

def create_synthetic_counterfeit(image):
    """Applies augmentations to a PIL image to simulate a counterfeit."""
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
    ])
    return transform(image)

# --- Hybrid Feature Extractor Model ---
class HybridFeatureExtractor(nn.Module):
    def __init__(self):
        super(HybridFeatureExtractor, self).__init__()
        # Load MobileNetV2 as the backbone (as mentioned in your report)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Use all layers except the final classifier
        self.features = mobilenet.features
        # The number of output channels from MobileNetV2's feature extractor is 1280
        num_features = 1280
        # Add the CBAM block to refine the features
        self.cbam = CBAM(num_features)
        # Add the final pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

def process_and_extract_features(paths, model, device, is_training=False):
    """Loads images, creates counterfeits, and extracts hybrid features."""
    all_features = []
    all_labels = []
    
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    geometric_augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])

    desc = "Processing and Augmenting Training Data" if is_training else "Processing Validation/Test Data"
    for path in tqdm(paths, desc=desc):
        try:
            genuine_pil = Image.open(path).convert('RGB')
        except Exception:
            continue

        counterfeit_pil = create_synthetic_counterfeit(genuine_pil)

        if is_training:
            for _ in range(AUGMENTATIONS_PER_IMAGE):
                aug_genuine = geometric_augment(genuine_pil)
                aug_counterfeit = geometric_augment(counterfeit_pil)
                all_features.append(transform(aug_genuine))
                all_labels.append(1) # 1 for genuine
                all_features.append(transform(aug_counterfeit))
                all_labels.append(0) # 0 for counterfeit
        else:
            all_features.append(transform(genuine_pil))
            all_labels.append(1)
            all_features.append(transform(counterfeit_pil))
            all_labels.append(0)

    # Process in batches to avoid memory issues
    batch_size = 32
    extracted_features = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(all_features), batch_size), desc="Extracting Hybrid Features"):
            batch = torch.stack(all_features[i:i+batch_size]).to(device)
            features = model(batch)
            extracted_features.append(features.cpu().numpy())
            
    return np.vstack(extracted_features), np.array(all_labels)

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_paths = load_image_paths(DATASET_PATH)
    dummy_labels = [1] * len(all_paths) # Needed for splitting

    train_paths, test_paths = train_test_split(all_paths, test_size=0.2, random_state=RANDOM_STATE)
    train_paths, val_paths = train_test_split(train_paths, test_size=0.2, random_state=RANDOM_STATE)

    # Initialize the hybrid model
    feature_extractor = HybridFeatureExtractor().to(device)

    train_features, y_train = process_and_extract_features(train_paths, feature_extractor, device, is_training=True)
    val_features, y_val = process_and_extract_features(val_paths, feature_extractor, device)
    test_features, y_test = process_and_extract_features(test_paths, feature_extractor, device)

    print(f"\nShape of training features: {train_features.shape}")
    print(f"Shape of validation features: {val_features.shape}")
    print(f"Shape of testing features: {test_features.shape}")

    os.makedirs('features', exist_ok=True)
    np.save('features/train_features.npy', train_features)
    np.save('features/val_features.npy', val_features)
    np.save('features/test_features.npy', test_features)
    np.save('features/train_labels.npy', y_train)
    np.save('features/val_labels.npy', y_val)
    np.save('features/test_labels.npy', y_test)

    print("\nHybrid features and labels saved to 'features/' directory.")
    print("You can now run '2_train_ensemble_classifiers.py'")
