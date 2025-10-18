import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
# This path is now set to look for the 'dataset' folder in the same directory.
DATASET_PATH = 'dataset/' 
IMG_SIZE = (224, 224)
RANDOM_STATE = 42
# Number of extra augmented versions to create for each training image.
# More augmentations = more training data, but longer processing time.
AUGMENTATIONS_PER_IMAGE = 5 

# --- Data Loading ---
def load_image_paths_and_labels(dataset_path):
    """Loads all image paths from the subdirectories of the dataset_path."""
    image_paths = []
    print(f"Loading image paths from '{dataset_path}'...")
    for denomination_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, denomination_folder)
        if not os.path.isdir(folder_path): continue
        for img_file in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, img_file))
    labels = [1] * len(image_paths) # All are initially labeled as genuine
    print(f"Found {len(image_paths)} genuine images.")
    return image_paths, labels

# --- Preprocessing and Augmentation Functions (using OpenCV) ---
def create_synthetic_counterfeit(image):
    """Applies random blur and brightness/contrast changes to simulate a counterfeit."""
    # Adjust brightness and contrast
    alpha = 1.0 + np.random.uniform(-0.4, 0.4) # Contrast control
    beta = 0 + np.random.uniform(-40, 40)      # Brightness control
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Apply Gaussian blur with a random kernel size
    k_size = np.random.choice([3, 5, 7, 9])
    blurred = cv2.GaussianBlur(adjusted, (k_size, k_size), 0)
    return blurred

def augment_image(image):
    """Applies geometric augmentations like rotation and flipping."""
    # Random Rotation
    angle = np.random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))
    
    # Random Horizontal Flip
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
        
    # Minor Brightness adjustment
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=np.random.uniform(-20, 20))
    return image

def process_and_extract_features(paths, base_model, is_training=False):
    """Loads images, applies augmentation, creates counterfeits, and extracts features using the base_model."""
    all_images = []
    all_labels = []
    
    desc = "Processing and Augmenting Training Data" if is_training else "Processing Validation/Test Data"
    for path in tqdm(paths, desc=desc):
        genuine_img = cv2.imread(path)
        if genuine_img is None: continue
        genuine_img = cv2.resize(genuine_img, IMG_SIZE)
        
        # Create one synthetic counterfeit for each genuine image
        counterfeit_img = create_synthetic_counterfeit(genuine_img)

        if is_training:
            # For training data, create multiple augmented versions
            for _ in range(AUGMENTATIONS_PER_IMAGE):
                aug_genuine = augment_image(genuine_img)
                aug_counterfeit = augment_image(counterfeit_img)
                all_images.extend([aug_genuine, aug_counterfeit])
                all_labels.extend([1, 0]) # 1 for genuine, 0 for counterfeit
        else:
            # For val/test data, just use the original and its one counterfeit
            all_images.extend([genuine_img, counterfeit_img])
            all_labels.extend([1, 0])

    # Convert to numpy array and preprocess for the base model
    images_np = np.array(all_images, dtype="float32")
    preprocessed_images = tf.keras.applications.mobilenet_v2.preprocess_input(images_np)
    
    # Use the base model to extract deep features
    print(f"\nExtracting deep features from {len(preprocessed_images)} images...")
    deep_features = base_model.predict(preprocessed_images, batch_size=32)
    return deep_features, np.array(all_labels)

# --- Main Execution ---
if __name__ == "__main__":
    # Load paths and split them into train, validation, and test sets
    all_image_paths, all_labels = load_image_paths_and_labels(DATASET_PATH)
    # Splitting the original paths of genuine images
    train_paths, test_paths, _, _ = train_test_split(all_image_paths, all_labels, test_size=0.2, random_state=RANDOM_STATE)
    train_paths, val_paths, _, _ = train_test_split(train_paths, train_paths, test_size=0.2, random_state=RANDOM_STATE) # Dummy labels for splitting

    # Load pre-trained MobileNetV2 model without the top classification layer
    print("\nLoading pre-trained MobileNetV2 model...")
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg')
    base_model.trainable = False # We are only using it for feature extraction

    # Process each data split to get the deep features
    train_features, y_train = process_and_extract_features(train_paths, base_model, is_training=True)
    val_features, y_val = process_and_extract_features(val_paths, base_model)
    test_features, y_test = process_and_extract_features(test_paths, base_model)

    print(f"\nShape of training features: {train_features.shape}")
    print(f"Shape of validation features: {val_features.shape}")
    print(f"Shape of testing features: {test_features.shape}")

    # Save the extracted features and labels to files for the next step
    os.makedirs('features', exist_ok=True)
    np.save('features/train_features.npy', train_features)
    np.save('features/val_features.npy', val_features)
    np.save('features/test_features.npy', test_features)
    np.save('features/train_labels.npy', y_train)
    np.save('features/val_labels.npy', y_val)
    np.save('features/test_labels.npy', y_test)
    print("\nAll features and labels have been saved to the 'features/' directory.")
    print("You can now run '2_train_classifiers.py'")
