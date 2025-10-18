import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

def test_model(model: nn.Module, test_loader: DataLoader, class_names: list, device: str = 'cpu'):
    """
    Evaluates the model on the test set and displays results.
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("\nEvaluating on the test set (containing genuine and synthetic fakes)...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Classification Report ---
    print("\n" + "="*25)
    print("  Classification Report")
    print("="*25)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # --- Confusion Matrix ---
    print("\n" + "="*25)
    print("    Confusion Matrix")
    print("="*25)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix plot saved as 'confusion_matrix.png'")
    plt.show()

