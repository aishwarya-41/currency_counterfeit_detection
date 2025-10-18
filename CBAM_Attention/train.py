import torch
import torch.nn as nn
from tqdm import tqdm
import copy

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_save_path):
    """
    Trains the model, validates it, and saves the best performing version.
    """
    best_val_accuracy = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        progress_bar_val = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for inputs, labels in progress_bar_val:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                progress_bar_val.set_postfix(loss=f"{loss.item():.4f}")
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # --- Save the best model ---
        if val_epoch_acc > best_val_accuracy:
            best_val_accuracy = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_save_path)
            print(f"New best model saved to {model_save_path} with accuracy: {best_val_accuracy:.4f}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_accuracy:.4f}")
    model.load_state_dict(best_model_wts)

