import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.dataset import get_dataloaders

def train_model(data_dir="data", models_dir="models", epochs=5, batch_size=32, lr=0.001):
    """
    Fine-tunes a pretrained ResNet-18 model on the custom geometric shapes dataset.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load DataLoaders
    print("Loading datasets...")
    train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(class_names)
    print(f"Dataset classes: {class_names}")
    
    # Save class names mapping for later inference
    class_names_path = os.path.join(models_dir, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=4)
    print(f"Saved class names mapping to {class_names_path}")
    
    # 2. Set up Device (CPU by default as requested/configured)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. Initialize Model (ResNet-18)
    print("Initializing pretrained ResNet-18...")
    # Use modern PyTorch API for weights
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Freeze model parameters (feature extractor weights)
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final fully connected layer (classifier head)
    # The new layer parameters will have requires_grad=True automatically
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)
    
    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Optimize ONLY the parameters of the final layer (which we just replaced)
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    
    # 5. Training Loop
    best_acc = 0.0
    best_model_path = os.path.join(models_dir, "best_model.pth")
    
    print("\nStarting training loop...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 10)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train_samples = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass & optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_train_samples += inputs.size(0)
            
        epoch_train_loss = running_loss / total_train_samples
        epoch_train_acc = running_corrects.double() / total_train_samples
        
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")
        
        # Validation Phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_val_samples += inputs.size(0)
                
        epoch_val_loss = running_loss / total_val_samples
        epoch_val_acc = running_corrects.double() / total_val_samples
        
        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        # Save best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path} (Acc: {best_acc:.4f})")
            
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    # Run training for 5 epochs
    train_model(epochs=5)
