import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
    """
    Returns data transforms for training and validation splits.
    Uses standard ImageNet normalization since we'll use a pretrained ResNet-18.
    """
    # ImageNet mean and standard deviation
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # Standard augmentations for classification tasks
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return train_transform, val_transform

def get_dataloaders(data_dir="data", batch_size=32, num_workers=0):
    """
    Creates and returns PyTorch DataLoader objects for training and validation splits,
    along with class-to-index mappings.
    """
    train_transform, val_transform = get_transforms()
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Dataset splits not found in '{data_dir}'. Make sure to run the dataset generation script first."
        )
        
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Class names list (sorted alphabetically by default in ImageFolder)
    class_names = train_dataset.classes
    
    return train_loader, val_loader, class_names

if __name__ == "__main__":
    # Test dataset loading
    try:
        train_loader, val_loader, class_names = get_dataloaders()
        print("Dataloaders initialized successfully!")
        print(f"Classes: {class_names}")
        print(f"Number of training samples: {len(train_loader.dataset)}")
        print(f"Number of validation samples: {len(val_loader.dataset)}")
        
        # Check a batch
        x, y = next(iter(train_loader))
        print(f"Batch shapes - X: {x.shape}, y: {y.shape}")
    except Exception as e:
        print(f"Error testing dataloaders: {e}")
