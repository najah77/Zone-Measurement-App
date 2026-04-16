import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add root backend directory to path if running script standalone
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from app.ml.disc_classifier import DiscCNN

def train_model(dataset_dir: str, epochs: int, batch_size: int, output_dir: str):
    """
    Trains the Disc Classifier CNN on cropped ROI images.
    Expects dataset_dir to have subfolders for each code (e.g., dataset/IPM/, dataset/FOX/).
    """
    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Dataset directory '{dataset_dir}' not found.")
        print("Please structure your data as: dataset/CODE/image1.jpg")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Data Augmentations
    train_transforms = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomRotation(degrees=180), # Handle unoriented text
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load Dataset
    try:
        train_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transforms)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    num_classes = len(train_dataset.classes)
    if num_classes == 0:
        print("❌ Dataset is empty or incorrectly structured.")
        return

    print(f"📦 Loaded {len(train_dataset)} images across {num_classes} classes.")
    print(f"🏷️ Classes: {train_dataset.class_to_idx}")

    # Create output directory for model and mapping
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Class Mapping immediately for API inference mapping
    class_map_path = os.path.join(output_dir, "class_map.json")
    with open(class_map_path, 'w') as f:
        json.dump(train_dataset.class_to_idx, f, indent=4)
    print(f"💾 Saved class mapping to {class_map_path}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize Model
    model = DiscCNN(num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n🔥 Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    model_path = os.path.join(output_dir, "disc_classifier.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n🎉 Training complete! Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AST Disc Classifier CNN")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to cropped dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--out", type=str, default="app/ml/models", help="Output directory for .pt and class map")

    args = parser.parse_args()
    
    train_model(args.dataset, args.epochs, args.batch_size, args.out)
