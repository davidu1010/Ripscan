import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Set paths
# train_dir = '/home/paperspace/rip_current_classification/split_data/train'
# val_dir = '/home/paperspace/rip_current_classification/split_data/val'

train_dir = os.path.join('..', 'data', 'train')
val_dir = os.path.join('..', 'data', 'val')
checkpoint_dir = os.path.join('..', 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# TensorBoard SummaryWriter
writer = SummaryWriter()

# Define augmentations for training and validation datasets
train_transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


# Load EfficientNet-B7 from torchvision
model = models.efficientnet_b7(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)  # Adjust for 2 classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)


# Function to save checkpoint
def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """Save checkpoint to disk"""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth.tar')
        torch.save(state, best_checkpoint_path)


# Function to load checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resuming from checkpoint: {checkpoint_path} at epoch {epoch}")
    return epoch, best_val_loss


# Training and validation loop
def train_and_validate(num_epochs=10, checkpoint_path=None):
    start_epoch = 0
    best_val_loss = float('inf')

    # If resuming from a checkpoint
    if checkpoint_path:
        print(checkpoint_path)
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        # Training loop
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Train Loss': running_loss / len(train_loader)})

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, is_best, checkpoint_dir)

    writer.close()


# Start training
train_and_validate(num_epochs=500, checkpoint_path=None)
