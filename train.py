import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Dataset path
dataset_path = "/Users/achal/Downloads/gender-classification/Dataset"

# Enhanced data transforms with augmentation for training
train_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # Reduced image size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Validation/Test transforms without augmentation
val_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # Reduced image size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load datasets with appropriate transforms
train_dataset = datasets.ImageFolder(
    root=os.path.join(dataset_path, "Train"), transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(dataset_path, "Validation"), transform=val_transform
)
test_dataset = datasets.ImageFolder(
    root=os.path.join(dataset_path, "Test"), transform=val_transform
)

# Use subset of data for faster training
train_size = 20000  # Reduced training size
val_size = 5000  # Reduced validation size
test_size = 5000  # Reduced test size

train_indices = torch.randperm(len(train_dataset))[:train_size]
val_indices = torch.randperm(len(val_dataset))[:val_size]
test_indices = torch.randperm(len(test_dataset))[:test_size]

train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)
test_dataset = Subset(test_dataset, test_indices)

# Create data loaders with increased batch size
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Print dataset sizes
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0


def plot_metrics(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    early_stopping_epoch=None,
):
    plt.figure(figsize=(15, 10))

    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label="Training Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    if early_stopping_epoch:
        plt.axvline(
            x=early_stopping_epoch,
            color="r",
            linestyle="--",
            label=f"Early Stopping (Epoch {early_stopping_epoch})",
        )
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy", marker="o")
    plt.plot(val_accuracies, label="Validation Accuracy", marker="o")
    if early_stopping_epoch:
        plt.axvline(
            x=early_stopping_epoch,
            color="r",
            linestyle="--",
            label=f"Early Stopping (Epoch {early_stopping_epoch})",
        )
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # Plot Learning Rate
    plt.subplot(2, 2, 3)
    plt.plot(train_accuracies, val_accuracies, marker="o")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Training Accuracy (%)")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()


def train_model():
    # Initialize model
    model = models.resnet18(pretrained=True)  # Using ResNet18 instead of GoogLeNet
    model.fc = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1)  # Reduced dropout
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Increased learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=3, verbose=True
    )
    early_stopping = EarlyStopping(patience=5)  # Reduced patience

    # Training metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0
    epochs = 20  # Reduced epochs

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.sigmoid(outputs) > 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_loader_tqdm.set_postfix(
                loss=running_loss / len(train_loader), acc=100 * correct / total
            )

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(
                    device
                ).float().unsqueeze(1)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()
                val_predicted = torch.sigmoid(val_outputs) > 0.5
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        # Plot current metrics
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

        if (
            early_stopping.early_stop and val_accuracy >= 90.0
        ):  # Reduced accuracy threshold
            print("Early stopping triggered with desired accuracy achieved!")
            break
        elif early_stopping.early_stop:
            print(
                "Early stopping triggered but continuing as desired accuracy not reached..."
            )
            early_stopping = EarlyStopping(patience=5)

    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))
    return model, device


def evaluate_model(model, device):
    model.eval()
    test_correct = 0
    test_total = 0
    all_probs = []
    true_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predicted = probs > 0.5

            all_probs.extend(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            test_total += labels.size(0)
            test_correct += (predicted.squeeze() == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")

    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    true_labels = np.array(true_labels)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(true_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 10))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(true_labels, (all_probs > 0.5).astype(int))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Female", "Male"],
        yticklabels=["Female", "Male"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Classification Report
    report = classification_report(
        true_labels, (all_probs > 0.5).astype(int), target_names=["Female", "Male"]
    )
    print("\nClassification Report:")
    print(report)

    return test_accuracy, report


if __name__ == "__main__":
    print("Starting model training...")
    print(
        f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )

    model, device = train_model()
    test_accuracy, report = evaluate_model(model, device)
