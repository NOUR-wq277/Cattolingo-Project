import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import mlflow.pytorch
from torchvision.models import ResNet50_Weights

# ==============================================================================
# 1. CONFIGURATION AND SETUP
# ==============================================================================

# --- Global Configuration ---
# NOTE: Replace the placeholder with your actual data path
DATA_DIR = r"D:\Abdalrhman\Cat-Emotion-Detector\Catto-Lingo\final_data" 
CONFIG = {
    "BATCH_SIZE": 16,
    "NUM_EPOCHS": 30,
    "IMG_SIZE": 224, # Standard input size for ResNet models
    "LR": 0.0001,
    "NUM_WORKERS": 4, # Number of subprocesses to use for data loading
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. UTILITY FUNCTIONS (Data Loading and Model Building)
# ==============================================================================

def get_transforms(img_size: int = 224):
    """Defines the transformation pipelines for training and validation/testing data."""
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transforms, val_test_transforms

def load_data(data_dir: str, config: dict):
    """Loads datasets and creates DataLoaders for train, validation, and test splits."""
    train_transforms, val_test_transforms = get_transforms(config["IMG_SIZE"])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS"])
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"])
    test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"])

    print(f"Data loaded. Train/Val/Test Samples: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_dataset.classes

def get_class_weights(train_dataset: datasets.ImageFolder, device: torch.device):
    """Calculates balanced class weights from the training dataset labels."""
    y_train = [s[1] for s in train_dataset.samples]
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    return torch.tensor(class_weights, dtype=torch.float).to(device)

def build_resnet50(num_classes: int):
    """Initializes a ResNet-50 model with pre-trained weights and modifies the final layer."""
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    # Replace the existing fc layer with a new one
    model.fc = nn.Linear(num_features, num_classes)
    return model

# ==============================================================================
# 3. TRAINING AND EVALUATION LOGIC
# ==============================================================================

def train_and_validate(model, criterion, optimizer, train_loader, val_loader, train_dataset, val_dataset, device, num_epochs):
    """Executes the training and validation loop, logging metrics and saving the best model via MLflow."""
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train() 
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(outputs.argmax(1) == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        # --- Validation Phase ---
        model.eval() 
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(outputs.argmax(1) == labels.data)

        val_loss /= len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")

        # MLflow logging
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_acc", epoch_acc.item(), step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc.item(), step=epoch)

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            mlflow.pytorch.log_model(model, "best_model")
            print(f"✔ New best model saved with val acc = {best_val_acc:.4f}")

    return best_val_acc

def test_model(model, test_loader, classes, device, run_id):
    """Evaluates the best model on the test set and logs final artifacts to MLflow."""
    print("--- Starting Final Evaluation on Test Set ---")
    model.eval() 
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="[Testing]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Classification Report ---
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print("\nClassification Report:")
    print(report)
    mlflow.log_text(report, "classification_report.txt")

    # --- Confusion Matrix Visualization ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.title(f"Confusion Matrix (Run ID: {run_id[:8]})")
    
    # Save and log the figure
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.show()

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    """Main execution function to run the full training and testing pipeline."""
    print(f"Starting Cat Emotion Detector. Using device: {DEVICE}")

    # 1. Load Data
    train_loader, val_loader, test_loader, train_dataset, val_dataset, _, classes = load_data(DATA_DIR, CONFIG)

    # 2. Setup Model, Loss, and Optimizer
    class_weights = get_class_weights(train_dataset, DEVICE)
    model = build_resnet50(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LR"])
    
    print(f"Model and Loss initialized with {len(classes)} classes and weighted CrossEntropyLoss.")

    # 3. MLflow Run and Training
    with mlflow.start_run() as run:
        # Log Hyperparameters and Weights
        mlflow.log_params(CONFIG)
        for i, w in enumerate(class_weights):
            mlflow.log_param(f"class_weight_{classes[i]}", w.item())

        run_id = run.info.run_id
        print(f"\nMLflow Run Started (ID: {run_id})")

        # Train and Validate
        best_val_acc = train_and_validate(
            model, criterion, optimizer, train_loader, val_loader, 
            train_dataset, val_dataset, DEVICE, CONFIG["NUM_EPOCHS"]
        )
        print(f"\nTraining finished. Best Validation Accuracy: {best_val_acc:.4f}")

        # 4. Load Best Model and Test
        best_model_uri = f"runs:/{run_id}/best_model"
        print(f"Loading best model from MLflow artifact: {best_model_uri}")
        best_model = mlflow.pytorch.load_model(best_model_uri).to(DEVICE)

        test_model(best_model, test_loader, classes, DEVICE, run_id)
        
        # Log final metrics and tags
        mlflow.log_metric("final_test_accuracy", best_val_acc.item()) # Using best validation acc as the final metric to log.
        mlflow.set_tag("model_architecture", "ResNet-50_TransferLearning")
        mlflow.set_tag("data_path", DATA_DIR)


if __name__ == "__main__":
    # Ensure MLflow is configured (e.g., set tracking URI if needed)
    # mlflow.set_tracking_uri("http://your-mlflow-server:5000")
    
    # Check if data directory is accessible before proceeding
    if not os.path.exists(DATA_DIR):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: Data directory not found at: {DATA_DIR}")
        print("Please update the 'DATA_DIR' variable in the script to the correct path.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        main()