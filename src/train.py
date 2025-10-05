import torch
import mlflow
from tqdm import tqdm

def train_and_validate(model, criterion, optimizer, train_loader, val_loader, train_dataset, val_dataset, device, num_epochs=30):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ----------------- Training -----------------
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

        # ----------------- Validation -----------------
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

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            mlflow.pytorch.log_model(model, "best_model")
            print(f"✔ New best model saved with val acc = {best_val_acc:.4f}")
