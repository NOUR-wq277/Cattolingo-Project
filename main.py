import torch
import mlflow
import mlflow.pytorch
from src.preprocess import load_data
from src.utils import get_class_weights
from src.train import train_and_validate
from src.evaluate import test_model
from models.resnet_model import build_resnet50

# ----------------- Config -----------------
DATA_DIR = r"D:\Abdalrhman\Cat-Emotion-Detector\data\final_data"
CONFIG = {
    "BATCH_SIZE": 16,
    "NUM_EPOCHS": 30,
    "IMG_SIZE": 224,
    "LR": 0.0001,
    "NUM_WORKERS": 4,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Main -----------------
if __name__ == "__main__":
    with mlflow.start_run():
        mlflow.log_params(CONFIG)

        # Load Data
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, classes = load_data(
            DATA_DIR, CONFIG["BATCH_SIZE"], CONFIG["NUM_WORKERS"], CONFIG["IMG_SIZE"]
        )

        # Class Weights
        class_weights = get_class_weights(train_dataset, DEVICE)
        for i, w in enumerate(class_weights):
            mlflow.log_param(f"class_weight_{classes[i]}", w.item())

        # Model + Loss + Optimizer
        model = build_resnet50(len(classes)).to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])

        # Train
        train_and_validate(model, criterion, optimizer, train_loader, val_loader, train_dataset, val_dataset, DEVICE, CONFIG["NUM_EPOCHS"])

        # Load Best Model
        best_model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
        best_model = mlflow.pytorch.load_model(best_model_uri).to(DEVICE)

        # Test
        test_model(best_model, test_loader, classes, DEVICE)
