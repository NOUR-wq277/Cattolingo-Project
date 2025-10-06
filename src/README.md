# Cat Classification Model - Source Code

This directory contains the modularized source code for the VGG16-based cat classification model.

## 📁 File Structure

```
src/
├── __init__.py           # Package initialization
├── config.py             # Configuration and hyperparameters
├── model.py              # Model architecture (VGG16-based)
├── data.py               # Data loading and augmentation
├── train.py              # Training pipeline
├── predict.py            # Inference/prediction utilities
├── utils.py              # Helper functions
├── mlflow_logger.py      # MLflow integration
└── README.md             # This file
```

## 📋 Module Descriptions

### `config.py`
Contains all configuration parameters:
- Model hyperparameters (learning rates, batch size, etc.)
- Data augmentation settings
- File paths
- Training configurations

### `model.py`
Model architecture functions:
- `create_model()`: Creates VGG16-based model
- `unfreeze_last_layers()`: Unfreezes layers for fine-tuning

### `data.py`
Data handling utilities:
- `create_data_generators()`: Creates training and validation data generators
- `get_class_names()`: Extracts class labels from generator

### `train.py`
Complete training pipeline:
- Two-phase training (feature extraction + fine-tuning)
- Callbacks setup (early stopping, learning rate reduction, checkpointing)
- Results saving

### `predict.py`
Inference utilities:
- `CatClassifier` class for making predictions
- Batch prediction support
- Image preprocessing

### `utils.py`
Helper functions:
- Load saved models and results
- Plot training history
- MLflow logging utilities
- Model information display

### `mlflow_logger.py`
MLflow integration:
- Log models to MLflow
- Track experiments
- Save metrics

## 🚀 Usage Examples

### Training the Model

```python
from src.train import train_model

# Train with default settings from config.py
model, history = train_model()

# Or specify custom data directories
model, history = train_model(
    train_dir="path/to/train",
    val_dir="path/to/val"
)
```

### Making Predictions

```python
from src.predict import CatClassifier

# Initialize classifier
classifier = CatClassifier(model_path="final_model.h5")

# Define class names
class_names = {
    0: "Breed_1",
    1: "Breed_2",
    2: "Breed_3",
    3: "Breed_4",
    4: "Breed_5"
}

# Predict single image
predicted_class, confidence, probs = classifier.predict(
    "path/to/image.jpg",
    class_names=class_names
)
```

### Loading Results

```python
from src.utils import load_model_and_results

# Load model and results
model, results = load_model_and_results(
    model_path="final_model.h5",
    results_path="results.json"
)
```

### Logging to MLflow

```python
from src.mlflow_logger import log_model_to_mlflow

# Log model with metrics
log_model_to_mlflow(
    model_path="best_vgg16_model.keras",
    val_acc=0.8118,
    val_loss=0.4427
)
```

## ⚙️ Configuration

Update `config.py` to customize:
- Data paths
- Model hyperparameters
- Training settings
- Augmentation parameters

## 📊 Model Architecture

The model uses:
- **Base**: VGG16 pre-trained on ImageNet
- **Custom Head**: Flatten → Dense(256) → Dropout(0.5) → Dense(5, softmax)
- **Training Strategy**: Two-phase (frozen → fine-tuned)

## 🎯 Training Strategy

1. **Phase 1 - Feature Extraction** (10 epochs)
   - All VGG16 layers frozen
   - Train only custom classification head
   - Learning rate: 1e-4

2. **Phase 2 - Fine-tuning** (20 epochs)
   - Unfreeze last 4 VGG16 layers
   - Fine-tune entire model
   - Learning rate: 1e-5 (lower)

## 📦 Requirements

```bash
tensorflow
numpy
matplotlib
mlflow
```

## 💡 Tips

1. **Update paths** in `config.py` before training
2. **Monitor training** with callbacks (saved automatically)
3. **Use MLflow** to track experiments
4. **Adjust hyperparameters** in `config.py` for your dataset

## 🔧 Extending the Code

To add new features:
1. Add configurations to `config.py`
2. Implement functionality in appropriate module
3. Update `__init__.py` to export new functions
4. Update this README

---

**Note**: This modular structure makes the code more maintainable, reusable, and easier to test compared to the original notebook format.
