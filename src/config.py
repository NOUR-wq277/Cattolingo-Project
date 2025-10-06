"""
Configuration file for the Cat Classification Model
"""
import os

# Model Configuration
INPUT_SHAPE = (128, 128, 3)
NUM_CLASSES = 5
BATCH_SIZE = 32
LEARNING_RATE_PHASE1 = 1e-4
LEARNING_RATE_PHASE2 = 1e-5

# Training Configuration
EPOCHS_PHASE1 = 10  # Feature extraction
EPOCHS_PHASE2 = 20  # Fine-tuning
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.2

# Data Augmentation Parameters
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
FILL_MODE = 'nearest'

# Paths (Update these according to your setup)
TRAIN_DATA_DIR = r"D:\depi_project\cats_dataset_balanced\train"
VAL_DATA_DIR = r"D:\depi_project\cats_dataset_balanced\val"
MODEL_SAVE_PATH = "best_vgg16_model.keras"
FINAL_MODEL_PATH = "final_model.h5"
RESULTS_JSON_PATH = "results.json"

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME = "Cats_Classification_VGG16"
