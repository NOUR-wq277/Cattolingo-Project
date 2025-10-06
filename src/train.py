"""
Training script for Cat Classification Model
"""
import json
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from model import create_model, unfreeze_last_layers
from data import create_data_generators
from config import (
    EPOCHS_PHASE1, EPOCHS_PHASE2,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    MODEL_SAVE_PATH, FINAL_MODEL_PATH, RESULTS_JSON_PATH
)


def get_callbacks(model_save_path=MODEL_SAVE_PATH):
    """
    Create training callbacks
    
    Args:
        model_save_path: Path to save best model
        
    Returns:
        List of callbacks
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    return [early_stopping, reduce_lr, checkpoint]


def train_model(train_dir=None, val_dir=None, save_results=True):
    """
    Complete training pipeline with two phases:
    1. Feature extraction (frozen base model)
    2. Fine-tuning (unfrozen last layers)
    
    Args:
        train_dir: Path to training data
        val_dir: Path to validation data
        save_results: Whether to save results to JSON
        
    Returns:
        Trained model and training history
    """
    print("=" * 60)
    print("Creating Model...")
    print("=" * 60)
    model = create_model()
    model.summary()
    
    print("\n" + "=" * 60)
    print("Loading Data...")
    print("=" * 60)
    train_generator, val_generator = create_data_generators(train_dir, val_dir)
    
    callbacks = get_callbacks()
    
    # ========== PHASE 1: Feature Extraction ==========
    print("\n" + "=" * 60)
    print("PHASE 1: Feature Extraction (Frozen Base Model)")
    print("=" * 60)
    history_phase1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks
    )
    
    # ========== PHASE 2: Fine-tuning ==========
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning (Unfreezing Last 4 Layers)")
    print("=" * 60)
    model = unfreeze_last_layers(model, num_layers=4)
    
    history_phase2 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks
    )
    
    # ========== Evaluation ==========
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    val_loss, val_acc = model.evaluate(val_generator)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # ========== Save Model and Results ==========
    print("\n" + "=" * 60)
    print("Saving Model...")
    print("=" * 60)
    model.save(FINAL_MODEL_PATH)
    print(f"✅ Model saved to {FINAL_MODEL_PATH}")
    
    if save_results:
        results = {
            "Validation Accuracy": float(val_acc),
            "Validation Loss": float(val_loss)
        }
        with open(RESULTS_JSON_PATH, "w") as f:
            json.dump(results, f, indent=4)
        print(f"✅ Results saved to {RESULTS_JSON_PATH}")
    
    return model, {
        'phase1': history_phase1,
        'phase2': history_phase2
    }


if __name__ == "__main__":
    model, history = train_model()
