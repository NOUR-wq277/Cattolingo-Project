"""
Cat Classification Model - Complete Implementation
VGG16-based CNN for cat breed classification

This file contains all functionality in one place:
- Configuration
- Model architecture
- Data loading and preprocessing
- Training pipeline
- Prediction/Inference
- Utilities and MLflow logging

Usage:
    # Training
    python cat_classification.py --train
    
    # Prediction
    python cat_classification.py --predict path/to/image.jpg
    
    # Or import as module
    from cat_classification import train_model, CatClassifier
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the Cat Classification Model"""
    
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


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def create_model(input_shape=Config.INPUT_SHAPE, num_classes=Config.NUM_CLASSES):
    """
    Create VGG16-based model for cat classification
    
    Args:
        input_shape: Tuple of (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    print("\n" + "=" * 60)
    print("Creating VGG16-based Model")
    print("=" * 60)
    
    # Load pre-trained VGG16 model without top layers
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers initially
    for layer in base_model.layers:
        layer.trainable = False
    
    # Build custom classification head
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create final model
    model = models.Model(inputs=base_model.input, outputs=output)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=Config.LEARNING_RATE_PHASE1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model created with {model.count_params():,} total parameters")
    
    return model


def unfreeze_last_layers(model, num_layers=4, learning_rate=Config.LEARNING_RATE_PHASE2):
    """
    Unfreeze last N layers of the base model for fine-tuning
    
    Args:
        model: Keras model
        num_layers: Number of layers to unfreeze from the end
        learning_rate: Learning rate for fine-tuning
        
    Returns:
        Model with unfrozen layers
    """
    print(f"\nUnfreezing last {num_layers} layers for fine-tuning...")
    
    # Get the base model (VGG16 part)
    base_model = model.layers[1] if len(model.layers) > 1 else model
    
    # Unfreeze last N layers
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ {num_layers} layers unfrozen and model recompiled")
    
    return model


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def create_data_generators(
    train_dir=Config.TRAIN_DATA_DIR,
    val_dir=Config.VAL_DATA_DIR,
    target_size=(Config.INPUT_SHAPE[0], Config.INPUT_SHAPE[1]),
    batch_size=Config.BATCH_SIZE
):
    """
    Create training and validation data generators with augmentation
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        target_size: Target image size (height, width)
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_generator, val_generator)
    """
    print("\n" + "=" * 60)
    print("Creating Data Generators")
    print("=" * 60)
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=Config.ROTATION_RANGE,
        width_shift_range=Config.WIDTH_SHIFT_RANGE,
        height_shift_range=Config.HEIGHT_SHIFT_RANGE,
        zoom_range=Config.ZOOM_RANGE,
        horizontal_flip=Config.HORIZONTAL_FLIP,
        fill_mode=Config.FILL_MODE
    )
    
    # Validation data generator (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Number of classes: {len(train_generator.class_indices)}")
    print(f"✅ Classes: {list(train_generator.class_indices.keys())}")
    
    return train_generator, val_generator


def get_class_names(generator):
    """
    Get class names from data generator
    
    Args:
        generator: Keras ImageDataGenerator
        
    Returns:
        Dictionary mapping class indices to names
    """
    class_indices = generator.class_indices
    return {v: k for k, v in class_indices.items()}


# ============================================================================
# TRAINING
# ============================================================================

def get_callbacks(model_save_path=Config.MODEL_SAVE_PATH):
    """
    Create training callbacks
    
    Args:
        model_save_path: Path to save best model
        
    Returns:
        List of callbacks
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=Config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=Config.REDUCE_LR_FACTOR,
        patience=Config.REDUCE_LR_PATIENCE,
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
    print("\n" + "=" * 70)
    print(" " * 15 + "CAT CLASSIFICATION MODEL TRAINING")
    print("=" * 70)
    
    # Create model
    model = create_model()
    model.summary()
    
    # Load data
    train_generator, val_generator = create_data_generators(train_dir, val_dir)
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # ========== PHASE 1: Feature Extraction ==========
    print("\n" + "=" * 70)
    print(" " * 10 + "PHASE 1: Feature Extraction (Frozen Base Model)")
    print("=" * 70)
    print(f"Training for {Config.EPOCHS_PHASE1} epochs with learning rate {Config.LEARNING_RATE_PHASE1}")
    
    history_phase1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=Config.EPOCHS_PHASE1,
        callbacks=callbacks
    )
    
    # ========== PHASE 2: Fine-tuning ==========
    print("\n" + "=" * 70)
    print(" " * 10 + "PHASE 2: Fine-tuning (Unfreezing Last 4 Layers)")
    print("=" * 70)
    print(f"Training for {Config.EPOCHS_PHASE2} epochs with learning rate {Config.LEARNING_RATE_PHASE2}")
    
    model = unfreeze_last_layers(model, num_layers=4)
    
    history_phase2 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=Config.EPOCHS_PHASE2,
        callbacks=callbacks
    )
    
    # ========== Evaluation ==========
    print("\n" + "=" * 70)
    print(" " * 25 + "FINAL EVALUATION")
    print("=" * 70)
    val_loss, val_acc = model.evaluate(val_generator)
    print(f"\n🎯 Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"📊 Validation Loss: {val_loss:.4f}")
    
    # ========== Save Model and Results ==========
    print("\n" + "=" * 70)
    print(" " * 25 + "SAVING MODEL")
    print("=" * 70)
    model.save(Config.FINAL_MODEL_PATH)
    print(f"✅ Model saved to {Config.FINAL_MODEL_PATH}")
    
    if save_results:
        results = {
            "Validation Accuracy": float(val_acc),
            "Validation Loss": float(val_loss),
            "Phase1_Epochs": Config.EPOCHS_PHASE1,
            "Phase2_Epochs": Config.EPOCHS_PHASE2
        }
        with open(Config.RESULTS_JSON_PATH, "w") as f:
            json.dump(results, f, indent=4)
        print(f"✅ Results saved to {Config.RESULTS_JSON_PATH}")
    
    print("\n" + "=" * 70)
    print(" " * 20 + "🎉 TRAINING COMPLETED! 🎉")
    print("=" * 70 + "\n")
    
    return model, {
        'phase1': history_phase1,
        'phase2': history_phase2
    }


# ============================================================================
# PREDICTION / INFERENCE
# ============================================================================

class CatClassifier:
    """Cat Classification Inference Class"""
    
    def __init__(self, model_path=Config.FINAL_MODEL_PATH):
        """
        Initialize classifier with trained model
        
        Args:
            model_path: Path to saved model file
        """
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        self.input_shape = Config.INPUT_SHAPE
        print(f"✅ Model loaded successfully!")
    
    def preprocess_image(self, img_path):
        """
        Preprocess image for prediction
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        img = image.load_img(
            img_path,
            target_size=(self.input_shape[0], self.input_shape[1])
        )
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Rescale
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    
    def predict(self, img_path, class_names=None):
        """
        Predict cat breed from image
        
        Args:
            img_path: Path to image file
            class_names: Optional dict mapping class indices to names
            
        Returns:
            Tuple of (predicted_class_index, confidence, all_probabilities)
        """
        img_array = self.preprocess_image(img_path)
        predictions = self.model.predict(img_array, verbose=0)
        
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        print("\n" + "=" * 60)
        print(" " * 20 + "PREDICTION RESULTS")
        print("=" * 60)
        
        if class_names:
            predicted_label = class_names[predicted_class]
            print(f"🐱 Predicted Breed: {predicted_label}")
        else:
            print(f"🐱 Predicted Class: {predicted_class}")
        
        print(f"🎯 Confidence: {confidence:.2%}")
        print("\nAll Class Probabilities:")
        for i, prob in enumerate(predictions[0]):
            label = class_names[i] if class_names else f"Class {i}"
            print(f"  {label}: {prob:.2%}")
        print("=" * 60 + "\n")
        
        return predicted_class, confidence, predictions[0]
    
    def predict_batch(self, img_paths, class_names=None):
        """
        Predict multiple images at once
        
        Args:
            img_paths: List of image paths
            class_names: Optional dict mapping class indices to names
            
        Returns:
            List of predictions
        """
        results = []
        for idx, img_path in enumerate(img_paths):
            print(f"\nProcessing image {idx + 1}/{len(img_paths)}: {img_path}")
            result = self.predict(img_path, class_names)
            results.append(result)
        return results


# ============================================================================
# UTILITIES
# ============================================================================

def load_saved_results(results_path=Config.RESULTS_JSON_PATH):
    """
    Load saved training results from JSON file
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        Dictionary containing results
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    print("\n" + "=" * 60)
    print(" " * 20 + "SAVED RESULTS")
    print("=" * 60)
    print(f"Validation Accuracy: {results['Validation Accuracy']:.4f}")
    print(f"Validation Loss: {results['Validation Loss']:.4f}")
    print("=" * 60 + "\n")
    
    return results


def load_model_and_results(model_path=Config.FINAL_MODEL_PATH, results_path=Config.RESULTS_JSON_PATH):
    """
    Load both model and results
    
    Args:
        model_path: Path to saved model
        results_path: Path to results JSON
        
    Returns:
        Tuple of (model, results)
    """
    print("Loading model...")
    model = load_model(model_path)
    print(f"✅ Model loaded from {model_path}")
    
    results = load_saved_results(results_path)
    
    return model, results


def log_to_mlflow(model_path=Config.MODEL_SAVE_PATH, val_acc=None, val_loss=None):
    """
    Log model and metrics to MLflow
    
    Args:
        model_path: Path to saved model
        val_acc: Validation accuracy
        val_loss: Validation loss
    """
    try:
        import mlflow
        import mlflow.keras
        
        # Load model
        model = load_model(model_path)
        
        # Set experiment
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        
        # Start MLflow run
        with mlflow.start_run():
            if val_acc is not None:
                mlflow.log_metric("val_accuracy", val_acc)
            if val_loss is not None:
                mlflow.log_metric("val_loss", val_loss)
            mlflow.keras.log_model(model, "model")
        
        print("✅ Model and results logged to MLflow successfully!")
        print("💡 Run 'mlflow ui' in terminal to view the results")
        
    except ImportError:
        print("⚠️ MLflow not installed. Install with: pip install mlflow")
    except Exception as e:
        print(f"❌ Error logging to MLflow: {str(e)}")


def load_results_and_log_mlflow():
    """Load results from JSON and log to MLflow"""
    try:
        results = load_saved_results()
        val_acc = results.get("Validation Accuracy")
        val_loss = results.get("Validation Loss")
        log_to_mlflow(val_acc=val_acc, val_loss=val_loss)
    except Exception as e:
        print(f"❌ Error: {str(e)}")


# ============================================================================
# MAIN / CLI
# ============================================================================

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cat Classification Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Path to image for prediction')
    parser.add_argument('--load-results', action='store_true', help='Load and display saved results')
    parser.add_argument('--mlflow', action='store_true', help='Log to MLflow')
    
    args = parser.parse_args()
    
    if args.train:
        # Train the model
        model, history = train_model()
    
    elif args.predict:
        # Make prediction
        classifier = CatClassifier()
        
        # Example class names (update based on your dataset)
        class_names = {
            0: "Breed_1",
            1: "Breed_2",
            2: "Breed_3",
            3: "Breed_4",
            4: "Breed_5"
        }
        
        classifier.predict(args.predict, class_names)
    
    elif args.load_results:
        # Load and display results
        load_saved_results()
    
    elif args.mlflow:
        # Log to MLflow
        load_results_and_log_mlflow()
    
    else:
        print("Cat Classification Model")
        print("=" * 60)
        print("Usage examples:")
        print("  python cat_classification.py --train")
        print("  python cat_classification.py --predict path/to/image.jpg")
        print("  python cat_classification.py --load-results")
        print("  python cat_classification.py --mlflow")
        print("=" * 60)


if __name__ == "__main__":
    main()
