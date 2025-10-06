"""
Utility functions for the Cat Classification project
"""
import json
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def load_saved_results(results_path="results.json"):
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
    
    print("=" * 60)
    print("Saved Results:")
    print("=" * 60)
    print(f"Validation Accuracy: {results['Validation Accuracy']:.4f}")
    print(f"Validation Loss: {results['Validation Loss']:.4f}")
    print("=" * 60)
    
    return results


def load_model_and_results(model_path="final_model.h5", results_path="results.json"):
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


def plot_training_history(history, save_path=None):
    """
    Plot training history (accuracy and loss)
    
    Args:
        history: Training history object from model.fit()
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()


def log_to_mlflow(model_path, val_acc, val_loss, experiment_name="Cats_Classification_VGG16"):
    """
    Log model and metrics to MLflow
    
    Args:
        model_path: Path to saved model
        val_acc: Validation accuracy
        val_loss: Validation loss
        experiment_name: Name of MLflow experiment
    """
    try:
        import mlflow
        import mlflow.keras
        from tensorflow.keras.models import load_model
        
        # Load model
        model = load_model(model_path)
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        with mlflow.start_run():
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.keras.log_model(model, "model")
        
        print("✅ Model and results logged to MLflow successfully!")
        
    except ImportError:
        print("⚠️ MLflow not installed. Install with: pip install mlflow")
    except Exception as e:
        print(f"❌ Error logging to MLflow: {str(e)}")


def display_model_info(model):
    """
    Display model architecture information
    
    Args:
        model: Keras model
    """
    print("\n" + "=" * 60)
    print("Model Architecture Summary")
    print("=" * 60)
    model.summary()
    
    print("\n" + "=" * 60)
    print("Model Configuration")
    print("=" * 60)
    print(f"Total Parameters: {model.count_params():,}")
    print(f"Trainable Parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print(f"Non-trainable Parameters: {sum([tf.size(w).numpy() for w in model.non_trainable_weights]):,}")
    print("=" * 60)
