"""
MLflow logging utilities
"""
import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model
from config import MLFLOW_EXPERIMENT_NAME, MODEL_SAVE_PATH


def log_model_to_mlflow(
    model_path=MODEL_SAVE_PATH,
    val_acc=None,
    val_loss=None,
    experiment_name=MLFLOW_EXPERIMENT_NAME
):
    """
    Log trained model and metrics to MLflow
    
    Args:
        model_path: Path to saved model
        val_acc: Validation accuracy
        val_loss: Validation loss
        experiment_name: Name of MLflow experiment
    """
    try:
        # Load model
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
        
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        with mlflow.start_run():
            # Log metrics if provided
            if val_acc is not None:
                mlflow.log_metric("val_accuracy", val_acc)
                print(f"✅ Logged validation accuracy: {val_acc:.4f}")
            
            if val_loss is not None:
                mlflow.log_metric("val_loss", val_loss)
                print(f"✅ Logged validation loss: {val_loss:.4f}")
            
            # Log model
            mlflow.keras.log_model(model, "model")
            print("✅ Model logged to MLflow successfully!")
        
        print(f"\n🎉 MLflow experiment '{experiment_name}' completed!")
        print("Run 'mlflow ui' in terminal to view the results")
        
    except Exception as e:
        print(f"❌ Error logging to MLflow: {str(e)}")
        raise


def load_results_and_log():
    """
    Load results from JSON and log to MLflow
    """
    import json
    from config import RESULTS_JSON_PATH
    
    try:
        with open(RESULTS_JSON_PATH, "r") as f:
            results = json.load(f)
        
        val_acc = results.get("Validation Accuracy")
        val_loss = results.get("Validation Loss")
        
        log_model_to_mlflow(val_acc=val_acc, val_loss=val_loss)
        
    except FileNotFoundError:
        print(f"❌ Results file not found: {RESULTS_JSON_PATH}")
        print("Please train the model first or provide metrics manually")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    # Log existing model and results to MLflow
    load_results_and_log()
