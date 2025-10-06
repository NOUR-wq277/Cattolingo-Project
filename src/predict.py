"""
Inference script for Cat Classification Model
"""
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from config import FINAL_MODEL_PATH, INPUT_SHAPE


class CatClassifier:
    """Cat Classification Inference Class"""
    
    def __init__(self, model_path=FINAL_MODEL_PATH):
        """
        Initialize classifier with trained model
        
        Args:
            model_path: Path to saved model file
        """
        self.model = load_model(model_path)
        self.input_shape = INPUT_SHAPE
        print(f"✅ Model loaded from {model_path}")
    
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
        
        if class_names:
            predicted_label = class_names[predicted_class]
            print(f"Predicted: {predicted_label} (Confidence: {confidence:.2%})")
        else:
            print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2%})")
        
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
        for img_path in img_paths:
            result = self.predict(img_path, class_names)
            results.append(result)
        return results


def main():
    """Example usage"""
    # Load classifier
    classifier = CatClassifier()
    
    # Example class names (update based on your dataset)
    class_names = {
        0: "Breed_1",
        1: "Breed_2",
        2: "Breed_3",
        3: "Breed_4",
        4: "Breed_5"
    }
    
    # Example prediction (update with actual image path)
    # predicted_class, confidence, probs = classifier.predict("path/to/image.jpg", class_names)
    print("\n⚠️ To use this script, provide an image path in the main() function")


if __name__ == "__main__":
    main()
