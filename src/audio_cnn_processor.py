import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

IMG_HEIGHT = 232
IMG_WIDTH = 231
MODEL_PATH = "models/last_model.h5"
CLASS_NAMES = ["Angry", "Happy", "Resting", "Sad", "Stressed"]

def load_cnn_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("CNN model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        return None

def create_spectrogram(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        temp_image_path = "temp_spectrogram.png"
        plt.figure(figsize=(IMG_WIDTH/100, IMG_HEIGHT/100))
        librosa.display.specshow(S_dB, sr=sr, fmax=8000)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(temp_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return temp_image_path
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return None

def preprocess_image(image_path):
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        if os.path.exists(image_path):
            os.remove(image_path)
            
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_with_cnn_model(model, audio_file_path):
    if model is None:
        return {"error": "Model not loaded."}
        
    spectrogram_path = create_spectrogram(audio_file_path)
    if spectrogram_path is None:
        return {"error": "Failed to create spectrogram."}
        
    processed_img = preprocess_image(spectrogram_path)
    if processed_img is None:
        return {"error": "Failed to preprocess image."}
        
    predictions = model.predict(processed_img)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return {
        "emotion": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }

if __name__ == '__main__':
    cnn_model = load_cnn_model()
    test_audio = 'path/to/your/test_audio.wav' 

    if os.path.exists(test_audio):
        result = predict_with_cnn_model(cnn_model, test_audio)
        print("\n--- Prediction Result ---")
        print(result)
    else:
        print(f"\\nTest audio file not found at: {test_audio}")
        print("Please update the 'test_audio' variable to a valid path.")