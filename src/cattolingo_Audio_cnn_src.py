import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import io
import tensorflow as tf
import os

CLASS_NAMES = ["Angry", "Happy", "Stressed", "Sad", "Resting"]

def load_cnn_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'models', 'cattolingo-Audio_cnn-model.h5')
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(" CNN Cat Emotion Model loaded successfully (safe mode).")
        return model
    except Exception as e:
        print(f" Error loading model: {e}")
        return None

def create_spectrogram_tensor(audio_file, target_size=(232, 231)):
    try:
        y, sr = librosa.load(audio_file, sr=44100)
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            hop_length=256,
            fmax=20000
        )
        S_DB = librosa.power_to_db(S, ref=np.max)

        fig = plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_DB, sr=sr, cmap="magma")
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print(f"❌ Error creating spectrogram tensor: {e}")
        return None

def get_prediction(model, audio_path):
    tensor = create_spectrogram_tensor(audio_path)
    if tensor is None:
        return None
    preds = model.predict(tensor)
    confidence = float(np.max(preds))
    label = CLASS_NAMES[int(np.argmax(preds))]
    print(f"🎯 Prediction: {label} ({confidence*100:.2f}%)")
    return label, confidence

if __name__ == "__main__":
    print("Testing src file... Model loading will be relative.")