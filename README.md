Catto-Lingo 😼: AI-Powered Cat Emotion Analyzer

Project Summary

Catto-Lingo is an intelligent, multi-modal, deep-learning-based system designed to bridge the communication gap between humans and cats. By translating feline vocalizations, visual cues, and textual descriptions of behavior into understandable emotions, this application helps owners and veterinarians better understand the emotional and physical well-being of their pets.

Technical Approach

    Audio Analysis: The project utilizes two distinct pipelines for audio classification:

        Traditional Machine Learning: Models like SVM are trained on engineered features (e.g., MFCCs).

        Deep Learning: A Convolutional Neural Network (CNN) is trained on Spectrogram images generated from the audio files.

    Vision Analysis: A Computer Vision model analyzes images and live video to interpret the cat's visual emotional state (e.g., relaxed, agitated).

    Text (NLP) Analysis: A large-scale Transformer model (roberta-large) was fine-tuned on a dataset of textual descriptions of cat behavior (e.g., "the cat is hissing," "he is purring") to predict the associated emotion. This 1.5GB model achieved 92% accuracy and is hosted on Hugging Face Hub.

    Data Augmentation: To prevent overfitting, datasets are artificially expanded. This includes frequency/time masking for audio, image transformations for vision, and synonym replacement for text.

Project Structure

    src/: Contains the final, clean, and reusable Python source code. Each model has its own processor file (e.g., audio_cnn_processor.py, nlp_processor.py).

    notebooks/: Contains Jupyter Notebooks used for experimentation, research, and model training (e.g., cattolingo-text-model.ipynb).

    models/: Contains the final, trained model files (.h5, .pkl). Smaller models (<100MB) are tracked by Git LFS. Ultra-large models (like the 1.5GB NLP model) are hosted externally on Hugging Face Hub and downloaded automatically by the application.

    data/: Intended for datasets, but data is provided via external links to keep the repository light.

    main.py: The main FastAPI application file for the backend API.

    streamlit_app.py: A multi-tab Streamlit application that combines all models for testing and demonstration.

Setup and Installation

    Clone the repository:
    Bash

git clone https://github.com/nhahub/NHA-145.git
cd NHA-145

Set up the environment:
Bash

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Download Model Files:

    Small Models (LFS): Make sure you have Git LFS installed (git lfs install). The .h5 and .pkl files will be downloaded automatically during the clone or pull process.
    Bash

    git lfs pull

    Large NLP Model (Hugging Face): The 1.5GB roberta-large model is downloaded automatically from Hugging Face Hub the first time you run the Streamlit app. (Requires an internet connection).

Run the application:
Bash

# To run the API
uvicorn main:app --reload

# To run the Streamlit testing app
streamlit run streamlit_app.py
