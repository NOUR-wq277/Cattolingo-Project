# Catto-Lingo 😼: AI-Powered Cat Emotion Analyzer

## Project Summary
Catto-Lingo is an intelligent, deep-learning-based system designed to bridge the communication gap between humans and cats. By translating feline vocalizations and visual cues into understandable emotions, this application helps owners and veterinarians better understand the emotional and physical well-being of their pets.

---

## Technical Approach

* **Audio Analysis**: The project utilizes two distinct pipelines for audio classification:
    1.  **Traditional Machine Learning**: Models like SVM are trained on engineered features (e.g., MFCCs).
    2.  **Deep Learning**: A Convolutional Neural Network (CNN) is trained on Spectrogram images generated from the audio files.
* **Vision Analysis**: A Computer Vision model analyzes images and live video to interpret the cat's visual emotional state (e.g., relaxed, agitated).
* **Data Augmentation**: To prevent overfitting, the image dataset is artificially expanded using techniques like frequency and time masking.

---

## Project Structure

* **`src/`**: Contains the final, clean, and reusable Python source code. Each model has its own processor file (e.g., `audio_cnn_processor.py`).
* **`notebooks/`**: Contains Jupyter Notebooks used for experimentation, research, and model training.
* **`models/`**: Contains the final, trained model files (`.h5`, `.pkl`). These are tracked by Git LFS.
* **`data/`**: Intended for datasets, but data is provided via external links to keep the repository light.
* **`main.py`**: The main FastAPI application file for the backend API.
* **`streamlit_app.py`**: A simple Streamlit application for testing and demonstration.

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nhahub/NHA-145.git](https://github.com/nhahub/NHA-145.git)
    cd NHA-145
    ```

2.  **Set up the environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Download Large Model Files:**
    Make sure you have Git LFS installed (`git lfs install`). The model files will be downloaded automatically during the clone or pull process.
    ```bash
    git lfs pull
    ```

4.  **Run the application:**
    ```bash
    # To run the API
    uvicorn main:app --reload

    # To run the Streamlit testing app
    streamlit run streamlit_app.py
    ```
