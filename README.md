# Catto-Lingo
Catto-Lingo: AI-Powered Cat Emotion Analyzer 😼

Project Summary

Catto-Lingo is an intelligent, multi-modal system designed to bridge the profound communication gap between humans and their feline companions. By leveraging deep learning models to interpret complex audio and visual cues, the application translates a cat's vocalizations and expressions into a set of understandable emotional and physical states. The ultimate goal is to foster a deeper human-animal bond by providing owners and veterinary professionals with actionable insights into a cat's well-being, moving beyond guesswork to data-driven understanding.

The Problem It Solves

The vocalizations and body language of cats are nuanced and often misinterpreted. A purr, for instance, is not always a sign of contentment; it can also indicate stress or pain. This ambiguity can lead to misunderstandings and delayed recognition of health or behavioral issues. Catto-Lingo provides a scientific tool to decode these signals, offering a clearer picture of a cat's state of mind and helping to identify potential problems before they escalate.

Key Features

    Multi-modal Analysis: The system's core strength lies in its ability to process multiple data streams for a holistic analysis.

        Audio Analysis: Two distinct pipelines were developed and compared to find the optimal approach. The first uses traditional machine learning models (like SVMs) on engineered features such as MFCCs and Chroma. The second, more advanced pipeline, uses a Convolutional Neural Network (CNN) trained on Spectrograms, which are visual representations of the audio, to learn features automatically.

        Vision Analysis: A YOLOv8 model is used to first detect the cat and its key facial landmarks (eyes, ears, whiskers) and body posture from an image or live video stream. These features are then fed into a classification model to interpret the cat's visual emotional state (e.g., relaxed, agitated, fearful).

    Dual Prediction Mode: To offer both transparency and a definitive answer, the app functions in two modes.

        Expert Panel Mode: This mode displays the individual predictions from all three underlying models (Traditional Audio ML, Audio CNN, and Vision). It's designed for users who want to see and compare the "opinions" of each specialized AI expert.

        Fused Verdict Mode: This mode utilizes a Late Fusion technique. It takes the probability scores from the best-performing audio model and the vision model and averages them to produce a single, robust, and typically more accurate final prediction.

System Architecture & Technical Stack

The project is built with a modern, scalable architecture and follows professional MLOps practices.

    Overall Architecture: The system uses a classic client-server model. A Flutter mobile app acts as the client, responsible for capturing user input (audio/video) and sending HTTP requests to a FastAPI backend API. This backend, deployed on Azure App Service, houses the ML models and all the business logic, processes the data, and returns a JSON response to the app.

    MLOps Workflow: The project is built on a robust MLOps foundation to ensure reproducibility and reliability.

        Git / GitHub is used for all source code versioning and team collaboration.

        DVC (Data Version Control) handles the versioning of large datasets and final model artifacts, keeping the Git repository lightweight.

        MLflow is used to meticulously track all training experiments, including hyperparameters, performance metrics, and model artifacts, creating a comprehensive and comparable log of every model iteration.

    Technologies Used:

        Machine Learning: TensorFlow/Keras, Scikit-learn, Librosa, OpenCV

        Backend: FastAPI, Uvicorn

        Frontend: Flutter

        Deployment & MLOps: Azure App Service, Git, DVC, MLflow

Future Work

    Expand Emotion Set: Increase the number of recognized emotions to include more nuanced states like boredom, curiosity, and loneliness.

    Long-Term Health Monitoring: Incorporate a feature to track a cat's vocal and behavioral patterns over time to flag potential chronic health issues.

    Breed-Specific Models: Train specialized models for different cat breeds to account for variations in vocalization and behavior.


