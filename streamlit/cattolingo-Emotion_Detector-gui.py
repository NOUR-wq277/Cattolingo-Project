"""
Streamlit GUI for Cat Emotion Detector - Image & Video Analysis
PyTorch Edition: Implemented with ResNet-50 architecture.
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import tempfile
from collections import Counter
import pandas as pd 
import plotly.express as px


import torch
import torch.nn as nn
from torchvision import transforms, models




MODEL_DEFAULT_PATH = r"D:\NHA-145\models\cattolingo-Emotion_Detector-model.pth" 
INPUT_IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEFAULT_CLASSES = {
    0: 'angry', 
    1: 'disgusted', 
    2: 'happy', 
    3: 'normal', 
    4: 'relaxed', 
    5: 'sad', 
    6: 'scared', 
    7: 'surprised', 
    8: 'uncomfortable'
}
NUM_FIXED_CLASSES = len(DEFAULT_CLASSES)




st.set_page_config(
    page_title="Cat Emotion Detector",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS STYLING (Preserved)
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
    }
    .info-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)




if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = DEFAULT_CLASSES.copy() 
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'num_classes' not in st.session_state:
    st.session_state.num_classes = NUM_FIXED_CLASSES





@st.cache_resource
def load_classification_model(model_path, num_classes):
    """
    Load the trained PyTorch model (ResNet-50 structure) with caching.
    Includes fix for 'weights_only' error by setting it to False.
    """
    try:
        if not os.path.exists(model_path):
            return None, False, f"Model file not found: {model_path}"
        
        model = models.resnet50(pretrained=True) 
        for param in model.parameters():
             param.requires_grad = False
             
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
      
        state_dict_or_model = torch.load(model_path, map_location=DEVICE, weights_only=False) 
        
        if isinstance(state_dict_or_model, dict):
            if 'model_state_dict' in state_dict_or_model:
                 model.load_state_dict(state_dict_or_model['model_state_dict'])
            elif 'state_dict' in state_dict_or_model:
                 model.load_state_dict(state_dict_or_model['state_dict'])
            else:
                 model.load_state_dict(state_dict_or_model)
        elif isinstance(state_dict_or_model, nn.Module):
             model = state_dict_or_model
        else:
             return None, False, "Error: The .pth file content is neither a state dictionary nor a PyTorch model instance."

        model.to(DEVICE)
        model.eval()
        
        return model, True, None
    except Exception as e:
        return None, False, str(e)


def preprocess_frame(frame, target_size):
    """
    Preprocess a single frame (from video or image) using PyTorch transforms.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    
    return input_tensor

def predict_emotion(model, frame, class_names):
    """Perform prediction on a single frame using PyTorch logic"""
    try:
        processed_input = preprocess_frame(frame, INPUT_IMAGE_SIZE)
        
        with torch.no_grad():
            output = model(processed_input)
        
        probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        predicted_class_index = np.argmax(probabilities)
        confidence = probabilities[predicted_class_index]
        predicted_emotion = class_names.get(predicted_class_index, f"Class {predicted_class_index}")
        
        return predicted_emotion, confidence, None
    except Exception as e:
        return "N/A", 0.0, str(e)


def process_video(video_path, model, class_names):
    """Reads a video, applies the model on each frame, and returns the result."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    temp_out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_out_file.name
    temp_out_file.close()

    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    emotion_counts = []
    
    my_bar = st.progress(0, text="Operation in progress. Please wait.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        predicted_emotion, confidence, _ = predict_emotion(model, frame, class_names)
        emotion_counts.append(predicted_emotion)
        
        text = f"Emotion: {predicted_emotion} ({confidence:.2%})"
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
        cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + baseline), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 10 + text_height), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)
        
        progress = frame_count / total_frames if total_frames > 0 else 0
        my_bar.progress(progress, text=f"Analyzing Frame {frame_count}/{total_frames}...")

    cap.release()
    out.release()
    my_bar.empty()
    
    return output_path, Counter(emotion_counts)


def create_probability_chart(df_summary):
    fig = px.bar(df_summary.sort_values(by='Frame Count', ascending=True), 
                 x='Frame Count', y='Emotion', orientation='h',
                 title="Emotion Distribution (Total Frames Analyzed)",
                 color='Frame Count', color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(xaxis_title="Frame Count", yaxis_title="Cat Emotion", height=400, showlegend=False)
    return fig

def create_donut_chart(df_summary):
    fig = px.pie(df_summary, values='Percentage', names='Emotion', 
                 title="Overall Percentage Breakdown of Emotions", hole=0.4,
                 color_discrete_sequence=px.colors.sequential.Mint)
    fig.update_layout(height=300, showlegend=True)
    return fig


def main():
    
    st.markdown('<p class="main-header">🐾 Cat Emotion Detector (Image & Video)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Frame-by-Frame Analysis Powered by Deep Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙ Settings")
        
        model_path_input = st.text_input(
            "Model Path (.pth)",
            value=MODEL_DEFAULT_PATH,
            help="Path to your trained PyTorch model file (.pth)"
        )
        
        st.markdown(f"*الجهاز المستخدم:* <span style='color:#4ECDC4;'>{DEVICE.type.upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"*هيكل النموذج:* <span style='color:#FF6B6B;'>ResNet-50</span>", unsafe_allow_html=True)

        st.markdown("---")
        
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                model, success, error = load_classification_model(model_path_input, st.session_state.num_classes)
                if success:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error(f"❌ Error loading model: {error}")
        
        st.markdown("---")
    
    # Main content area
    if not st.session_state.model_loaded:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.warning("⚠ يرجى تحميل النموذج من الشريط الجانبي (Sidebar) للبدء!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
  
    st.markdown("---")
    st.subheader("📤 Upload Cat Image or Video")
    
    uploaded_file = st.file_uploader(
        "Choose a file...",
        type=['mp4', 'mov', 'avi', 'mkv', 'jpg', 'jpeg', 'png'], 
        help="Upload a file for emotion detection"
    )
    
    if uploaded_file is not None:
        if uploaded_file.type.startswith('image/'):
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        elif uploaded_file.type.startswith('video/'):
            st.video(uploaded_file)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_image_button = st.button("🔮 Analyze Image", use_container_width=True)
        with col2:
            analyze_video_button = st.button("🎬 Analyze Video", use_container_width=True)

        if analyze_image_button:
            if not uploaded_file.type.startswith('image/'):
                st.error("❌ Please upload an IMAGE file to use this button.")
            else:
                with st.spinner("Analyzing image..."):
                    pil_image = Image.open(uploaded_file).convert('RGB')
                    cv2_image = np.array(pil_image)
                   
                    cv2_image = cv2_image[:, :, ::-1].copy() 
                    
                    emotion, confidence, error_msg = predict_emotion(st.session_state.model, cv2_image, st.session_state.class_names)
                    
                    if error_msg:
                        st.error(f"An error occurred during prediction: {error_msg}")
                    else:
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown(f"## 🏆 *{emotion}*")
                        st.markdown(f"### Confidence: *{confidence:.2%}*")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.session_state.prediction_history.append({
                            'emotion': emotion, 'confidence': confidence, 'image': uploaded_file.name 
                        })

        if analyze_video_button:
            if not uploaded_file.type.startswith('video/'):
                st.error("❌ Please upload a VIDEO file to use this button.")
            else:
                with st.spinner("Analyzing video... This might take a while..."):
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}")
                    temp_file.write(uploaded_file.read())
                    video_path = temp_file.name
                    temp_file.close()

                    try:
                        output_video_path, summary = process_video(video_path, st.session_state.model, st.session_state.class_names)
                        os.unlink(video_path)

                        if output_video_path and summary:
                            st.subheader("🎯 Processed Video with Predictions")
                            
                           
                            with open(output_video_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                            st.video(video_bytes)

                            os.unlink(output_video_path)
                            
                            st.subheader("📊 Video Analysis Summary")
                            df_summary = pd.DataFrame(summary.items(), columns=['Emotion', 'Frame Count'])
                            total_frames = df_summary['Frame Count'].sum()
                            df_summary['Percentage'] = (df_summary['Frame Count'] / total_frames)
                            dominant = df_summary.loc[df_summary['Frame Count'].idxmax()]
                            
                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            st.markdown(f"## 🏆 *{dominant['Emotion']}* (Dominant Emotion)")
                            st.markdown(f"### Appeared in *{dominant['Frame Count']}* frames ({dominant['Percentage']:.1%})")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            tab1, tab2 = st.tabs(["📊 Frame Count", "🎯 Breakdown"])
                            with tab1:
                                st.plotly_chart(create_probability_chart(df_summary), use_container_width=True)
                            with tab2:
                                st.plotly_chart(create_donut_chart(df_summary), use_container_width=True)
                            
                            st.session_state.prediction_history.append({
                                'emotion': dominant['Emotion'], 'confidence': dominant['Percentage'], 'image': uploaded_file.name
                            })
                        else:
                            st.error("❌ Video processing failed.")
                    except Exception as e:
                        st.error(f"❌ Error during video analysis: {str(e)}")


    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📜 Analysis History")
        with st.expander("View Summaries", expanded=False):
            for pred in reversed(st.session_state.prediction_history[-10:]):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1: st.write(f"*File:* {pred['image']}")
                with col2: st.write(f"*Result:* {pred.get('emotion', 'N/A')}")
                with col3: st.write(f"*Score:* {pred['confidence']:.2%}")
                st.markdown("---")
        if st.button("🗑 Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    
  
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 20px;'><p>Built with ❤ using Streamlit, OpenCV, and PyTorch</p></div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()