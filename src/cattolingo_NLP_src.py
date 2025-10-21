import torch
import torch.nn as nn
import re
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import streamlit as st
import os

class NLPConfig:
    MODEL_WEIGHTS_PATH = r"D:\NHA-145\models\model.safetensors" 
    CONFIG_FILE_PATH = r"D:\NHA-145\models\config.json"
    VOCAB_FILE_PATH = r"D:\NHA-145\models\vocab.json" 
    MERGES_FILE_PATH = r"D:\NHA-145\models\merges.txt"
    TOKENIZER_CONFIG_PATH = r"D:\NHA-145\models\tokenizer_config.json"
    SPECIAL_TOKENS_PATH = r"D:\NHA-145\models\special_tokens_map.json"

    MAX_LEN = 192 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ID2LABEL = {
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

def light_clean(s):
    s = str(s)
    s = re.sub(r'http\S+|www\S+|https\S+', '', s) 
    s = re.sub(r'\s+', ' ', s).strip()           
    return s

@st.cache_resource
def load_nlp_model():
    model_directory = os.path.dirname(NLPConfig.CONFIG_FILE_PATH) 
    
    print(f"Loading NLP model from directory: {model_directory}")
    
    try:
        if not os.path.exists(model_directory):
            st.error(f"Model directory not found: {model_directory}")
            return None, None
            
        tokenizer = RobertaTokenizer.from_pretrained(model_directory)
        model = RobertaForSequenceClassification.from_pretrained(model_directory)
        
        model.to(NLPConfig.DEVICE)
        model.eval()
        print("NLP model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading NLP model: {e}")
        return None, None

def predict_emotion(text, model, tokenizer):
    if not text or not text.strip():
        return None

    cleaned_text = light_clean(text)
    
    enc = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=NLPConfig.MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = enc['input_ids'].to(NLPConfig.DEVICE)
    attention_mask = enc['attention_mask'].to(NLPConfig.DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    pred_idx = torch.argmax(outputs.logits, dim=1).cpu().item()
    
    return NLPConfig.ID2LABEL.get(pred_idx, "Unknown")