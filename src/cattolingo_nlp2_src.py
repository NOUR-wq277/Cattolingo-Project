import torch
import torch.nn as nn
import re
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import streamlit as st
import os

class NLPConfig:
    
    MODEL_PATH = "Nour87/cattolingo-nlp-roberta-large" 
    
  
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
   
    model_path_or_id = NLPConfig.MODEL_PATH
    
    print(f"Loading NLP model from Hugging Face Hub: {model_path_or_id}")
    
    try:
       
        tokenizer = RobertaTokenizer.from_pretrained(model_path_or_id)
        model = RobertaForSequenceClassification.from_pretrained(model_path_or_id)
        
        model.to(NLPConfig.DEVICE)
        model.eval()
        print("NLP model loaded successfully from Hub.")
        return model, tokenizer
    except Exception as e:
       
        st.error(f"Error loading NLP model from Hub: {e}")
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