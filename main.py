import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
import nltk
from nltk.corpus import stopwords
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

# load saved model
model = AutoModelForSequenceClassification.from_pretrained('saved_mental_status_bert')
tokenizer = AutoTokenizer.from_pretrained('saved_mental_status_bert')
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# custom function
# Get English stopwords from NLTK
stop_words = set(stopwords.words('english'))
def clean_statement(statement):
    # Convert to lowercase
    statement = statement.lower()

    # Remove special characters (punctuation, non-alphabetic characters)
    statement = re.sub(r'[^\w\s]', '', statement)

    # Remove numbers (optional, depending on your use case)
    statement = re.sub(r'\d+', '', statement)

    # Tokenize the statement (split into words)
    words = statement.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Rejoin words into a cleaned statement
    cleaned_statement = ' '.join(words)

    return cleaned_statement

# Detection System
def detect_anxiety(text):
    cleaned_text = clean_statement(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=200)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]

# FastAPI for serving the prediction API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Or "*" to allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(text_input: TextInput):
    text = text_input.text
    predicted_class = detect_anxiety(text)
    return {"predicted_status": predicted_class}

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
