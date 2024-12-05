import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(page_title="Mood Detector", layout="centered")

# Define the correct mood mapping (mapping mood to labels)
MOOD_MAPPING = {
    0: 'Sadness',
    1: 'Joy', 
    2: 'Love', 
    3: 'Anger', 
    4: 'Fear', 
    5: 'Surprise'
}

# Load the pre-trained model, tokenizer, and encoder
@st.cache_resource
def load_resources():
    try:
        # Path to the folder containing the SavedModel and other assets
        model_folder = 'Model-Final'  # Path to the 'Model-Final' folder
        
        # Construct the path to saved_model.pb
        saved_model_path = os.path.join(model_folder, 'saved_model.pb')  # Path to the saved_model.pb file
        if not os.path.exists(saved_model_path):
            raise FileNotFoundError(f"SavedModel file does not exist at: {saved_model_path}")
        
        # Load the model using tf.saved_model.load() 
        model = tf.saved_model.load(model_folder)  # Load the full SavedModel folder
        
        # Load the tokenizer (ensure it's the one used during training)
        tokenizer_path = os.path.join(model_folder, 'assets', 'tokenizer.json')  # Path to tokenizer.json
        with open(tokenizer_path, 'r') as f:
            tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
        
        # Load the encoder
        encoder_path = os.path.join(model_folder, 'assets', 'encoder.pkl')  # Path to encoder.pkl
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        
        return model, tokenizer, encoder
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None

model, tokenizer, encoder = load_resources()

# Set the max length for padding (same as used during training)
MAX_LEN = 150

# Function to preprocess the input text
def preprocess_text(text, tokenizer, MAX_LEN):
    try:
        # Tokenize the input text
        tokenized_text = tokenizer.texts_to_sequences([text])
        
        # Pad the tokenized text to match the expected input shape
        padded_text = pad_sequences(tokenized_text, maxlen=MAX_LEN, padding='post', truncating='post')
        
        return padded_text
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

def predict_mood(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')  # Gunakan 'post' untuk padding dan truncating
    
    # Ubah tipe data input menjadi float32
    padded = np.array(padded, dtype=np.float32)
    
    # Get the default signature for prediction
    infer = model.signatures["serve"]
    
    # Predict mood using the pre-trained model
    prediction = infer(tf.convert_to_tensor(padded))["output_0"][0].numpy()  # Use 'output_0' as the output layer
    
    predicted_class = np.argmax(prediction)
    mood = MOOD_MAPPING[predicted_class]
    
    confidence_scores = {MOOD_MAPPING[i]: round(prediction[i] * 100, 2) for i in range(len(prediction))}
    
    return mood, confidence_scores

# Streamlit app
st.title("Mood Detector")
st.write("This app helps you understand how your text might represent different moods.")

st.write("### Instructions:")
st.write("1. Type a sentence or statement in the box below.")
st.write("2. Click on the 'Detect Mood' button to see which mood your text expresses.")
st.write("3. The app will display the mood and confidence level.")

# Text input area
text_input = st.text_area("Enter a sentence:", height=100)

# Display button and output
if st.button("Detect Mood"):
    if model is not None and tokenizer is not None:
        if text_input.strip():  # Check if the input is not empty
            predicted_mood, confidence = predict_mood(text_input)
            if predicted_mood:
                st.write(f"### Predicted Mood: **{predicted_mood}**")
                
                # Display confidence for each mood
                st.write("### Confidence Scores:")
                for mood, score in confidence.items():
                    st.write(f"{mood}: {score:.2f}%")
            else:
                st.warning("Something went wrong with the prediction. Please try again.")
        else:
            st.warning("Please enter a valid sentence.")
    else:
        st.warning("Model, tokenizer, or encoder is not loaded properly. Please check the resources.")

# Add some fun info and encourage kids to experiment
st.write("---")
st.write("### Fun Fact:")
st.write("Did you know that people express their moods in different ways? This tool helps you see how your words can show emotions!")

# Footer for encouragement
st.write("---")
st.write("Give it a try! Type a sentence, and let's see which mood your words express!")
