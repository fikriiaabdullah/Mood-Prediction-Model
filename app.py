import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle
import os
import plotly.graph_objs as go

# Set page configuration with a custom icon
st.set_page_config(
    page_title="Mood Detector", 
    page_icon=":brain:", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    color: #333;
}
.mood-title {
    font-size:24px !important;
    color: #2C3E50;
    text-align: center;
}
.stButton>button {
    background-color: #3498DB;
    color: white;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #2980B9;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# Define the correct mood mapping (mapping mood to labels)
MOOD_MAPPING = {
    0: 'Sadness',
    1: 'Joy', 
    2: 'Love', 
    3: 'Anger', 
    4: 'Fear', 
    5: 'Surprise'
}

# Mood color mapping for visualization
MOOD_COLORS = {
    'Sadness': '#3498DB',    # Blue
    'Joy': '#F1C40F',        # Yellow
    'Love': '#E74C3C',       # Red
    'Anger': '#E67E22',      # Orange
    'Fear': '#9B59B6',       # Purple
    'Surprise': '#1ABC9C'    # Turquoise
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

def predict_mood(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Ubah tipe data input menjadi float32
    padded = np.array(padded, dtype=np.float32)
    
    # Get the default signature for prediction
    infer = model.signatures["serve"]
    
    # Predict mood using the pre-trained model
    prediction = infer(tf.convert_to_tensor(padded))["output_0"][0].numpy()
    
    predicted_class = np.argmax(prediction)
    mood = MOOD_MAPPING[predicted_class]
    
    confidence_scores = {MOOD_MAPPING[i]: round(prediction[i] * 100, 2) for i in range(len(prediction))}
    
    return mood, confidence_scores

def create_mood_chart(confidence_scores):
    # Create a bar chart using Plotly
    fig = go.Figure(data=[
        go.Bar(
            x=list(confidence_scores.keys()),
            y=list(confidence_scores.values()),
            marker_color=[MOOD_COLORS.get(mood, '#3498DB') for mood in confidence_scores.keys()]
        )
    ])
    
    fig.update_layout(
        title='Mood Confidence Breakdown',
        xaxis_title='Moods',
        yaxis_title='Confidence (%)',
        title_x=0.5,
        height=400,
        width=600
    )
    
    return fig

# Streamlit app main layout
def main():
    # Sidebar for additional information
    st.sidebar.title("🧠 Mood Detector Guide")
    st.sidebar.info("""
    ### How to Use:
    1. Enter a sentence in the text area
    2. Click 'Detect Mood'
    3. See the predicted mood and confidence scores
    
    ### Tip:
    - Try different types of sentences
    - Longer sentences might give more accurate results
    """)
    
    # Main content
    st.title("🌈 Mood Detector: Understand Your Emotions")
    
    st.markdown("""
    <p class="big-font">
    Explore the emotional undertones of your text. 
    Our AI-powered tool analyzes your words to detect the underlying mood.
    </p>
    """, unsafe_allow_html=True)
    
    # Text input area with custom styling
    text_input = st.text_area(
        "Enter a sentence:", 
        height=200, 
        help="Type or paste a sentence to detect its emotional tone"
    )
    
    # Mood detection button
    col1, col2 = st.columns([3, 1])
    with col1:
        detect_button = st.button("🔍 Detect Mood", use_container_width=True)
    
    # Prediction logic
    if detect_button:
        if model is not None and tokenizer is not None:
            if text_input.strip():
                try:
                    # Predict mood
                    predicted_mood, confidence_scores = predict_mood(text_input)
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"<h2 class='mood-title'>Predicted Mood: {predicted_mood}</h2>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style='background-color:{MOOD_COLORS.get(predicted_mood, '#3498DB')};
                                    color:white;
                                    padding:10px;
                                    border-radius:10px;
                                    text-align:center;'>
                            <h3>🎭 {predicted_mood}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Create and display mood chart
                        mood_chart = create_mood_chart(confidence_scores)
                        st.plotly_chart(mood_chart, use_container_width=True)
                    
                    # Detailed confidence scores
                    st.write("### Confidence Scores:")
                    confidence_cols = st.columns(len(confidence_scores))
                    for i, (mood, score) in enumerate(confidence_scores.items()):
                        with confidence_cols[i]:
                            st.metric(mood, f"{score:.2f}%", help=f"Confidence for {mood} mood")
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a valid sentence.")
        else:
            st.warning("Model not loaded properly. Please check the resources.")
    
    # Additional information
    st.markdown("---")
    st.markdown("""
    ### 💡 About Mood Detection
    This AI-powered tool uses machine learning to analyze the emotional tone of text.
    It can help you understand the underlying sentiment of sentences.
    """)

# Run the app
if __name__ == "__main__":
    main()