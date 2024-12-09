import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle
import os
import pandas as pd
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

def load_random_quote(mood):
    """
    Load a random quote for the given mood from the Quotes_Mood CSV file
    
    Args:
        mood (str): The detected mood
    
    Returns:
        str: A random quote for the specified mood
    """
    try:
        # Construct the path to the Quotes_Mood CSV file
        quotes_filename = os.path.join('Cafe-Data', 'Quotes_Mood.csv')
        
        # Read the CSV file
        df = pd.read_csv(quotes_filename)
        
        # Filter quotes by mood (case-insensitive)
        mood_quotes = df[df['Mood'].str.lower() == mood.lower()]
        
        # If quotes are found, return a random quote
        if not mood_quotes.empty:
            random_quote = mood_quotes.sample(n=1).iloc[0]['Quotes']
            return random_quote
        else:
            return "No quote found for this mood."
    
    except FileNotFoundError:
        st.warning("Quotes file not found.")
        return "Quote file is missing."
    except Exception as e:
        st.error(f"An error occurred while loading quotes: {e}")
        return "Error retrieving quote."

def load_cafe_data(mood):
    # Convert mood to lowercase and find the corresponding CSV file
    mood_lower = mood.lower()
    csv_filename = os.path.join('Cafe-Data', f"{mood_lower}_caffe.csv")
    
    try:
        # Try to load the CSV file
        df = pd.read_csv(csv_filename)
        return df
    except FileNotFoundError:
        st.warning(f"No cafe data found for {mood} mood.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading cafe data: {e}")
        return None

def display_cafe_recommendations(df):
    if df is None or df.empty:
        st.warning("No cafe recommendations available.")
        return
    
    # Select relevant columns for display
    display_columns = ['Title', 'Rating', 'Address', 'Category']
    
    # Ensure selected columns exist in the DataFrame
    available_columns = [col for col in display_columns if col in df.columns]
    
    if not available_columns:
        st.warning("No suitable columns found for display.")
        return
    
    # Display cafes in a more interactive way
    st.subheader("🍵 Cafe Recommendations")
    
    # Add pagination
    cafes_per_page = 10
    total_cafes = len(df)
    
    # Initialize page number in session state if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # Function to increment page
    def increment_page():
        if st.session_state.current_page < max(1, (total_cafes - 1) // cafes_per_page + 1):
            st.session_state.current_page += 1
    
    # Function to decrement page
    def decrement_page():
        if st.session_state.current_page > 1:
            st.session_state.current_page -= 1
    
    # Create columns for pagination
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.button('Previous', on_click=decrement_page, 
                  disabled=st.session_state.current_page == 1)
    
    with col2:
        st.write(f"Page {st.session_state.current_page} of {max(1, (total_cafes - 1) // cafes_per_page + 1)}")
    
    with col3:
        st.button('Next', on_click=increment_page, 
                  disabled=st.session_state.current_page == max(1, (total_cafes - 1) // cafes_per_page + 1))
    
    # Calculate start and end indices for current page
    start_idx = (st.session_state.current_page - 1) * cafes_per_page
    end_idx = min(start_idx + cafes_per_page, total_cafes)
    
    # Display cafes for the current page
    st.markdown(f"**Showing {start_idx + 1} - {end_idx} of {total_cafes} cafes**")
    
    for index, row in df.iloc[start_idx:end_idx].iterrows():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {row['Title']}")
            st.write(f"**Rating:** {row.get('Rating', 'N/A')}")
        
        with col2:
            st.write(f"**Address:** {row.get('Address', 'Address not available')}")
            st.write(f"**Category:** {row.get('Category', 'N/A')}")
        
        st.markdown("---")

def main():
    # Sidebar for additional information
    st.sidebar.title("🧠 Mood Detector Guide")
    st.sidebar.info("""
    ### How to Use:
    1. Enter a sentence in the text area
    2. Click 'Detect Mood'
    3. See the predicted mood and cafe recommendations
    
    ### Tip:
    - Try different types of sentences
    - Longer sentences might give more accurate results
    """)
    
    # Main content
    st.title("🌈 Mood Detector: Understand Your Emotions")
    
    st.markdown("""
    <p class="big-font">
    Explore the emotional undertones of your text and get personalized cafe recommendations!
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
                    col1, col2 = st.columns([2, 2])
                    
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
                        
                        # Load and display random quote
                        random_quote = load_random_quote(predicted_mood)
                        st.markdown(f"""
                            <div style='text-align:center; 
                                        margin-top:15px; 
                                        padding:10px; 
                                        background-color:rgba(0,0,0,0.05); 
                                        border-radius:10px;'>
                                <h4 style='color:#555;'>💬 Today's word for you</h4>
                                <p style='font-size:18px; 
                                        font-style:italic; 
                                        line-height:1.5; 
                                        color:#333;'>
                                    {random_quote}
                                </p>
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
                    
                    # Load and display cafe recommendations
                    cafe_data = load_cafe_data(predicted_mood)
                    if cafe_data is not None:
                        display_cafe_recommendations(cafe_data)
                
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
    This AI-powered tool uses machine learning to analyze the emotional tone of text
    and provide personalized cafe recommendations based on your mood!
    """)

# Run the app
if __name__ == "__main__":
    main()