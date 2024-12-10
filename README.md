# 🌈 Mood Detector: AI-Powered Emotion Analysis 🧠

I recently completed a **web-based application** designed to test a machine learning model capable of detecting human emotions from text input. This app combines **AI technology** with a seamless user experience to identify emotional states and recommend relevant places to match the user's mood.

## ✨ Key Features:

- **Accurate Mood Detection**: The app detects 6 mood categories (Sadness, Joy, Love, Anger, Fear, Surprise) and visualizes confidence scores for each mood.
- **Personalized Recommendations**: Based on the detected mood, it suggests cafes aligned with the user's emotional state, providing details such as names, ratings, reviews, and addresses.
- **Interactive Visualization**: Confidence levels for each mood are displayed dynamically through interactive bar charts.
- **User-Friendly Interface**: Built with **Streamlit**, the app offers a simple yet intuitive experience for mood analysis.

## 📂 Data Structure:

### **cafe-data**
- This folder contains CSV files, each representing a mood (e.g., `joy_caffe.csv`, `sadness_caffe.csv`, etc.).
- Each file includes data about cafes that match a specific mood, with columns such as:
  - **Name**
  - **Rating**
  - **Reviews**
  - **Address**
  - **Price Range**
  - **Category**
  - **Atmosphere**
- The application reads these files to provide personalized recommendations based on the predicted mood.

### **mood-data**
- This folder contains quotes and motivational messages categorized by mood.
- Quotes are displayed after a mood is detected, enhancing the user experience with an emotional touch.
- Example: For a detected mood of *Joy*, the app might display an uplifting quote to complement the recommendations.

## 💡 Technologies Used:

### **For Model Development (NLP):**
- **Natural Language Processing (NLP)** techniques were used to preprocess and analyze text data.
- **TensorFlow**: For training and deploying the deep learning model.
- **Keras Tokenizer**: For text tokenization and sequence padding.
- **Word Embedding**: To represent textual data numerically for input into the model.
- **Categorical Encoding**: For converting mood labels into numerical formats.

### **For the Web Application:**
- **Streamlit**: For developing the interactive web-based interface.
- **Pandas**: For processing cafe recommendation data stored in CSV files.
- **Plotly**: For creating engaging and interactive visualizations.

## 🎯 Purpose:

- This web-based app serves as a testing platform for the mood detection model and cafe recommendations.
- The final model, along with the cafe and mood recommendation system, will be integrated into a **mobile application** for a more comprehensive and user-friendly experience.

## 🚀 Conclusion:

This project demonstrates how **AI, machine learning, NLP, and creative data utilization** can work together to enhance emotional awareness and deliver personalized solutions. By combining technical expertise with thoughtful design, this application bridges the gap between technology and human emotion.

Feel free to connect with me to discuss AI, machine learning, NLP, or app development! 😊
