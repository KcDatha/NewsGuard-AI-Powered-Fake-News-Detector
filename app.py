import streamlit as st
import pickle
import numpy as np

# Set page title and icon
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5f7fa;
        }
        .title {
            color: #1f77b4;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .prediction {
            font-size: 20px;
            font-weight: bold;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .fake {
            background-color: #ffcccc;
            color: red;
        }
        .real {
            background-color: #ccffcc;
            color: green;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display title
st.markdown('<h1 class="title">📰 Fake News Detector</h1>', unsafe_allow_html=True)

# Load the trained model & vectorizer with error handling
try:
    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
except FileNotFoundError:
    st.error("🚨 **Error:** Model files not found. Ensure 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl' exist in the current directory.")
    st.stop()

# User input for news title
news_title = st.text_input("🔎 Enter a news title:", placeholder="Type or paste a news headline here...")

# Prediction button
if st.button("🔍 Check Authenticity"):
    if not news_title.strip():
        st.warning("⚠️ **Please enter a news title before checking.**")
    else:
        # Transform the input title
        transformed_title = vectorizer.transform([news_title])
        
        # Predict and get confidence scores
        prediction = model.predict(transformed_title)[0]
        probs = model.predict_proba(transformed_title)[0]  # Get probability scores

        fake_prob = probs[0] * 100  # Fake news probability
        real_prob = probs[1] * 100  # Real news probability

        # Display results with styling
        if prediction == 1:
            st.markdown(f'<p class="prediction real">✅ This news is REAL! (Confidence: {real_prob:.2f}%)</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="prediction fake">❌ This news is FAKE! (Confidence: {fake_prob:.2f}%)</p>', unsafe_allow_html=True)

        # Add a probability progress bar
        st.markdown("### 🔥 Confidence Levels:")
        st.progress(int(real_prob if prediction == 1 else fake_prob))
        
        # Show probability bars
        st.markdown(f"**🟢 Real News Confidence:** {real_prob:.2f}%")
        st.markdown(f"**🔴 Fake News Confidence:** {fake_prob:.2f}%")
