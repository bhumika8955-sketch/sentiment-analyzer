import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🤖", layout="centered")

# 2. CACHE DATASET TRAINING (Keeps the app fast)
@st.cache_resource
def train_model():
    data = {
        'text': [
            "This product is amazing, I love it so much!",
            "Best purchase I have ever made, highly recommend.",
            "Excellent quality, functions perfectly and smoothly.",
            "Wow, incredible customer service and fantastic item.",
            "Great value for money, exceeded my expectations.",
            "Terrible item, it broke on the first day of use.",
            "Total waste of money, completely cheap garbage.",
            "Horrible experience, stopped working after two hours.",
            "Worst quality ever, do not buy this product!",
            "Disappointed, did not match the description at all."
        ] * 10,
        'sentiment': ['positive', 'positive', 'positive', 'positive', 'positive', 
                      'negative', 'negative', 'negative', 'negative', 'negative'] * 10
    }
    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vectors = vectorizer.fit_transform(df['text'])
    model = LogisticRegression()
    model.fit(X_vectors, df['sentiment'])
    return vectorizer, model

vectorizer, model = train_model()

# 3. INTERACTIVE DISPLAY LAYOUT
st.title("🤖 Live Sentiment Analyzer System")
st.write("Type any product feedback phrase below to see your machine learning model sort it dynamically!")

user_input = st.text_input("Review Text Entry:", placeholder="Type a sentence here... (e.g., 'Wow, this is an incredible gadget!')")

if st.button("Submit Analysis", type="primary"):
    if user_input.strip() != "":
        # Run prediction pipeline
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]
        
        # Display clear visual status card
        if prediction == 'positive':
            st.success("### ✨ POSITIVE FEEDBACK DETECTED! 😄")
        else:
            st.error("### ⚠️ NEGATIVE FEEDBACK DETECTED! 😡")
    else:
        st.warning("Please enter some text first!")
