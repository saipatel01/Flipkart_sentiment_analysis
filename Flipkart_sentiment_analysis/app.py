import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Page Configuration

st.set_page_config(
    page_title="Flipkart Sentiment Analysis",
    layout="centered"
)

st.title("🛒 Flipkart Review Sentiment Analysis")
st.write(
    "This application predicts whether a Flipkart product review is **Positive** or **Negative** "
    "using a machine learning model trained on real customer reviews."
)

# Load Model & Vectorizer

@st.cache_resource
def load_model():
    with open("models/sentiment_model.pkl", "rb") as model_file:
        return pickle.load(model_file)

@st.cache_resource
def load_vectorizer():
    with open("models/tfidf_vectorizer.pkl", "rb") as vec_file:
        return pickle.load(vec_file)

model = load_model()
vectorizer = load_vectorizer()

st.success("Model and TF-IDF vectorizer loaded successfully ")

# Text Preprocessing

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)


# User Input Section

st.header("Enter a Product Review")

user_review = st.text_area(
    "Paste or type a Flipkart product review below:",
    height=150,
    placeholder="Example: The product quality is very poor and not worth the price."
)

# Prediction Logic

if st.button("Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review before clicking Analyze.")
    else:
        # Clean the input
        cleaned_review = clean_text(user_review)

        # Vectorize
        review_vector = vectorizer.transform([cleaned_review])

        # Predict
        prediction = model.predict(review_vector)[0]

        # Display result
        st.subheader("Prediction Result")
        if prediction == "Positive":
            st.success("Sentiment: Positive")
        else:
            st.error("Sentiment: Negative")
