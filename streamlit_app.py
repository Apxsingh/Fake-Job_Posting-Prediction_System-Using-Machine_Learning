import re
import string
import joblib
import streamlit as st
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Page Configuration


st.set_page_config(
    page_title="Fake Job Posting Detection",
    page_icon="@",
    layout="centered"
)


# Download NLTK Resources


nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# Load Saved Model and TF-IDF


@st.cache_resource
def load_model():
    model = joblib.load("models/logistic_regression.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    return model, tfidf


model, tfidf = load_model()


# Text Preprocessing


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Prediction Function


def predict_job(job_text):

    cleaned = clean_text(job_text)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    return prediction, probability



# UI


st.title("Fake Job Posting Detection System")

st.markdown("""
This application uses **Machine Learning** and **Natural Language Processing (NLP)** to determine whether a job posting is **Genuine** or **Fraudulent**.
""")

st.divider()

job_description = st.text_area(
    "Paste Job Description",
    height=250,
    placeholder="Paste the complete job description here..."
)

predict = st.button(
    "Predict",
    use_container_width=True
)

if predict:

    if job_description.strip() == "":

        st.warning("Please enter a job description.")

    else:

        prediction, probability = predict_job(job_description)

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 0:

            st.success("Genuine Job Posting")

        else:

            st.error("Fraudulent Job Posting")

        st.write("### Confidence Scores")

        genuine = float(probability[0])
        fraud = float(probability[1])

        st.write(f"**Genuine:** {genuine:.2%}")
        st.progress(genuine)

        st.write(f"**Fraudulent:** {fraud:.2%}")
        st.progress(fraud)

        st.divider()

        st.subheader("Model Used")

        st.info("Logistic Regression + TF-IDF Vectorizer")

st.divider()

st.caption("Developed by Arpit Singh Tomar | Fake Job Posting Detection Project")