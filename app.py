# Import Required Libraries

import re
import string
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load TF-IDF Vectorizer
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Load Best Model (Logistic Regression)
model = joblib.load("models/logistic_regression.pkl")

# Initialize preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean input text using the same preprocessing
    steps used during model training.
    """

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


def predict_job(job_description):
    """
    Predict whether a job posting is Genuine or Fraudulent.
    """

    cleaned = clean_text(job_description)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)

    probability = model.predict_proba(vector)[0]

    if prediction[0] == 1:
        result = "Fraudulent Job Posting"
    else:
        result = "Genuine Job Posting"

    return result, probability


def main():

    print("=" * 60)
    print(" Fake Job Posting Detection System ")
    print("=" * 60)

    print("\nEnter the complete Job Description.\n")

    print("Press ENTER twice when finished.\n")

    lines = []

    while True:

        line = input()

        if line == "":
            break

        lines.append(line)

    job_description = "\n".join(lines)

    if len(job_description.strip()) == 0:

        print("\nNo job description entered.")

        return

    prediction, probability = predict_job(job_description)

    print("\n" + "=" * 60)

    print("Prediction Result")

    print("=" * 60)

    print(prediction)

    print()

    print(f"Genuine Probability     : {probability[0]:.4f}")

    print(f"Fraudulent Probability : {probability[1]:.4f}")


if __name__ == "__main__":
    main()