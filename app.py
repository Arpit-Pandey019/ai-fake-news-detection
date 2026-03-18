from flask import Flask, render_template, request
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

# Load dataset
df = pd.read_csv("fake_news_dataset.csv")

# Clean dataset text
df['text'] = df['text'].fillna("").apply(clean_text)

# Features and labels
X = df['text']
y = df['label']

# Convert labels if needed (FAKE = 0, REAL = 1)
if y.dtype == object:
    y = y.str.upper().map({'FAKE': 0, 'REAL': 1})

# Better vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
X_vectorized = vectorizer.fit_transform(X)

# Better model
model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    confidence = ""

    if request.method == "POST":
        news = request.form["news"]

        # Clean user input before prediction
        cleaned_news = clean_text(news)
        input_data = vectorizer.transform([cleaned_news])

        pred = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]

        if pred == 1:
            prediction = "REAL NEWS"
            confidence = f"Confidence: {probs[1] * 100:.2f}%"
        else:
            prediction = "FAKE NEWS"
            confidence = f"Confidence: {probs[0] * 100:.2f}%"

    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)