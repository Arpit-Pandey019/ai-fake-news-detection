from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

app = Flask(__name__)

# -----------------------------
# Text cleaning function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("fake_news_dataset.csv")  # Make sure this CSV has 'text' and 'label' columns
df['text'] = df['text'].fillna("").apply(clean_text)
df['label'] = df['label'].fillna("FAKE")

# Shuffle the dataset to avoid bias
df = shuffle(df, random_state=42)

# Convert labels to numeric: REAL=1, FAKE=0
df['label_num'] = df['label'].apply(lambda x: 1 if str(x).upper() == "REAL" else 0)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

# -----------------------------
# TF-IDF Vectorization + Logistic Regression
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

# -----------------------------
# Flask route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        user_input = request.form["news"]
        cleaned = clean_text(user_input)
        vect_text = vectorizer.transform([cleaned])

        pred = model.predict(vect_text)[0]
        proba = model.predict_proba(vect_text)[0]

        if pred == 1:
            prediction = "REAL NEWS"
            confidence = round(proba[1] * 100, 2)
        else:
            prediction = "FAKE NEWS"
            confidence = round(proba[0] * 100, 2)

    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
