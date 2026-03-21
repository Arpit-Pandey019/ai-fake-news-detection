import pandas as pd
import re
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Load dataset
df = pd.read_csv("news.csv")

# Fill missing values
df['title'] = df['title'].fillna("")
df['text'] = df['text'].fillna("")

# Combine title + text
df['text'] = (df['title'] + " " + df['text']).apply(clean_text)

# Clean labels
df['label'] = df['label'].astype(str).str.strip().str.upper()

# Convert labels safely
df['label'] = df['label'].replace({
    'FAKE': 0,
    'REAL': 1,
    '0': 0,
    '1': 1
})

# Keep only valid labels
df = df[df['label'].isin([0, 1])]

# Convert to int
df['label'] = df['label'].astype(int)

# Check if both classes exist
if df['label'].nunique() < 2:
    raise ValueError("Dataset must contain both FAKE and REAL labels in news.csv")

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vect = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        news = request.form["news"]
        cleaned_news = clean_text(news)
        news_vect = vectorizer.transform([cleaned_news])

        pred = model.predict(news_vect)[0]
        proba = model.predict_proba(news_vect)[0]

        prediction = "REAL NEWS" if pred == 1 else "FAKE NEWS"
        confidence = round(max(proba) * 100, 2)

    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
