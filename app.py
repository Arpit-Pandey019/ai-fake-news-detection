from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)


# -------------------------------
# Load Model and Vectorizer
# -------------------------------
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# -------------------------------
# Home Route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("news_text", "")

        if user_text.strip():
            cleaned_text = clean_text(user_text)
            vectorized_text = vectorizer.transform([cleaned_text])
            result = model.predict(vectorized_text)[0]

            if result == "FAKE":
                prediction = "FAKE NEWS"
            else:
                prediction = "REAL NEWS"
        else:
            prediction = "Please enter some news text."

    return render_template("index.html", prediction=prediction, user_text=user_text)


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)