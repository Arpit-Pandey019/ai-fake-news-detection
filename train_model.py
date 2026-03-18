import pandas as pd
import pickle
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("fake_news_dataset.csv")

# Keep only required columns
df = df[['text', 'label']]

# Remove missing values
df.dropna(inplace=True)


# -------------------------------
# 2. Text Cleaning Function
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


# Apply cleaning
df['text'] = df['text'].apply(clean_text)


# -------------------------------
# 3. Features and Labels
# -------------------------------
X = df['text']
y = df['label']


# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# 5. TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# -------------------------------
# 6. Train Model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)


# -------------------------------
# 7. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -------------------------------
# 8. Save Model and Vectorizer
# -------------------------------
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel and vectorizer saved successfully!")