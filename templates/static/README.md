# AI Fake News Detection

AI Fake News Detection is a machine learning web application that predicts whether a news article or headline is FAKE or REAL using Natural Language Processing (NLP) and Logistic Regression.

## Features
- Detects fake or real news from text input
- Uses TF-IDF for text vectorization
- Uses Logistic Regression for classification
- Flask-based web interface
- Deployable on Render for free

## Tech Stack
- Python
- Flask
- Pandas
- Scikit-learn
- HTML
- CSS
- Render

## Project Structure
- train_model.py -> trains the model
- app.py -> Flask backend
- templates/index.html -> frontend HTML
- static/style.css -> frontend CSS
- model.pkl -> saved ML model
- vectorizer.pkl -> saved TF-IDF vectorizer

## How to Run
1. Install dependencies
2. Run train_model.py
3. Run app.py
4. Open browser at http://127.0.0.1:5000