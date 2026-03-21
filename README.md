# 📰 AI Fake News Detection

An AI-powered web application that detects whether a given news headline or short news text is **Real** or **Fake** using **Natural Language Processing (NLP)** and **Machine Learning (ML)**.

🚀 **Live Demo:** https://ai-fake-news-detection-4.onrender.com

---

##  Project Overview

This project is a beginner-friendly **AI/ML + NLP web application** built using **Flask**.  
It classifies user-entered news text as:

- ✅ **REAL**
- ❌ **FAKE**

The model is trained on a labeled dataset (`news.csv`) using:

- **TF-IDF Vectorization** (for text feature extraction)
- **Logistic Regression** (for classification)

---

## ✨ Features

- Enter any news headline or short news text
- Detect whether the news is **Real** or **Fake**
- Shows **prediction confidence percentage**
- Clean and simple web interface
- Fully deployed online using **Render**
- Beginner-friendly AI + NLP project for resume/portfolio

---

##  Tech Stack

### **Frontend**
- HTML
- CSS

### **Backend**
- Python
- Flask

### **AI / ML / NLP**
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Regular Expressions (Text Cleaning)

### **Deployment**
- Render

---

##  How It Works

### 1. Dataset Loading
The app reads the dataset from `news.csv`, which contains:
- News text
- Labels (`REAL` or `FAKE`)

### 2. Text Preprocessing
The input text is cleaned by:
- converting to lowercase
- removing punctuation
- removing numbers
- removing extra spaces

### 3. NLP Feature Extraction
The cleaned text is converted into numerical form using:

- **TF-IDF (Term Frequency - Inverse Document Frequency)**

This helps the model understand the importance of words in the news text.

### 4. Machine Learning Prediction
A **Logistic Regression** model is trained on the dataset and predicts whether the news is:

- **REAL**
- **FAKE**

### 5. Confidence Score
The app also displays the model's confidence score as a percentage.

---

## 📂 Project Structure

```bash
AI-Fake-News-Detection/
│── app.py
│── news.csv
│── requirements.txt
│── Procfile
│── README.md
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
