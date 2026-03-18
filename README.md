# 📰 AI Fake News Detection using Machine Learning and NLP

A web-based **AI/ML project** that predicts whether a news headline or article is **Fake** or **Real** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

This project is built using **Python, Flask, Pandas, Scikit-learn, and TF-IDF Vectorization**, and is deployed live on **Render**.

---

## 🚀 Live Demo

🔗 **Live Project:** https://ai-fake-news-detection-5.onrender.com

🔗 **GitHub Repository:** https://github.com/Arpit-Pandey019/ai-fake-news-detection

---

## 📌 Project Overview

Fake news spreads rapidly on digital platforms and can mislead people with false or manipulated information.  
This project aims to build a simple **Fake News Detection System** that can classify news text as **Fake** or **Real** using **Machine Learning and NLP**.

The system takes a user-input news headline or article, preprocesses the text, converts it into numerical features using **TF-IDF Vectorization**, and then predicts the result using a trained **Machine Learning classification model**.

---

## 🎯 Features

- Detects whether a news headline/article is **Fake** or **Real**
- Built using **Machine Learning + NLP**
- Text preprocessing and cleaning
- TF-IDF feature extraction
- Flask-based web interface
- User-friendly UI with HTML + CSS
- Live deployment on Render

---

## 🧠 AI/ML & NLP Concepts Used

This project demonstrates the practical use of **Artificial Intelligence, Machine Learning, and Natural Language Processing**.

### ✅ Natural Language Processing (NLP)
NLP is used to process and clean raw text before giving it to the ML model.

**NLP tasks used in this project:**
- Convert text to lowercase
- Remove special characters and punctuation
- Remove unwanted spaces
- Clean user input for better model performance

### ✅ Machine Learning (ML)
After preprocessing, the text is converted into numerical form using **TF-IDF Vectorization**.

Then a **Machine Learning classification model** is trained on labeled news data:
- **REAL**
- **FAKE**

The model learns patterns from the dataset and predicts the class of new input text.

### ✅ TF-IDF Vectorization
TF-IDF converts text into numerical vectors so that the ML model can understand and classify text.

---

## 🛠️ Tech Stack

- **Python**
- **Flask**
- **Pandas**
- **Scikit-learn**
- **HTML**
- **CSS**
- **Render** (Deployment)

---

## 📂 Project Structure

```bash
AI Fake News Detection/
│── app.py
│── fake_news_dataset.csv
│── requirements.txt
│── templates/
│   └── index.html
│── static/
│   └── style.css
│── README.md
