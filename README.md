# 📰 Fake News Detector

## 🚀 Project Overview
The **Fake News Detector** is a web-based application built using **Streamlit**. It allows users to enter a news headline and determine whether the news is **real or fake** using a trained **Machine Learning model**.

The model uses **TF-IDF vectorization** and a **classification algorithm** to predict the authenticity of the news, displaying confidence levels for both **fake and real** classifications.

---

## 🏗️ **How It Works**
1. Users enter a news title in the input field.
2. The text is **vectorized** using a trained **TF-IDF vectorizer**.
3. The vectorized text is fed into a **pre-trained classifier model**.
4. The model **predicts** whether the news is fake or real and provides a **confidence score**.
5. The results are displayed with a **progress bar and a colored indicator**.

---

## 📸 **Screenshots**

### Example 1: **Fake News Prediction**
![Fake News Prediction](Fake.png)

- **News Title Entered**: *"Gwen Stefani got dumped by Blake Shelton over 'jealousy and drama' (exclusive)"*
- **Prediction**: ❌ *This news is FAKE!*  
- **Confidence**: **96.90% Fake, 3.10% Real**

---

### Example 2: **Real News Prediction**
![Real News Prediction](Real.png)

- **News Title Entered**: *"Avatar is a good movie"*
- **Prediction**: ✅ *This news is REAL!*  
- **Confidence**: **87.34% Real, 12.66% Fake**

---


### Install Dependencies
pip install -r requirements.txt

### Run Application
Streamlit run app.py
