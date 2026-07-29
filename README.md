# 🎭 Sentiment Analyser

An end-to-end **NLP-based Sentiment Classification System** that predicts whether a movie review is **Positive 😊 or Negative 😞** using Machine Learning.

This project implements a complete Natural Language Processing pipeline including text preprocessing, exploratory data analysis, feature extraction, model training, and deployment using Streamlit.

---

# 🚀 Features

✅ Sentiment prediction from user review text  
✅ Complete NLP preprocessing pipeline  
✅ TF-IDF feature extraction  
✅ Logistic Regression classification model  
✅ Interactive Streamlit web application  
✅ Exploratory Data Analysis (EDA)  
✅ WordCloud visualization  
✅ Saved ML model and vectorizer  

---

# 🧠 Machine Learning Workflow

```
Raw Movie Review
        |
        ↓
Text Cleaning & Preprocessing
        |
        ↓
Tokenization
        |
        ↓
TF-IDF Vectorization
        |
        ↓
Logistic Regression Model
        |
        ↓
Sentiment Prediction
```

---

# 📂 Project Structure

```
Sentiment-Analyser-main
│
├── Dataset
│   └── cleaned_dataset.csv
│
├── Images
│   ├── class_distribution.png
│   ├── wordcloud_positive.png
│   └── wordcloud_negative.png
│
├── models
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebook
│   └── sentiment_analysis.ipynb
│
├── app.py
├── model.py
├── requirements.txt
└── README.md
```

---

# 📊 Exploratory Data Analysis

## 📌 Sentiment Distribution

<img src="./Images/class_distribution.png" width="700">


## ☁️ Positive Reviews Word Cloud

<img src="./Images/wordcloud_positive.png" width="700">


## ☁️ Negative Reviews Word Cloud

<img src="./Images/wordcloud_negative.png" width="700">


---

# 🛠️ Tech Stack

## Programming Language
- Python

## Machine Learning
- Scikit-Learn
- Logistic Regression
- TF-IDF Vectorizer

## Data Processing
- Pandas
- NumPy

## Visualization
- Matplotlib
- WordCloud

## Deployment
- Streamlit

---

# ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/Tusharsingh011/Sentiment-Analyser-main.git
```

### 2. Navigate to Project Folder

```bash
cd Sentiment-Analyser-main
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit Application

```bash
streamlit run app.py
```

---

# 💻 Application Preview

### Enter a movie review:

Example:

```
The movie was amazing and the acting was brilliant.
```

### Prediction:

```
Positive 😊
```

---

Example:

```
The movie was boring and disappointing.
```

### Prediction:

```
Negative 😞
```

---

# 🔮 Future Improvements

🚀 Implement advanced Transformer models like BERT  
🚀 Real-time social media sentiment monitoring  
🚀 Multi-class emotion detection  
🚀 Cloud deployment using AWS / Azure / Streamlit Cloud  
🚀 Improve accuracy using deep learning approaches  

---

# 👨‍💻 Author

**Tushar Singh**

Computer Science Engineering

AI | Machine Learning | Data Analytics

---

⭐ If you found this project useful, consider giving it a star!
