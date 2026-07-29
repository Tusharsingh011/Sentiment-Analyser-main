import os
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Configuration
DATASET_PATH = "dataset/IMDB Dataset.csv"
CLEANED_DATASET_PATH = "dataset/cleaned_dataset.csv"
MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
IMAGES_DIR = "Images"

# Download and load NLTK English stopwords
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Preprocesses reviews: removes HTML tags, punctuation, and stopwords."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters/punctuation (keep alphabetic words and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Split text into words and remove stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in STOPWORDS]
    # Rejoin words
    return " ".join(cleaned_words)

def build_and_save_pipeline():
    # 1. Load dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_PATH}'. Please download the IMDB Dataset of 50K Movie Reviews "
            "from Kaggle and place it in the dataset folder as 'IMDB Dataset.csv'."
        )
        
    print(f"[INFO] Loading dataset from '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)
    print(f"[INFO] Dataset loaded with shape: {df.shape}")
    
    # Handle possible empty or missing rows
    df = df.dropna(subset=["review", "sentiment"])
    
    # 2. Text Preprocessing
    print("[INFO] Cleaning reviews (removing HTML tags, punctuation, and stopwords)...")
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Save a 1,000-row sample of the cleaned dataset for the Streamlit app explorer (avoids pushing 100MB+ file to git)
    os.makedirs(os.path.dirname(CLEANED_DATASET_PATH), exist_ok=True)
    df.head(1000).to_csv(CLEANED_DATASET_PATH, index=False)
    print(f"[INFO] Cleaned dataset sample (1,000 rows) saved to '{CLEANED_DATASET_PATH}'")
    
    # 3. Generating EDA Plots
    print("[INFO] Generating EDA visualizations...")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Plot 1: Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sentiment', data=df, palette='Set2')
    plt.title('Sentiment Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    dist_path = os.path.join(IMAGES_DIR, 'class_distribution.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved distribution plot to '{dist_path}'")
    
    # Plot 2: Word Clouds for Positive Reviews
    print("[INFO] Generating word clouds...")
    pos_text = " ".join(df[df['sentiment'] == 'positive']['cleaned_review'])
    if pos_text.strip():
        pos_wordcloud = WordCloud(width=800, height=400, background_color='black', 
                                  colormap='viridis', max_words=100).generate(pos_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(pos_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Positive Reviews', fontsize=16, fontweight='bold')
        plt.tight_layout(pad=0)
        pos_wc_path = os.path.join(IMAGES_DIR, 'wordcloud_positive.png')
        plt.savefig(pos_wc_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved positive wordcloud to '{pos_wc_path}'")
        
    # Plot 3: Word Clouds for Negative Reviews
    neg_text = " ".join(df[df['sentiment'] == 'negative']['cleaned_review'])
    if neg_text.strip():
        neg_wordcloud = WordCloud(width=800, height=400, background_color='black', 
                                  colormap='magma', max_words=100).generate(neg_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(neg_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Negative Reviews', fontsize=16, fontweight='bold')
        plt.tight_layout(pad=0)
        neg_wc_path = os.path.join(IMAGES_DIR, 'wordcloud_negative.png')
        plt.savefig(neg_wc_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved negative wordcloud to '{neg_wc_path}'")

    # 4. Train-Test Split
    X = df['cleaned_review']
    y = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. TF-IDF Vectorization
    print("[INFO] Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # 6. Model Training
    print("[INFO] Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vectorized, y_train)
    
    # 7. Model Evaluation
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[INFO] Model Accuracy: {accuracy:.4f}")
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
    
    # 8. Save Model and Vectorizer
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"[SUCCESS] Saved trained model to '{MODEL_PATH}'")
    print(f"[SUCCESS] Saved TF-IDF vectorizer to '{VECTORIZER_PATH}'")
    print("[SUCCESS] Pipeline run successfully complete!")

if __name__ == "__main__":
    build_and_save_pipeline()
