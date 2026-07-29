🎭 Sentiment Analyser
An end-to-end sentiment classification application that predicts the positive or negative sentiment of movie reviews. The project features a full machine learning pipeline (text preprocessing, EDA generation, TF-IDF vectorization, and Logistic Regression classification) and exposes it via an interactive, modern Streamlit dashboard.

📂 Project Structure
Sentiment Analyser/
│
├── dataset/
│   ├── IMDB Dataset.csv         <- [User Downloaded] Original 50k IMDB dataset
│   └── cleaned_dataset.csv      <- Preprocessed dataset (generated during training)
│
├── Images/
│   ├── class_distribution.png   <- EDA: Bar plot of sentiments
│   ├── wordcloud_positive.png   <- EDA: Positive review word cloud
│   └── wordcloud_negative.png   <- EDA: Negative review word cloud
│
├── models/
│   ├── sentiment_model.pkl      <- Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl     <- Fitted TF-IDF Vectorizer
│
├── notebook/
│   └── sentiment_analysis.ipynb <- Step-by-step pipeline notebook
│
├── app.py                       <- Streamlit application code
├── model.py                     <- Standalone training and pipeline script
├── README.md                    <- Project documentation
└── requirements.txt             <- Project dependencies
🚀 Getting Started
1. Clone the repository and navigate to it:
git clone <repository-url>
cd "Sentiment Analyser"
2. Download the Dataset
Download the IMDB Dataset of 50K Movie Reviews from Kaggle: 👉 Kaggle Dataset Link

Place the downloaded IMDB Dataset.csv inside the dataset/ folder: dataset/IMDB Dataset.csv

Note: If you launch the app or training script without the dataset, a small mock dataset of 100 sample reviews will be automatically generated so you can try out the application immediately!

3. Install Dependencies
Make sure you have Python 3.9+ installed, then install the required libraries:

pip install -r requirements.txt
4. Train the Model
Run the training script to preprocess the data, train the classifier, generate EDA plots, and save the model assets:

python model.py
5. Launch the Web Dashboard
Start the Streamlit application:

streamlit run app.py
Open your browser and navigate to the local address provided (typically http://localhost:8501).

🛠️ Machine Learning Pipeline Details
Preprocessing:

Strips HTML tags (e.g. <br />).
Converts text to lowercase.
Removes special characters, numbers, and punctuation.
Filters out common English stopwords.
Feature Extraction:

TF-IDF (Term Frequency - Inverse Document Frequency) with Unigram and Bigram features (ngram_range=(1, 2)) capped at 10,000 features.
Classification Model:

Logistic Regression model (C=1.0), achieving ~89.4% accuracy on the 50K IMDB Movie Reviews test set.
