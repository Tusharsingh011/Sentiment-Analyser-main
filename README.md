🎭 Sentiment Analyser
End-to-End NLP Pipeline for Movie Review Sentiment Classification
Python scikit-learn Streamlit License

Clean movie reviews → explore the data → train a TF‑IDF + Logistic Regression classifier → ship it in an interactive app.

📖 Overview
Sentiment Analyser is a complete, from-scratch NLP pipeline that classifies IMDB movie reviews as positive or negative. It covers the full workflow you'd expect in a real project:

🧹 Text cleaning (HTML tag removal, punctuation stripping, stopword filtering)
📊 Exploratory data analysis with class distribution plots and word clouds
⚡ Feature extraction using TF-IDF (unigrams + bigrams)
🏋️ Model training with Logistic Regression
🎯 Evaluation with accuracy, precision/recall, and a classification report
💾 Model persistence for downstream use (e.g. a Streamlit app)
The project ships both as a guided Jupyter notebook (sentiment_analysis.ipynb) for exploration, and a reusable script (model.py) for reproducible, one-command pipeline runs.

✨ Features
Feature	Description
🧼 Text Preprocessing	Strips HTML, lowercases, removes punctuation/numbers, filters NLTK stopwords
📈 EDA Visualizations	Sentiment class distribution + positive/negative word clouds
🔤 TF-IDF Vectorization	Up to 10,000 features, unigrams & bigrams (ngram_range=(1, 2))
🤖 Model	Logistic Regression classifier (max_iter=1000)
📦 Artifacts	Trained model & vectorizer pickled for reuse in apps
🖼️ Auto-generated Plots	Saved to Images/ for easy sharing in reports/READMEs
🗂️ Project Structure
sentiment-analyser/
├── dataset/
│   ├── IMDB Dataset.csv          # Raw dataset (download separately, see below)
│   └── cleaned_dataset.csv       # 1,000-row cleaned sample (auto-generated)
├── models/
│   ├── sentiment_model.pkl       # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
├── Images/
│   ├── class_distribution.png
│   ├── wordcloud_positive.png
│   └── wordcloud_negative.png
├── notebooks/
│   └── sentiment_analysis.ipynb  # Step-by-step exploratory notebook
├── model.py                      # End-to-end training pipeline script
└── README.md
🧠 How It Works

🚀 Getting Started
1. Clone the repository
git clone https://github.com/<Tusharsingh011>/<Sentiment-analyser-main>.git
cd <Sentiment-analyser-main>
2. Set up a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud nltk
Or, if you maintain a requirements.txt:

pip install -r requirements.txt
4. Download the dataset
This project uses the IMDB Dataset of 50K Movie Reviews from Kaggle.

Download it and place it here:

dataset/IMDB Dataset.csv
5. Run the pipeline
python model.py
This will:

Load and clean the dataset
Save EDA plots to Images/
Save a cleaned 1,000-row sample to dataset/cleaned_dataset.csv
Train and evaluate the Logistic Regression model
Save sentiment_model.pkl and tfidf_vectorizer.pkl to models/
Alternatively, open the notebook for a guided, cell-by-cell walkthrough:

jupyter notebook notebooks/sentiment_analysis.ipynb
🔮 Using the Trained Model
import pickle

# Load the saved artifacts
with open("models/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Predict on new text (remember to run it through the same clean_text() function first)
review = "This movie was an absolute masterpiece, I loved every second of it!"
vectorized = vectorizer.transform([review])
prediction = model.predict(vectorized)[0]

print("Positive 👍" if prediction == 1 else "Negative 👎")
📊 Example Outputs
The pipeline automatically generates the following visualizations in Images/:

Plot	Description
class_distribution.png	Bar chart of positive vs. negative review counts
wordcloud_positive.png	Most frequent words in positive reviews
wordcloud_negative.png	Most frequent words in negative reviews
(Embed screenshots here once available, e.g. ![Class Distribution](Images/class_distribution.png))

🛠️ Tech Stack
Language: Python 3.9+
Data Handling: pandas, numpy
Visualization: matplotlib, seaborn, wordcloud
NLP: NLTK (stopwords)
Machine Learning: scikit-learn (TF-IDF, Logistic Regression)
Interface: Streamlit (optional companion app)
🗺️ Roadmap
 Add a requirements.txt / environment.yml
 Try additional models (SVM, Naive Bayes, transformer-based embeddings)
 Add hyperparameter tuning (GridSearchCV)
 Deploy the Streamlit app publicly
 Add unit tests for clean_text() and the pipeline
🤝 Contributing
Contributions, issues, and feature requests are welcome!

Fork the project
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
🙏 Acknowledgements
IMDB Dataset of 50K Movie Reviews — Kaggle
scikit-learn documentation and community
NLTK for stopword corpora
📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

AUTHOR = ** Tushar Singh **
Made with 🍿 and a bit of Logistic Regression
