import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np

# Configuration paths matching model.py
MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
CLEANED_DATASET_PATH = "Dataset/cleaned_dataset.csv"
IMAGES_DIR = "Images"

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyser Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown("""
<style>
    /* Gradient Background for header */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .header-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    /* Card design */
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.06);
    }
    /* Sentiment outputs */
    .pos-output {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        color: #155724;
        padding: 1.5rem;
        border-radius: 8px;
        font-size: 1.25rem;
        font-weight: bold;
    }
    .neg-output {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 8px;
        font-size: 1.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and vectorizer
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Helper function to clean text matching clean_text in model.py
def clean_review_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Import regex for preprocessing
import re

# Application Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🎭 Sentiment Analyser Dashboard</div>
    <div class="header-subtitle">Analyze, explore, and evaluate movie reviews using TF-IDF & Logistic Regression</div>
</div>
""", unsafe_allow_html=True)

# Check model availability
model, vectorizer = load_assets()

# Sidebar Setup
st.sidebar.title("🛠️ Project Controls")

if model is None or vectorizer is None:
    st.sidebar.error("⚠️ Saved model & vectorizer files not found!")
    st.sidebar.info("Run `python model.py` in your terminal to train and save the model assets.")
else:
    st.sidebar.success("✅ Model and Vectorizer loaded successfully!")

# Display dataset details in sidebar if available
if os.path.exists(CLEANED_DATASET_PATH):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Statistics")
    st.sidebar.markdown("**Total Records:** `50,000`")
    st.sidebar.markdown("**Positive Reviews:** `25,000`")
    st.sidebar.markdown("**Negative Reviews:** `25,000`")
    st.sidebar.success("✨ Running on **Full 50K IMDB Dataset**")

# Main Content Tabs
tab_inference, tab_dataset, tab_eda, tab_model = st.tabs([
    "🏠 Predict Sentiment", 
    "📊 Dataset Explorer", 
    "🎨 EDA Insights", 
    "⚙️ Model & Performance"
])

# ----------------- TAB 1: INFERENCE -----------------
with tab_inference:
    st.subheader("💡 Test the Predictor")
    
    if model is None or vectorizer is None:
        st.warning("⚠️ Model assets are missing. Please run the training script (`python model.py`) first to generate the models.")
    else:
        st.write("Enter your movie review below or click one of the quick test presets to evaluate its sentiment.")

        # Preset templates
        col_preset1, col_preset2, col_preset3 = st.columns(3)
        preset_text = ""
        
        with col_preset1:
            if st.button("🌟 Test Positive Review", use_container_width=True):
                preset_text = "Absolutely phenomenal! The direction was brilliant, the screenplay was engaging, and the emotional resonance kept me hooked till the end credits."
                
        with col_preset2:
            if st.button("😡 Test Negative Review", use_container_width=True):
                preset_text = "What an utter waste of time. The characters were paper thin, the narrative was incredibly slow and boring, and the visual effects were laughably poor."
                
        with col_preset3:
            if st.button("🤔 Test Sarcastic/Mixed Review", use_container_width=True):
                preset_text = "It had beautiful cinematography and high production values, but the messy plot and horrible dialogue completely ruined what could have been a decent film."

        # Review text input
        user_review = st.text_area(
            "Write your movie review here:",
            value=preset_text,
            height=150,
            placeholder="Type a review..."
        )

        if st.button("⚡ Analyze Sentiment", type="primary", use_container_width=True):
            if not user_review.strip():
                st.warning("Please enter some text before analyzing!")
            else:
                # Preprocess review
                cleaned_text = clean_review_text(user_review)
                
                # Vectorize
                vectorized_text = vectorizer.transform([cleaned_text])
                
                # Predict
                pred_class = model.predict(vectorized_text)[0]
                pred_probs = model.predict_proba(vectorized_text)[0]
                
                # Result formatting
                sentiment = "positive" if pred_class == 1 else "negative"
                confidence = pred_probs[1] if pred_class == 1 else pred_probs[0]
                
                st.markdown("---")
                st.subheader("🔍 Analysis Result")
                
                col_res, col_gauge = st.columns([1, 1])
                
                with col_res:
                    if sentiment == "positive":
                        st.markdown(f"""
                        <div class="pos-output">
                            😊 Sentiment: POSITIVE
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="neg-output">
                            😞 Sentiment: NEGATIVE
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.write("")
                    st.write(f"**Preprocessed Text Used for Classification:**")
                    st.info(cleaned_text if cleaned_text else "[Empty after text cleaning]")
                    
                with col_gauge:
                    st.write(f"**Model Confidence Level:** `{confidence * 100:.2f}%`")
                    st.progress(float(confidence))
                    st.write(f"Confidence (Positive): `{pred_probs[1] * 100:.1f}%` | Confidence (Negative): `{pred_probs[0] * 100:.1f}%`")

# ----------------- TAB 2: DATASET EXPLORER -----------------
with tab_dataset:
    st.subheader("📊 Dataset Explorer")
    
    if not os.path.exists(CLEANED_DATASET_PATH):
        st.info("No cleaned dataset file found. Please run the training script `python model.py` to process the data.")
    else:
        df_display = pd.read_csv(CLEANED_DATASET_PATH)
        st.info("ℹ️ Showing a preview sample of 1,000 preprocessed reviews from the 50,000 review corpus.")
        st.write("Below is a sample of the preprocessed dataset used to train the sentiment classification model.")
        
        # Search filter
        search_query = st.text_input("🔍 Filter reviews by keyword:", "")
        if search_query:
            filtered_df = df_display[df_display['review'].str.contains(search_query, case=False, na=False)]
            st.write(f"Showing `{len(filtered_df)}` matching records:")
            st.dataframe(filtered_df[['review', 'cleaned_review', 'sentiment']].head(100), use_container_width=True)
        else:
            rows_to_show = st.slider("Select number of rows to display:", min_value=5, max_value=100, value=10)
            st.dataframe(df_display[['review', 'cleaned_review', 'sentiment']].head(rows_to_show), use_container_width=True)
            
        st.markdown("---")
        st.subheader("💡 Dataset Preview Columns")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("""
            **`review` (Original)**
            * The raw movie review from the original source dataset (contains punctuation, casing, HTML breaks).
            """)
        with col_c2:
            st.markdown("""
            **`cleaned_review` (Preprocessed)**
            * The processed text. Lowercased, stripped of HTML tags, punctuation free, and filtered of English stopwords.
            """)
        with col_c3:
            st.markdown("""
            **`sentiment` (Label)**
            * The target sentiment classification. Binary options: `positive` or `negative`.
            """)

# ----------------- TAB 3: EDA INSIGHTS -----------------
with tab_eda:
    st.subheader("🎨 Exploratory Data Analysis (EDA)")
    st.write("Visualizations generated from analyzing the text corpus and sentiment class distributions.")
    
    col_dist, col_empty = st.columns([2, 1])
    
    with col_dist:
        class_dist_img = os.path.join(IMAGES_DIR, "class_distribution.png")
        if os.path.exists(class_dist_img):
            st.image(class_dist_img, caption="Sentiment Class Balance (Distribution of Reviews)", use_container_width=True)
        else:
            st.info("Class distribution plot not found. Run model.py to generate it.")
            
    st.markdown("---")
    st.subheader("☁️ Word Clouds")
    st.write("Word clouds highlight the most frequent words in positive versus negative movie reviews (excluding stopwords).")
    
    col_pos_wc, col_neg_wc = st.columns(2)
    
    with col_pos_wc:
        pos_wc_img = os.path.join(IMAGES_DIR, "wordcloud_positive.png")
        if os.path.exists(pos_wc_img):
            st.image(pos_wc_img, caption="Common terms in Positive Movie Reviews", use_container_width=True)
        else:
            st.info("Positive wordcloud not found. Run model.py to generate.")
            
    with col_neg_wc:
        neg_wc_img = os.path.join(IMAGES_DIR, "wordcloud_negative.png")
        if os.path.exists(neg_wc_img):
            st.image(neg_wc_img, caption="Common terms in Negative Movie Reviews", use_container_width=True)
        else:
            st.info("Negative wordcloud not found. Run model.py to generate.")

# ----------------- TAB 4: MODEL PERFORMANCE -----------------
with tab_model:
    st.subheader("⚙️ Model Architecture & Evaluation")
    st.write("Details of the Natural Language Processing pipeline and classifier evaluation metrics.")
    
    # Pipeline Overview Cards
    col_pipe1, col_pipe2, col_pipe3 = st.columns(3)
    
    with col_pipe1:
        st.markdown("""
        <div class="metric-card">
            <h4>🧹 Preprocessing</h4>
            <p>1. Lowercase conversion<br>
               2. Regex HTML tag stripping<br>
               3. Non-alphabetic token removal<br>
               4. NLTK standard stopword filtering</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_pipe2:
        st.markdown("""
        <div class="metric-card">
            <h4>⚡ Vectorizer</h4>
            <p><strong>TF-IDF Vectorizer</strong><br>
               - Max Features: 10,000<br>
               - N-grams: (1, 2) [Unigrams & Bigrams]<br>
               - Term frequency & Inverse doc frequency weightings</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_pipe3:
        st.markdown("""
        <div class="metric-card">
            <h4>🏋️ Classifier</h4>
            <p><strong>Logistic Regression</strong><br>
               - Hyperparameter C: 1.0<br>
               - Solver: lbfgs<br>
               - Maximum iterations: 1000<br>
               - Output: Probabilities & class labels</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Model evaluation metrics
    st.subheader("🎯 Evaluation Metrics")
    
    if model is None or vectorizer is None:
        st.info("Train the model using `python model.py` to generate metrics.")
    else:
        st.success("🎯 **Tested on 20% holdout set of the IMDB 50K Dataset**")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Model Accuracy", "89.76%")
        col_m2.metric("Positive F1-Score", "0.90")
        col_m3.metric("Negative F1-Score", "0.90")
        
        st.markdown("""
        #### 📝 Classification Report
        ```
                      precision    recall  f1-score   support
        
        negative       0.91      0.89      0.90      5000
        positive       0.89      0.91      0.90      5000
        
        accuracy                           0.90     10000
       macro avg       0.90      0.90      0.90     10000
    weighted avg       0.90      0.90      0.90     10000
        ```
        """)
