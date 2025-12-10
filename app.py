import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Fake News Detector", layout="wide")

# ==============================
# Custom CSS
# ==============================
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(to right, #f0f8ff, #e6f2ff);
        color: #000;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Headings */
    h1, h2, h3 {
        color: #003366;
    }
    /* Buttons */
    div.stButton > button {
        background-color: #0066cc;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1em;
        font-size: 1em;
    }
    </style>
    """, unsafe_allow_html=True
)

# ==============================
# Logo + Title
# ==============================
st.image("", width=150)  # ضع هنا مسار اللوجو الخاص بك
st.title("📰 Fake News Detector App")

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("🔧 Options")
uploaded = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
user_text = st.sidebar.text_area("Enter text to predict")

train_log = st.sidebar.button("Train Logistic Regression")
train_svc = st.sidebar.button("Train SVM (Linear)")
predict_btn = st.sidebar.button("Predict Text")

# ==============================
# Session State
# ==============================
for key in ["df","X_train","X_test","y_train","y_test","tfidf",
            "log_model","svc_model","X_train_vec","X_test_vec"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ==============================
# 1️⃣ Load Dataset
# ==============================
st.header("1️⃣ Load Dataset")
if uploaded is not None:
    df = pd.read_csv(uploaded, on_bad_lines="skip", engine="python")
    df.dropna(inplace=True)
    st.session_state.df = df
    st.success("✅ Dataset Loaded!")
    st.dataframe(df.head())
else:
    st.info("Upload a dataset to start.")

# ==============================
# 2️⃣ Preprocess & Clean Text
# ==============================
st.header("2️⃣ Preprocess Text")
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = df["title"] + " " + df["text"]
    df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    text = df["content"].str.lower()
    text = text.str.replace(r"http\S+", "", regex=True)
    text = text.str.replace(f"[{string.punctuation}]", "", regex=True)
    text = text.str.replace(r"\d+", "", regex=True)
    text = text.str.split()
    text = text.apply(lambda w: [x for x in w if x not in stop_words])
    text = text.apply(lambda w: [lemmatizer.lemmatize(x) for x in w])
    text = text.apply(lambda w: " ".join(w))
    df["clean"] = text
    st.session_state.df = df
    st.success("✅ Text Cleaning Completed!")
    st.dataframe(df[["clean","label"]].head())
else:
    st.info("Upload dataset first.")

# ==============================
# 3️⃣ Split Train/Test
# ==============================
st.header("3️⃣ Split Data")
if st.session_state.df is not None:
    df = st.session_state.df
    X = df["clean"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.success("✅ Train/Test Split Done!")
    st.write("Train size:", len(X_train))
    st.write("Test size:", len(X_test))

# ==============================
# 4️⃣ TF-IDF Vectorizer
# ==============================
st.header("4️⃣ TF-IDF Vectorizer")
if st.session_state.X_train is not None:
    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1,2))
    X_train_vec = tfidf.fit_transform(st.session_state.X_train)
    X_test_vec = tfidf.transform(st.session_state.X_test)

    st.session_state.tfidf = tfidf
    st.session_state.X_train_vec = X_train_vec
    st.session_state.X_test_vec = X_test_vec

    st.success("✅ TF-IDF Trained!")
    st.write("Train Matrix Shape:", X_train_vec.shape)
    st.write("Test Matrix Shape:", X_test_vec.shape)
else:
    st.info("Split data first.")

# ==============================
# 5️⃣ Train Models
# ==============================
st.header("5️⃣ Train Models")
col1, col2 = st.columns(2)

with col1:
    if train_log:
        if st.session_state.X_train_vec is not None:
            log_model = LogisticRegression()
            log_model.fit(st.session_state.X_train_vec, st.session_state.y_train)
            st.session_state.log_model = log_model
            st.success("✅ Logistic Regression Trained")
        else:
            st.info("Train TF-IDF first.")

with col2:
    if train_svc:
        if st.session_state.X_train_vec is not None:
            svc_model = SVC(kernel="linear")
            svc_model.fit(st.session_state.X_train_vec, st.session_state.y_train)
            st.session_state.svc_model = svc_model
            st.success("✅ SVM Trained")
        else:
            st.info("Train TF-IDF first.")

# ==============================
# 6️⃣ Evaluate Models
# ==============================
st.header("6️⃣ Evaluate Models")
if st.session_state.log_model and st.session_state.svc_model:
    X_test_vec = st.session_state.X_test_vec
    y_test = st.session_state.y_test

    pred_log = st.session_state.log_model.predict(X_test_vec)
    pred_svc = st.session_state.svc_model.predict(X_test_vec)

    metrics_df = pd.DataFrame({
        "Model": ["Logistic", "SVM"],
        "Accuracy": [accuracy_score(y_test, pred_log), accuracy_score(y_test, pred_svc)],
        "Recall": [recall_score(y_test, pred_log), recall_score(y_test, pred_svc)],
        "F1 Score": [f1_score(y_test, pred_log), f1_score(y_test, pred_svc)]
    })
    st.subheader("📊 Metrics")
    st.dataframe(metrics_df)

    st.subheader("Confusion Matrix (SVM)")
    cm = confusion_matrix(y_test, pred_svc)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
else:
    st.info("Train both models to evaluate.")

# ==============================
# 7️⃣ Predict User Text
# ==============================
st.header("7️⃣ Predict Your Text")
if predict_btn:
    if st.session_state.tfidf and st.session_state.svc_model:
        clean = user_text.lower()
        clean = re.sub(r"http\S+", "", clean)
        clean = re.sub(f"[{string.punctuation}]", "", clean)
        clean = re.sub(r"\d+", "", clean)
        vect = st.session_state.tfidf.transform([clean])
        pred = st.session_state.svc_model.predict(vect)[0]
        label = "REAL" if pred==1 else "FAKE"
        st.success(f"Prediction: **{label}**")
    else:
        st.info("Train models first.")
