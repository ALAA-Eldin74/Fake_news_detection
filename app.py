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
# Page Config + Custom CSS
# ==============================
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.markdown(
    """
    <style>
    /* Background transparent */
    .stApp {
        background: rgba(255, 255, 255, 0.05);
        color: #000;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Headings style */
    h1, h2, h3 {
        color: #1f77b4;
    }
    /* Buttons style */
    div.stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1em;
        font-size: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📰 Fake News Detector App")


# ==============================
# GLOBAL SESSION STATE
# ==============================
for key in ["df","X_train","X_test","y_train","y_test","tfidf",
            "log_model","svc_model","X_train_vec","X_test_vec"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ==============================
# 1) Load Dataset
# ==============================
st.header("1️⃣ Load Dataset")
uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded, on_bad_lines="skip", engine="python")
    df.dropna(inplace=True)
    st.session_state.df = df
    st.success("✅ Dataset Loaded Successfully!")
    st.dataframe(df.head())


# ==============================
# 2) Preprocessing
# ==============================
st.header("2️⃣ Preprocess & Clean Text")
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
    st.success("✅ Cleaning Completed Successfully!")
    st.dataframe(df[["clean", "label"]].head())
else:
    st.info("Please upload a dataset first.")


# ==============================
# 3) Split Data
# ==============================
st.header("3️⃣ Split Train/Test")
if st.session_state.df is not None:
    df = st.session_state.df
    X = df["clean"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.success("✅ Data split completed!")
    st.write("Train size:", len(X_train))
    st.write("Test size:", len(X_test))
else:
    st.info("Clean the data first.")


# ==============================
# 4) TF-IDF Vectorizer
# ==============================
st.header("4️⃣ Train TF-IDF Vectorizer")
if st.session_state.X_train is not None:
    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1,2))
    X_train_vec = tfidf.fit_transform(st.session_state.X_train)
    X_test_vec = tfidf.transform(st.session_state.X_test)

    st.session_state.tfidf = tfidf
    st.session_state.X_train_vec = X_train_vec
    st.session_state.X_test_vec = X_test_vec

    st.success("✅ TF-IDF trained successfully!")
    st.write("Train Matrix Shape:", X_train_vec.shape)
    st.write("Test Matrix Shape:", X_test_vec.shape)
    st.subheader("Top 50 Words/Terms")
    st.write(tfidf.get_feature_names_out())
else:
    st.info("Split the data first.")


# ==============================
# 5) Train Models
# ==============================
st.header("5️⃣ Train ML Models")
col1, col2 = st.columns(2)

# Logistic Regression
with col1:
    if st.button("Train Logistic Regression"):
        if st.session_state.X_train_vec is not None:
            log_model = LogisticRegression()
            log_model.fit(st.session_state.X_train_vec, st.session_state.y_train)
            st.session_state.log_model = log_model
            st.success("✅ Logistic Regression Trained ✔")
        else:
            st.info("Train TF-IDF first.")

# SVM
with col2:
    if st.button("Train SVM (Linear)"):
        if st.session_state.X_train_vec is not None:
            svc = SVC(kernel="linear")
            svc.fit(st.session_state.X_train_vec, st.session_state.y_train)
            st.session_state.svc_model = svc
            st.success("✅ SVM Trained ✔")
        else:
            st.info("Train TF-IDF first.")


# ==============================
# 6) Evaluate Models
# ==============================
st.header("6️⃣ Evaluate Models")
if st.session_state.log_model is not None and st.session_state.svc_model is not None:
    X_test = st.session_state.X_test_vec
    y_test = st.session_state.y_test

    # Logistic
    pred_log = st.session_state.log_model.predict(X_test)
    acc_log = accuracy_score(y_test, pred_log)
    rec_log = recall_score(y_test, pred_log)
    f1_log = f1_score(y_test, pred_log)

    # SVC
    pred_svc = st.session_state.svc_model.predict(X_test)
    acc_svc = accuracy_score(y_test, pred_svc)
    rec_svc = recall_score(y_test, pred_svc)
    f1_svc = f1_score(y_test, pred_svc)

    st.subheader("📊 Results")
    st.write(
        pd.DataFrame({
            "Model": ["Logistic", "SVM"],
            "Accuracy": [acc_log, acc_svc],
            "Recall": [rec_log, rec_svc],
            "F1 Score": [f1_log, f1_svc]
        })
    )

    # Confusion Matrix
    st.subheader("Confusion Matrix (SVM)")
    cm = confusion_matrix(y_test, pred_svc)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
else:
    st.info("Train both models first.")


# ==============================
# 7) Predict User Text
# ==============================
st.header("7️⃣ Try Your Own Text")
user_text = st.text_area("Enter news text to check if it’s FAKE or REAL")

if st.button("Predict Text"):
    if st.session_state.tfidf is not None and st.session_state.svc_model is not None:
        clean = user_text.lower()
        clean = re.sub(r"http\S+", "", clean)
        clean = re.sub(f"[{string.punctuation}]", "", clean)
        clean = re.sub(r"\d+", "", clean)

        vect = st.session_state.tfidf.transform([clean])
        pred = st.session_state.svc_model.predict(vect)[0]
        label = "REAL" if pred == 1 else "FAKE"

        st.success(f"Prediction: **{label}**")
    else:
        st.info("Train models first.")
