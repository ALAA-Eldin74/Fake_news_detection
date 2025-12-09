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

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector – Streamlit App (Medium Level)")


# =================================================================
# GLOBAL STORAGE (Session State)
# =================================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None
if "log_model" not in st.session_state:
    st.session_state.log_model = None
if "svc_model" not in st.session_state:
    st.session_state.svc_model = None


# =================================================================
# 1) Load Dataset
# =================================================================
st.header("1️⃣ Load Dataset")

uploaded = st.file_uploader("Upload your news.csv file", type=["csv"])

if st.button("Load Data"):
    if uploaded is None:
        st.error("❌ Please upload CSV file first.")
    else:
        df = pd.read_csv(uploaded, on_bad_lines="skip", engine="python")
        df.dropna(inplace=True)

        st.session_state.df = df
        st.success("Dataset Loaded Successfully!")
        st.dataframe(df.head())


# =================================================================
# 2) Preprocessing
# =================================================================
st.header("2️⃣ Preprocess & Clean Text")

if st.button("Start Cleaning"):
    if st.session_state.df is None:
        st.error("❌ Load dataset first.")
    else:
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
        st.success("Cleaning Completed Successfully!")
        st.dataframe(df[["clean", "label"]].head())


# =================================================================
# 3) Split Data
# =================================================================
st.header("3️⃣ Split Train/Test")

if st.button("Split Data"):
    if st.session_state.df is None or "clean" not in st.session_state.df:
        st.error("❌ Clean the data first.")
    else:
        df = st.session_state.df
        X = df["clean"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        st.success("Data split completed!")
        st.write("Train size:", len(X_train))
        st.write("Test size:", len(X_test))


# =================================================================
# 4) TF-IDF
# =================================================================
st.header("4️⃣ Train TF-IDF Vectorizer")

if st.button("Train TF-IDF"):
    if st.session_state.X_train is None:
        st.error("❌ Split data first.")
    else:
        tfidf = TfidfVectorizer(max_features=5000)
        X_train_vec = tfidf.fit_transform(st.session_state.X_train)
        X_test_vec = tfidf.transform(st.session_state.X_test)

        st.session_state.tfidf = tfidf
        st.session_state.X_train_vec = X_train_vec
        st.session_state.X_test_vec = X_test_vec

        st.success("TF-IDF trained successfully!")
        st.write("Train Matrix Shape:", X_train_vec.shape)
        st.write("Test Matrix Shape:", X_test_vec.shape)


# =================================================================
# 5) Train Models
# =================================================================
st.header("5️⃣ Train ML Models")

col1, col2 = st.columns(2)

# Train Logistic Regression
with col1:
    if st.button("Train Logistic Regression"):
        if st.session_state.tfidf is None:
            st.error("❌ Train TF-IDF first.")
        else:
            log_model = LogisticRegression()
            log_model.fit(st.session_state.X_train_vec, st.session_state.y_train)

            st.session_state.log_model = log_model
            st.success("Logistic Regression Trained ✔")


# Train SVC
with col2:
    if st.button("Train SVM (Linear Kernel)"):
        if st.session_state.tfidf is None:
            st.error("❌ Train TF-IDF first.")
        else:
            svc = SVC(kernel="linear")
            svc.fit(st.session_state.X_train_vec, st.session_state.y_train)

            st.session_state.svc_model = svc
            st.success("SVM Trained ✔")


# =================================================================
# 6) Evaluate Models
# =================================================================
st.header("6️⃣ Evaluate Models")

if st.button("Evaluate"):
    if st.session_state.log_model is None or st.session_state.svc_model is None:
        st.error("❌ Train both models first.")
    else:
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

        # Confusion Matrix for SVM
        cm = confusion_matrix(y_test, pred_svc)
        st.subheader("Confusion Matrix (SVM)")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)


# =================================================================
# 7) Predict User Text
# =================================================================
st.header("7️⃣ Try Your Own Text")

user_text = st.text_area("Enter news text to check if it’s FAKE or REAL")

if st.button("Predict Text"):
    if st.session_state.tfidf is None:
        st.error("❌ Train models first.")
    else:
        clean = user_text.lower()
        clean = re.sub(r"http\S+", "", clean)
        clean = re.sub(f"[{string.punctuation}]", "", clean)
        clean = re.sub(r"\d+", "", clean)

        vect = st.session_state.tfidf.transform([clean])

        pred = st.session_state.svc_model.predict(vect)[0]
        label = "REAL" if pred == 1 else "FAKE"

        st.success(f"Prediction: **{label}**")
