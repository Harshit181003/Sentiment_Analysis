import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Amazon Sentiment Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("sentiment_cleaned.csv")
    df['sentiment'] = df['sentiment'].str.lower().str.strip()
    return df

df = load_data()

st.title("📦 Amazon Review Sentiment Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", len(df))
col2.metric("Positive", (df['sentiment'] == 'positive').sum())
col3.metric("Negative", (df['sentiment'] == 'negative').sum())

col4, col5 = st.columns(2)

with col4:
    st.markdown("### 📊 Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

with col5:
    st.markdown("### 🧮 Score Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x='Score', hue='sentiment', ax=ax2)
    ax2.set_title("Score vs Sentiment")
    st.pyplot(fig2)

with st.expander("🔍 Show Raw Data"):
    st.dataframe(df.sample(50))

import joblib

model = joblib.load("sentiment_model.pkl")           
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.markdown("### 💬 Predict Sentiment of Your Own Review")

user_input = st.text_area("Enter your review text below:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        clean_input = user_input.lower()
        vectorized = vectorizer.transform([clean_input])
        pred = model.predict(vectorized)
        sentiment = label_encoder.inverse_transform(pred)[0]
        st.success(f"🧠 Predicted Sentiment: **{sentiment.upper()}**")
    else:
        st.warning("Please enter some review text.")
