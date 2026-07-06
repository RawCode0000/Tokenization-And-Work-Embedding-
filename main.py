import pandas as pd

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud

import nltk

from nltk.corpus import stopwords

import string

nltk.download("stopwords")

df = pd.read_csv("data/bbc-text.csv")

df.head()

df.columns

stop_words = set(stopwords.words("english"))

def clean(text):

    text = text.lower()

    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

df["clean_text"] = df["text"].apply(clean)

vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(df["clean_text"])

print(df.head())

print(df.columns)

print(df.shape)

print("========== PROJECT STATUS ==========")
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("TF-IDF Matrix Shape:", tfidf.shape)
print("Program executed successfully!")

feature_names = vectorizer.get_feature_names_out()

for category in df["category"].unique():

    print("\n==============================")
    print("Category:", category)

    category_text = " ".join(df[df["category"] == category]["clean_text"])

    category_tfidf = vectorizer.transform([category_text])

    scores = category_tfidf.toarray()[0]

    top_indices = scores.argsort()[-20:][::-1]

    top_words = [feature_names[i] for i in top_indices]

    print(top_words)

import os

os.makedirs("images", exist_ok=True)

for category in df["category"].unique():

    category_text = " ".join(df[df["category"] == category]["clean_text"])

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(category_text)

    filename = f"images/{category}.png"

    wordcloud.to_file(filename)

    print(f"Saved {filename}")


import numpy as np

for category in df["category"].unique():

    category_text = " ".join(df[df["category"] == category]["clean_text"])

    category_tfidf = vectorizer.transform([category_text])

    scores = category_tfidf.toarray()[0]

    top_indices = scores.argsort()[-10:][::-1]

    top_words = [feature_names[i] for i in top_indices]

    top_scores = [scores[i] for i in top_indices]

    plt.figure(figsize=(10,5))
    plt.bar(top_words, top_scores)
    plt.xticks(rotation=45)
    plt.title(f"Top TF-IDF Words - {category}")
    plt.tight_layout()

    plt.savefig(f"images/{category}_graph.png")
    plt.close()

print("\nCategories Present:")
print(df["category"].unique())