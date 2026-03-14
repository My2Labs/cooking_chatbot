# from google.colab import drive
# drive.mount('/content/drive')

import csv
import numpy as np
import pandas as pd

from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense


# ---------- Load recipes.csv ----------
def load_recipes_dataframe(csv_path, max_rows=2000):
    rows = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            name = row.get("Name", "")
            desc = row.get("Description", "")
            category = row.get("RecipeCategory", "")
            keywords = row.get("Keywords", "")
            ingredients = row.get("RecipeIngredientParts", "")
            instructions = row.get("RecipeInstructions", "")

            keywords = keywords.replace('c("', "").replace('")', "").replace('", "', " ")

            text = (
                f"recipe: {name}\n"
                f"description: {desc}\n"
                f"category: {category}\n"
                f"keywords: {keywords}\n"
                f"ingredients: {ingredients}\n"
                f"instructions: {instructions}\n"
            ).lower()

            rows.append({
                "name": name,
                "description": desc,
                "text": text
            })

    return pd.DataFrame(rows)


# ---------- Build sentiment labels from description ----------
def add_sentiment_labels(df):
    polarities = []
    labels_num = []
    labels_text = []

    for desc in df["description"]:
        desc = str(desc).strip()
        polarity = TextBlob(desc).sentiment.polarity
        polarities.append(polarity)

        if polarity > 0:
            labels_num.append(1)
            labels_text.append("positive")
        else:
            labels_num.append(0)
            labels_text.append("negative")

    df = df.copy()
    df["polarity"] = polarities
    df["sentiment"] = labels_text
    df["label"] = labels_num
    return df


# ---------- Lab 10: TextBlob ----------
def run_textblob(df, limit=10):
    print("\n===== LAB 10: TextBlob =====")

    for i, row in df.head(limit).iterrows():
        print(f"\nDocument {i + 1}: {row['name']}")
        print(f"Description: {row['description']}")
        print(f"Polarity: {row['polarity']:.4f}")
        print(f"Assigned Label: {row['sentiment']}")


# ---------- Lab 11: Scikit-learn ----------
def run_scikit_learn(df):
    print("\n===== LAB 11: Scikit-Learn LinearSVC =====")

    corpus = df["description"].fillna("").astype(str).apply(lambda x: x[:200])
    labels = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        corpus,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(X_train_vec, y_train)

    y_predict = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_predict, target_names=["negative", "positive"]))


# ---------- Lab 12: Keras ----------
def run_keras(df):
    print("\n===== LAB 12: Keras =====")

    corpus = df["description"].fillna("").astype(str).tolist()
    corpus = [text[:200] for text in corpus]   # only short segment
    labels = df["label"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        corpus,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )

    max_vocab = 1000
    max_len = 50
    embed_dim = 16
    lstm_units = 16
    epochs = 2
    batch_size = 16

    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

    model = Sequential([
        Embedding(input_dim=max_vocab, output_dim=embed_dim),
        LSTM(lstm_units),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(
        X_train_pad,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")


def main():
    csv_path = "project_data/original_files/recipes.csv"
    #csv_path = "/content/drive/MyDrive/Colab_Notebooks/recipes.csv"

    df = load_recipes_dataframe(csv_path, max_rows=100)
    df = add_sentiment_labels(df)

    print("Loaded rows:", len(df))
    print("\nLabel counts:")
    print(df["sentiment"].value_counts())

    run_textblob(df, limit=5)
    run_scikit_learn(df)
    run_keras(df)


if __name__ == "__main__":
    main()