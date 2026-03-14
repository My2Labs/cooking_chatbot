import csv
import pandas as pd
from sklearn.decomposition import TruncatedSVD, NMF

from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora
from gensim.models import LdaModel, HdpModel
from gensim.parsing.preprocessing import STOPWORDS
from pprint import pprint

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF



#------- Load a large corpus of text data -------
def load_recipes_csv(csv_path, max_rows=200):
    parts = []
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

            parts.append(
                f"recipe: {name}\n"
                f"description: {desc}\n"
                f"category: {category}\n"
                f"keywords: {keywords}\n"
                f"ingredients: {ingredients}\n"
                f"instructions: {instructions}\n"
            )

    return [p.lower() for p in parts]




#------- Load a large corpus of text data -------
def load_food_parquet(parquet_path, max_rows=200):
    df = pd.read_parquet(parquet_path)

    if max_rows is not None:
        df = df.head(max_rows)

    def col(name):
        if name in df.columns:
            return df[name].fillna("").astype(str)
        return pd.Series([""] * len(df), index=df.index)

    names = col("name")
    desc = col("description")
    tags = col("tags")
    ingredients = col("ingredients")
    steps = col("steps")

    parts = []
    for i in range(len(df)):
        parts.append(
            f"recipe: {names.iloc[i]}\n"
            f"description: {desc.iloc[i]}\n"
            f"tags: {tags.iloc[i]}\n"
            f"ingredients: {ingredients.iloc[i]}\n"
            f"steps: {steps.iloc[i]}\n"
        )

    return [p.lower() for p in parts]




#------- Load a small corpus of text data -------
def load_cooking_conversions(txt_path, max_rows=None):
    docs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break

            line = line.strip()
            if line:
                docs.append(f"conversion: {line}\n")

    return [d.lower() for d in docs]






# ------- 5. Experiment with documents of varying sizes -------
DATASETS = [
    ("recipes_csv", "project_data/original_files/recipes.csv", load_recipes_csv),
    ("food_recipes_parquet", "project_data/original_files/food_recipes.parquet", load_food_parquet),
    ("cooking_conversions", "project_data/original_files/cooking_conversions.txt", load_cooking_conversions),
]

CORPUS_SIZES = [200, 2000]

# ------- 3. Experiment with different epoch values -------
EPOCHS_LIST = [3, 10, 20]

# ------- 2. Experiment with different architecture structures (at least 3 different architectures) -------
ARCHITECTURES = {
    "Arch1_64": {"embed_dim":32, "lstm_units": [64]},
    "Arch2_128": {"embed_dim":64, "lstm_units": [128]},
    "Arch3_128x2": {"embed_dim":64, "lstm_units": [128, 128]},
}

SEQUENCE_LENGTH = 10
MAX_VOCAB = 5000
BATCH_SIZE = 128


def run_lsa(docs, num_topics=5, num_words=10):
    documents = docs
    vectorizer = TfidfVectorizer(stop_words="english", max_features=MAX_VOCAB)
    X = vectorizer.fit_transform(documents)
    lsa = TruncatedSVD(n_components=num_topics, random_state=42)
    lsa.fit(X)
    terms = vectorizer.get_feature_names_out()
    print("\n" + "-" * 40 + "LSA" + "-" * 40)
    for i, comp in enumerate(lsa.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:num_words]
        print("\n")
        print(f"Topic {i}: ")
        for term, weight in sorted_terms:
            print(f"{term}: {weight:4f}")

def run_lda(docs, num_topics=5, num_words=10):
    texts = [doc.split() for doc in docs]
    dictionary = corpora.Dictionary(texts)
    corpus_bow = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)
    print("\n" + "-" * 40 + "LDA" + "-" * 40)
    print(f"LDA Top {num_topics} topics:")
    for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=num_words):
        print("\n")
        print(f"Topic {idx}: ")
        print(f"{topic}")

def run_hdp(docs, num_topics=5, num_words=10):
    texts = [doc.split(', ') for doc in docs]
    dictionary = corpora.Dictionary(texts)
    corpus_bow = [dictionary.doc2bow(text) for text in texts]
    hdp_model = HdpModel(corpus_bow, id2word=dictionary)
    print("\n" + "-" * 40 + "HDP" + "-" * 40)
    print(f"HDP Top {num_topics} topics:")
    for idx, topic in hdp_model.print_topics(num_topics=num_topics, num_words=num_words):
        print("\n")
        print(f"Topic {idx}: ")
        print(f"{topic}")

def run_nmf(docs, num_topics=5, num_words=10):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=MAX_VOCAB)
    X = vectorizer.fit_transform(docs)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    model = NMF(n_components=num_topics, init="random", random_state=0, max_iter=500)
    model.fit(df)
    H = pd.DataFrame(model.components_, columns=df.columns)
    W = pd.DataFrame(model.transform(df))
    V = pd.DataFrame(np.dot(W, H), columns=df.columns)
    V.index = df.index


    print("\n" + "-" * 40 + "NMF" + "-" * 40)
    for topic_idx in range(H.shape[0]):
        topic = H.iloc[topic_idx]
        top_words = topic.sort_values(ascending=False).head(num_words)
        print("\n")
        print(f"Topic {topic_idx}: ")
        for word, weight in top_words.items():
            print(f"{word}: {weight:4f}")



def main():
    for dataset_name, path, loader in DATASETS:
        for max_rows in CORPUS_SIZES:
            print("\n" + "-" * 80)
            print(f"Dataset: {dataset_name} | Rows: {max_rows}")

            docs = loader(path, max_rows=max_rows)
            run_lsa(docs)
            run_lda(docs)
            run_hdp(docs)
            run_nmf(docs)





if __name__ == "__main__":
    main()