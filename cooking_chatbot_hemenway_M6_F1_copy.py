import csv
import re
import pandas as pd

from gensim import corpora
from gensim.models import LdaModel, HdpModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from pprint import pprint


# ------- Load a large corpus of text data -------
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


# ------- Load a large corpus of text data -------
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


# ------- Load a small corpus of text data -------
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


# ------- Dataset settings -------
DATASETS = [
    ("recipes_csv", "project_data/original_files/recipes.csv", load_recipes_csv),
    ("food_recipes_parquet", "project_data/original_files/food_recipes.parquet", load_food_parquet),
    ("cooking_conversions", "project_data/original_files/cooking_conversions.txt", load_cooking_conversions),
]

CORPUS_SIZES = [200, 2000]

# Topic-model settings
NUM_TOPICS = 5
NUM_WORDS = 10
RANDOM_STATE = 42


# ------- Text preprocessing -------
def preprocess_text(doc):
    """
    Tokenize and clean text for Gensim topic models.
    """
    tokens = simple_preprocess(doc, deacc=True)  # lowercase, remove punctuation
    tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    return tokens


# ------- Display helper -------
def print_topic_words(model_name, components, feature_names, top_n=10):
    print(f"\n===== {model_name} Topics =====")
    for topic_idx, topic in enumerate(components):
        top_features_idx = topic.argsort()[::-1][:top_n]
        top_terms = [feature_names[i] for i in top_features_idx]
        print(f"Topic {topic_idx}: {', '.join(top_terms)}")


# ------- LSA -------
def run_lsa(documents, num_topics=5, num_words=10):
    print("\n===== Running LSA =====")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    X = vectorizer.fit_transform(documents)

    lsa_model = TruncatedSVD(n_components=num_topics, random_state=RANDOM_STATE)
    lsa_model.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    print_topic_words("LSA", lsa_model.components_, feature_names, top_n=num_words)


# ------- NMF -------
def run_nmf(documents, num_topics=5, num_words=10):
    print("\n===== Running NMF =====")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    X = vectorizer.fit_transform(documents)

    nmf_model = NMF(
        n_components=num_topics,
        random_state=RANDOM_STATE,
        init="nndsvda",
        max_iter=200
    )
    nmf_model.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    print_topic_words("NMF", nmf_model.components_, feature_names, top_n=num_words)


# ------- LDA -------
def run_lda(documents, num_topics=5, num_words=10):
    print("\n===== Running LDA =====")
    texts = [preprocess_text(doc) for doc in documents]

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.5)

    corpus_bow = [dictionary.doc2bow(text) for text in texts]

    if len(dictionary) == 0:
        print("Skipping LDA: dictionary is empty after filtering.")
        return

    lda_model = LdaModel(
        corpus=corpus_bow,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=RANDOM_STATE,
        passes=10
    )

    print("Topics:")
    pprint(lda_model.print_topics(num_topics=num_topics, num_words=num_words))


# ------- HDP -------
def run_hdp(documents, num_topics=5, num_words=10):
    print("\n===== Running HDP =====")
    texts = [preprocess_text(doc) for doc in documents]

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.5)

    corpus_bow = [dictionary.doc2bow(text) for text in texts]

    if len(dictionary) == 0:
        print("Skipping HDP: dictionary is empty after filtering.")
        return

    hdp_model = HdpModel(
        corpus=corpus_bow,
        id2word=dictionary
    )

    print("Topics:")
    pprint(hdp_model.print_topics(num_topics=num_topics, num_words=num_words))


def main():
    for dataset_name, path, loader in DATASETS:
        for max_rows in CORPUS_SIZES:
            print("\n" + "=" * 80)
            print(f"DATASET: {dataset_name} | ROWS: {max_rows}")
            print("=" * 80)

            docs = loader(path, max_rows=max_rows)

            if not docs:
                print("No documents loaded.")
                continue

            print(f"Loaded {len(docs)} documents.")

            # Run all topic modeling methods
            run_lsa(docs, num_topics=NUM_TOPICS, num_words=NUM_WORDS)
            run_nmf(docs, num_topics=NUM_TOPICS, num_words=NUM_WORDS)
            run_lda(docs, num_topics=NUM_TOPICS, num_words=NUM_WORDS)
            run_hdp(docs, num_topics=NUM_TOPICS, num_words=NUM_WORDS)


if __name__ == "__main__":
    main()