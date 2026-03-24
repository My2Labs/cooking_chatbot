# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge import Rouge

#LOCAL PATH
df = pd.read_parquet("project_data/original_files/food_recipes.parquet")

#GOOGLE COLAB PATH
# df = pd.read_parquet("/content/drive/MyDrive/Colab_Notebooks/data/food_recipes.parquet")
text = " ".join(df["description"].dropna().astype(str).head(300))



# Extractive Summarization Using TextRank
sentences = [s for s in sent_tokenize(text) if s.strip()]
stop_words = set(stopwords.words('english'))

def normalize(s):
    tokens = [w for w in word_tokenize(s.lower()) if w.isalnum() and w not in stop_words]
    return " ".join(tokens)

normalized = [normalize(s) for s in sentences]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(normalized)

sim_matrix = cosine_similarity(X, X)
np.fill_diagonal(sim_matrix, 0.0)

graph = nx.from_numpy_array(sim_matrix)
scores = nx.pagerank(graph)

extractive_num_sentences = 2
extractive_ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
extractive_summary_sentences = [s for _, s in extractive_ranked[:extractive_num_sentences]]
extractive_summary = " ".join(extractive_summary_sentences)

print("Extractive Summarization Using TextRank:")
print(extractive_summary)


# ----------------------------------------------------------------------------------------
# Abstractive Summarization with T5
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
abstractive_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\n")
print("------------------------------------------------------------------------")
print("Abstractive Summarization using T5:")
print(abstractive_summary)


# ----------------------------------------------------------------------------------------

print("\n")
print("------------------------------------------------------------------------")
print("Rouge Comparison:")
ROUGE = Rouge()
reference = extractive_summary
candidate = abstractive_summary
print(ROUGE.get_scores(candidate, reference))
