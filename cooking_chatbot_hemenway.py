import csv
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.lm import Vocabulary

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical


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

# ------- 2. Experiment with different architecture structures (at least 3 different architectures) -------
ARCHITECTURES = {
    "Arch1_64": {"embed_dim":32, "lstm_units": [64]},
    "Arch2_128": {"embed_dim":64, "lstm_units": [128]},
    "Arch3_128x2": {"embed_dim":64, "lstm_units": [128, 128]},
}

# ------- 3. Experiment with different epoch values -------
EPOCHS_LIST = [3, 10, 20]


# ------- 5. Experiment with documents of varying sizes -------
DATASETS = [
    ("recipes_csv", "project_data/original_files/recipes.csv", load_recipes_csv),
    ("food_recipes_parquet", "project_data/original_files/food_recipes.parquet", load_food_parquet),
    ("cooking_conversions", "project_data/original_files/cooking_conversions.txt", load_cooking_conversions),
]

CORPUS_SIZES = [200, 2000]

SEQUENCE_LENGTH = 10
MAX_VOCAB = 5000
BATCH_SIZE = 128

# docs_recipes = load_recipes_csv("project_data/original_files/recipes.csv")
# docs_food_parquet = load_food_parquet("project_data/original_files/food_recipes.parquet")
# docs_conversions = load_cooking_conversions("project_data/original_files/cooking_conversions.txt")
#
# chars_recipes = sorted(set("".join(docs_recipes)))
# chars_food_parquet = sorted(set("".join(docs_food_parquet)))
# chars_conversions = sorted(set("".join(docs_conversions)))
#
# char_to_idx_recipes = {char: idx for idx, char in enumerate(chars_recipes)}
# char_to_idx_food_parquet = {char: idx for idx, char in enumerate(chars_food_parquet)}
# char_to_idx_conversions = {char: idx for idx, char in enumerate(chars_conversions)}
#
# idx_to_char_recipes = {idx: char for char, idx in char_to_idx_recipes.items()}
# idx_to_char_food_parquet = {idx: char for char, idx in char_to_idx_food_parquet.items()}
# idx_to_char_conversions = {idx: char for char, idx in char_to_idx_conversions.items()}


def build_vocab_from_docs(docs, max_chars=None):
    text = "".join(docs)
    if max_chars is not None:
        text = text[:max_chars]
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return text, chars, char_to_idx, idx_to_char

def make_char_sequences(text, seq_len, char_to_idx):
    X = []
    y = []
    for i in range(len(text) - seq_len):
        seq = text[i:i + seq_len]
        nxt = text[i + seq_len]
        X.append([char_to_idx[c] for c in seq])
        y.append(char_to_idx[nxt])

    X = np.array(X).reshape((-1, seq_len, 1))
    y = to_categorical(y, num_classes=len(char_to_idx))
    return X, y



def train_model(text, chars, char_to_idx, seq_len, epochs, lstm_units_list):
    X, y = make_char_sequences(text, seq_len, char_to_idx)

    model = Sequential()

    for layer_i, units in enumerate(lstm_units_list):
        last_layer = (layer_i == len(lstm_units_list) - 1)
        model.add(
            LSTM(units, input_shape=(seq_len, 1) if layer_i == 0 else None, return_sequences=not last_layer)
        )

    # model.add(LSTM(lstm_units, input_shape=(seq_len, 1)))
    model.add(Dense(len(chars), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X, y, epochs=epochs, batch_size=BATCH_SIZE, verbose=1)
    return model

# ------- 4. Generate a new text that mimics the style of your chatbot document -------
def generate_text(model, start_string, num_generate, seq_len, char_to_idx, idx_to_char):
    start_string = "".join([c for c in start_string if c in char_to_idx])
    if len(start_string) < 1:
        return ""
    if len(start_string) < seq_len:
        start_string = (" " * (seq_len - len(start_string))) + start_string
    else:
        start_string = start_string[-seq_len:]

    input_eval = np.array([char_to_idx[c] for c in start_string]).reshape((1, seq_len, 1))

    text_generated = []
    for i in range(num_generate):
        predictions = model.predict(input_eval, verbose=0)
        predicted_id = int(np.argmax(predictions[0]))

        text_generated.append(idx_to_char[predicted_id])
        input_eval = np.concatenate([input_eval[:, 1:, :], [[[predicted_id]]]], axis=1)

    return start_string + ''.join(text_generated)



def main():
    for dataset_name, path, loader in DATASETS:
        for max_rows in CORPUS_SIZES:
            docs = loader(path, max_rows=max_rows)
            text, chars, char_to_idx, idx_to_char = build_vocab_from_docs(docs, max_chars=50_000)

            if dataset_name == "cooking_conversions":
                start_string = "conversion: 1 cup equals"
            else:
                start_string = "preheat oven to"


            print("dataset: ", dataset_name, flush=True)
            print("max_rows: ", max_rows, flush=True)
            print("length of docs: ", len(docs), flush=True)
            print("preview:\n", "\n".join(docs[:3])[:800], "\n", flush=True)
            print("---------------\n")

            for architecture in ARCHITECTURES:
                for epochs in EPOCHS_LIST:
                    print("architecture: ", architecture, flush=True)
                    print("epochs: ", epochs, flush=True)
                    model = train_model(
                        text = text,
                        chars = chars,
                        char_to_idx = char_to_idx,
                        seq_len = SEQUENCE_LENGTH,
                        epochs = epochs,
                        lstm_units_list = cfg["lstm_units"]
                    )
                    sample = generate_text(
                        model = model,
                        start_string = start_string,
                        num_generate = 50,
                        seq_len = SEQUENCE_LENGTH,
                        char_to_idx = char_to_idx,
                        idx_to_char = idx_to_char
                    )
                    print("Sample: \n", sample, "\n", flush=True)



if __name__ == "__main__":
    main()