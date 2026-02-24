import csv
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


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


# ------- Implement an LSTM model -------
def build_model(vocab_size, embed_dim, lstm_units):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim))

    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        model.add(LSTM(units, return_sequences=return_sequences))

    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        run_eagerly=True,
    )
    return model



# References: https://tomarcher.io/posts/temperature-top-p-creativity-knobs/
# -------     https://cyrilzakka.github.io/llm-playbook/nested/temp.html
def sample_from_probs(probs, temperature=1.0, top_k=40):
    probs = np.asarray(probs).astype(np.float64)


    if temperature is None or temperature <= 0:
        temperature = 1.0

    logits = np.log(probs + 1e-12) / temperature
    exp = np.exp(logits - np.max(logits))
    probs_t = exp / np.sum(exp)

    if top_k is not None and top_k > 0 and top_k < len(probs_t):
        top_idx = np.argpartition(probs_t, -top_k)[-top_k:]
        top_probs = probs_t[top_idx]
        top_probs = top_probs / np.sum(top_probs)
        return int(np.random.choice(top_idx, p=top_probs))

    return int(np.random.choice(len(probs_t), p=probs_t))



# ------- 4. Generate a new text that mimics the style of your chatbot document -------
def generate_text(model, tokenizer, seed_text, num_words=60, temperature=0.9, top_k=40):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=SEQUENCE_LENGTH, padding="pre")
        token_tensor = tf.convert_to_tensor(token_list, dtype=tf.int32)
        preds = model(token_tensor, training=False).numpy()[0]
        next_id = sample_from_probs(preds, temperature=temperature, top_k=top_k)
        next_word = tokenizer.index_word.get(next_id, "")
        if not next_word:
            break

        last_words = seed_text.split()[-3:]
        if next_word in last_words:
            next_id = sample_from_probs(preds, temperature=max(temperature, 1.2), top_k=top_k)
            next_word = tokenizer.index_word.get(next_id, "")
            if not next_word:
                break

        seed_text += " " + next_word
    return seed_text




def train_on_batch_loop(model, X, y, batch_size, epochs, steps_per_epoch, log_every=25, shuffle=True):
    n = len(X)

    for ep in range(1, epochs + 1):
        print(f"Epoch {ep}/{epochs}:", flush=True)
        t0 = time.time()

        if shuffle:
            idx = np.random.permutation(n)
        else:
            idx = np.arange(n)

        for step in range(steps_per_epoch):
            start = (step * batch_size) % n
            batch_idx = idx[start:start + batch_size]
            if len(batch_idx) < batch_size:
                batch_idx = np.concatenate([batch_idx, idx[:batch_size - len(batch_idx)]])

            model.train_on_batch(X[batch_idx], y[batch_idx])

            if step % log_every == 0:
                print(f" step {step+1}/{steps_per_epoch}", flush=True)

        print("  epoch_seconds: ", round(time.time() - t0, 2), flush=True)



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




def main():
    for dataset_name, path, loader in DATASETS:
        for max_rows in CORPUS_SIZES:

            docs = loader(path, max_rows=max_rows)

            print("dataset: ", dataset_name, flush=True)
            print("max_rows: ", max_rows, flush=True)
            print("length of docs: ", len(docs), flush=True)
            print("preview:\n", "\n".join(docs[:3])[:800], "\n", flush=True)
            print("---------------\n")

            tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>", lower=True)
            tokenizer.fit_on_texts(docs)
            seqs = tokenizer.texts_to_sequences(docs)
            sequences = np.array([tok for s in seqs for tok in s], dtype=np.int32)
            vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) +1)

            if len(sequences) <= SEQUENCE_LENGTH + 1:
                continue

            windows = np.lib.stride_tricks.sliding_window_view(
                sequences, window_shape=SEQUENCE_LENGTH +1
            )
            X = np.ascontiguousarray(windows[:, :-1], dtype=np.int32)
            y = np.ascontiguousarray(windows[:, -1], dtype=np.int32)

            steps_per_epoch = max(1, len(X) // BATCH_SIZE)
            steps_used = min(steps_per_epoch, 50)

            for arch_name, config in ARCHITECTURES.items():
                for epoch in EPOCHS_LIST:
                    tf.keras.backend.clear_session()

                    model = build_model(
                        vocab_size=vocab_size,
                        embed_dim=config["embed_dim"],
                        lstm_units=config["lstm_units"],
                    )
                    t0 = time.time()
                    train_on_batch_loop(
                        model,
                        X,
                        y,
                        batch_size=BATCH_SIZE,
                        epochs=epoch,
                        steps_per_epoch=steps_used,
                        log_every=50,
                        shuffle=True,
                    )
                    train_seconds = round(time.time() - t0, 2)
                    final_loss = float(model.test_on_batch(X[:BATCH_SIZE], y[:BATCH_SIZE]))


                    if dataset_name == "cooking_conversions":
                        seed = "conversion: 1 cup equals"
                    else:
                        seed = "preheat oven to"

                    sample = generate_text(
                        model,
                        tokenizer,
                        seed,
                        num_words=80,
                        temperature=0.9,
                        top_k=40,
                    )

                    print(f"\n------- RESULT DATASET {dataset_name} -------", flush=True)
                    print(f"------- ROWS: {max_rows} -------", flush=True)
                    print(f"------- ARCHITECTURE: {arch_name} -------", flush=True)
                    print(f"------- EPOCH: {epoch} -------", flush=True)
                    print(f"------- TRAIN SECONDS: {train_seconds} -------", flush=True)
                    print(f"------- FINAL LOSS: {final_loss} -------", flush=True)
                    print(f"------- SAMPLE: {sample} -------", flush=True)
                    print("-" * 80, flush=True)



if __name__ == "__main__":
    main()