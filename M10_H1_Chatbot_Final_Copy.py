# workspace/Chatbot/
# data/..
#         cooking_conversions.txt
#         food_recipes.parquet
#         recipes.csv
#         squad_cooking_transformed.json
# chatbot.ipynb


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/dev/null"

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# !pip install rouge-score -q
# !pip install tensorflow=2.15.0

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# COSC524 - Natural Language Processing
# Dr. Joshua Fagan
#
# M10.H1 Chatbot
# By: Sharron Creasy Hemenway
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


import json
import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split


SQUAD_COOKING_JSON = "/workspace/Chatbot/data/squad_cooking_transformed.json"
CONVERSIONS_TXT = "/workspace/Chatbot/data/cooking_conversions.txt"
RECIPES_CSV = "/workspace/Chatbot/data/recipes.csv"


START_TOKEN = "<START>"
END_TOKEN = "<END>"

MAX_ROWS = 10000
RECIPE_ROWS = 1000
dimensionality = 256
embedding_dim = 128
batch_size = 32
epochs = 100

######################################
# ------- DATA PREPROCESSING -------
######################################


def clean_text(text):
    text = str(text).lower().strip()
    text = text.replace("Â°", " degrees ")
    text = text.replace("°", " degrees ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9,.?!:/()'\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def shorten_context(text, max_words=240):
    words = clean_text(text).split()
    return " ".join(words[:max_words])

def clean_recipe_field(text):
    text = str(text)
    text = text.replace("c(", "")
    text = text.replace(")", "")
    text = text.replace('"', "")
    text = text.replace(",", ", ")
    return clean_text(text)

def limit_words(text, max_words=40):
    words = clean_text(text).split()
    return " ".join(words[:max_words])

def remove_repeated_words(text):
    words = text.split()
    cleaned = []
    for word in words:
        if not cleaned or cleaned[-1] != word:
            cleaned.append(word)
    return " ".join(cleaned)

def parse_fraction(text):
    text = text.strip()
    if "/" in text:
        try:
            top, bottom = text.split("/")
            return float(top) / float(bottom)
        except:
            return None

def handle_conversion(user_input):
    text = clean_text(user_input)
    conversion_rates = {
        ("cup", "ounces"): 8,
        ("cup", "tablespoons"): 16,
        ("cup", "teaspoons"): 48,
        ("tablespoon", "teaspoons"): 3,
        ("tablespoons", "teaspoons"): 3,
        ("pint", "cups"): 2,
        ("quart", "cups"): 4,
        ("gallon", "cups"): 16,
    }
    quantity_match = re.search(r"(\d+\s*/\s*\d+|\d+(\.\d+)?)", text)
    quantity = 1.0
    if quantity_match:
        quantity_string = quantity_match.group(1).replace(" ", "")
        parsed = parse_fraction(quantity_string)
        if parsed is not None:
            quantity = parsed
    for (from_unit, to_unit), rate in conversion_rates.items():
        if from_unit in text and to_unit in text:
            result = quantity * rate
            if result.is_integer():
                result = int(result)
            quantity_display = int(quantity) if quantity.is_integer() else quantity
            return f"{quantity_display} {from_unit} equals {result} {to_unit}"
    return None


##########################################
# ------- DATA SOURCES/COLLECTION -------
##########################################



def load_squad_cooking(path, max_rows=None, include_context=False):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []

    for i, item in enumerate(data):
        if max_rows is not None and i >= max_rows:
            break

        context = clean_text(item.get("context", ""))
        question = clean_text(item.get("question", ""))
        answers = item.get("answers", {})
        answer_text = answers.get("text", "")

        if isinstance(answer_text, list):
            answer_text = answer_text[0] if answer_text else ""

        answer_text = clean_text(answer_text)

        if not question or not answer_text:
            continue

        if include_context:
            context = shorten_context(context, max_words=240)
            input_text = f"{context} {question}"
        else:
            input_text = question

        target_text = f"{START_TOKEN} {answer_text} {END_TOKEN}"
        pairs.append((input_text, target_text))

    return pairs


def load_recipe_csv_pairs(path, max_rows=1000):
    df = pd.read_csv(path, low_memory=False).head(max_rows)
    pairs = []
    for _, row in df.iterrows():
        name = clean_text(row.get("Name", ""))
        total_time = clean_text(row.get("TotalTime", ""))
        servings = str(row.get("RecipeServings", ""))
        calories = str(row.get("Calories", ""))

        ingredients_raw = clean_recipe_field(row.get("RecipeIngredientParts", ""))
        instructions_raw = clean_recipe_field(row.get("RecipeInstructions", ""))
        ingredients = limit_words(ingredients_raw, max_words=25)
        instructions = limit_words(instructions_raw, max_words=40)
        ingredients = remove_repeated_words(clean_text(ingredients))
        instructions = remove_repeated_words(clean_text(instructions))

        if name and ingredients and ingredients != "nan":
            pairs.append(
                (
                    f"what ingredients are needed for {name}?",
                    f"{START_TOKEN} {ingredients} {END_TOKEN}",
                )
            )
        if name and instructions and instructions != "nan":
            pairs.append(
                (
                    f"how do i make {name}?",
                    f"{START_TOKEN} {instructions} {END_TOKEN}",
                )
            )
        if name and total_time and total_time != "nan" and total_time != "pt0s":
            pairs.append(
                (
                    f"how long does {name} take to make?",
                    f"{START_TOKEN} {total_time} {END_TOKEN}",
                )
            )
        if name and servings and servings != "nan":
            pairs.append(
                (
                    f"how many servings does {name} make?",
                    f"{START_TOKEN} {servings} {END_TOKEN}",
                )
            )
        if name and calories and calories != "nan":
            pairs.append(
                (
                    f"how many calories are in {name}?",
                    f"{START_TOKEN} {calories} {END_TOKEN}",
                )
            )
    return pairs

def load_conversion_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = clean_text(line.strip())

            if not line or " equals " not in line:
                continue
            left, right = line.split(" equals ", 1)
            unit = right.split()[-1]
            answer = f"{START_TOKEN} {left} equals {right} {END_TOKEN}"
            pairs.append((f"how many {unit} are in {left}?", answer))
            pairs.append((f"convert {left} to {unit}", answer))
    return pairs



#################################
# ------- BUILD DATASET -------
#################################


recipe_lookup_df = None

def load_recipe_lookup_df(path, max_rows=1000):
    df = pd.read_csv(path, low_memory=False).head(max_rows)
    return df

def handle_recipe_lookup(user_input):
    global recipe_lookup_df
    if recipe_lookup_df is None:
        return None
    text = clean_text(user_input)
    for _, row in recipe_lookup_df.iterrows():
        name = clean_text(row.get("Name", ""))
        if not name or name == "nan":
            continue
        if name in text:
            ingredients = clean_recipe_field(row.get("RecipeIngredientParts", ""))
            instructions = clean_recipe_field(row.get("RecipeInstructions", ""))
            total_time = clean_recipe_field(row.get("TotalTime", ""))
            servings = clean_recipe_field(row.get("RecipeServings", ""))
            calories = clean_recipe_field(row.get("Calories", ""))

            if "ingredient" in text and ingredients and ingredients != "nan":
                return f"The ingredients for {name} are: {limit_words(ingredients, 40)}."
            if "how do i make" in text or "instructions" in text or "recipe" in text:
                if instructions and instructions != "nan":
                    return f"To make {name}: {limit_words(instructions, 50)}."
            if "how long" in text or "time" in text:
                if total_time and total_time != "nan" and total_time != "pt0s":
                    return f"{name} takes {total_time} to make."
            if "serving" in text:
                if servings and servings != "nan":
                    servings = re.sub(r"\b(\d+)\s*\.\s*0\b", r"\1", servings)
                    return f"{name} makes {servings} servings."
            if "calorie" in text:
                if calories and calories != "nan":
                    calories = re.sub(r"\b(\d+)\s*\.\s*0\b", r"\1", calories)
                    return f"{name} has {calories} calories."
    return None

recipe_lookup_df = load_recipe_lookup_df(RECIPES_CSV, max_rows=RECIPE_ROWS)

pairs_squad = load_squad_cooking(SQUAD_COOKING_JSON, max_rows=MAX_ROWS, include_context=False)
conversion_pairs = load_conversion_pairs(CONVERSIONS_TXT)
recipe_pairs = load_recipe_csv_pairs(RECIPES_CSV, max_rows=RECIPE_ROWS)

pairs = pairs_squad + conversion_pairs + recipe_pairs

print("Squad pairs: ", len(pairs_squad))
print("Conversion pairs: ", len(conversion_pairs))
print("Recipe pairs: ", len(recipe_pairs))
print("Total pairs: ", len(pairs))

for i in range(min(10, len(pairs))):
    print("INPUT: ", pairs[i][0])
    print("OUTPUT: ", pairs[i][1])
    print("-"* 80)



#########################
# ------- TRAIN -------
#########################

train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
print("Training pairs: ", len(train_pairs))
print("Testing pairs: ", len(test_pairs))

input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()

token_pattern = r"<START>|<END>|\d+/\d+|[\w']+|[^\s\w]"

for input_text, target_text in train_pairs:
    input_docs.append(input_text)
    target_docs.append(target_text)
    input_words = re.findall(token_pattern, input_text)
    target_words = re.findall(token_pattern, target_text)
    for token in input_words:
        input_tokens.add(token)
    for token in target_words:
        target_tokens.add(token)

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))
input_features_dict = {token: i + 1 for i, token in enumerate(input_tokens)}
target_features_dict = {token: i + 1 for i, token in enumerate(target_tokens)}
reverse_target_features_dict = {i: token for token, i in target_features_dict.items()}
num_encoder_tokens = len(input_tokens) + 1
num_decoder_tokens = len(target_tokens) + 1
max_encoder_seq_length = max(len(re.findall(token_pattern, txt)) for txt in input_docs)
max_decoder_seq_length = max(len(re.findall(token_pattern, txt)) for txt in target_docs)

print("num_encoder_tokens: ", num_encoder_tokens)
print("num_decoder_tokens: ", num_decoder_tokens)
print("max_encoder_seq_length: ", max_encoder_seq_length)
print("max_decoder_seq_length: ", max_decoder_seq_length)

encoder_input_data = np.zeros((len(input_docs), max_encoder_seq_length), dtype="int32")
decoder_input_data = np.zeros((len(input_docs), max_decoder_seq_length), dtype="int32")
decoder_target_data = np.zeros((len(input_docs), max_decoder_seq_length), dtype="int32")

for i, (input_text, target_text) in enumerate(zip(input_docs, target_docs)):
    for t, token in enumerate(re.findall(token_pattern, input_text)):
        encoder_input_data[i, t] = input_features_dict[token]
    for t, token in enumerate(re.findall(token_pattern, target_text)):
        decoder_input_data[i, t] = target_features_dict[token]
        if t > 0:
            decoder_target_data[i, t - 1] = target_features_dict[token]




#########################
# ------- MODEL -------
#########################

encoder_inputs = Input(shape=(None,), name="encoder_inputs")
encoder_embedding_layer = Embedding(
    input_dim=num_encoder_tokens,
    output_dim=embedding_dim,
    mask_zero=True,
    name="encoder_embedding"
)

encoder_embedding = encoder_embedding_layer(encoder_inputs)

encoder_lstm = LSTM(
    dimensionality,
    return_sequences=True,
    return_state=True,
    name="encoder_lstm"
)

encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_embedding)
encoder_states = [state_hidden, state_cell]
decoder_inputs = Input(shape=(None,), name="decoder_inputs")
decoder_embedding_layer = Embedding(
    input_dim=num_decoder_tokens,
    output_dim=embedding_dim,
    mask_zero=True,
    name="decoder_embedding"
)
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_lstm = LSTM(
    dimensionality,
    return_sequences=True,
    return_state=True,
    name="decoder_lstm"
)

decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states,
)

attention_layer = Attention(name="attention_layer")
attention_output = attention_layer([decoder_outputs, encoder_outputs])
decoder_concat = Concatenate(axis=-1, name="concat_layer")(
    [decoder_outputs, attention_output],
)

decoder_dense = Dense(
    num_decoder_tokens,
    activation="softmax",
    name="decoder_dense"
)

decoder_outputs = decoder_dense(decoder_concat)

training_model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)

training_model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

training_model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = training_model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stop]
)



####################################
# ------- INFERENCE MODELS --------
####################################

encoder_model = Model(encoder_inputs, [encoder_outputs, state_hidden, state_cell])
decoder_input_single = Input(shape=(None,), name="decoder_input_single")
encoder_output_input = Input(shape=(None, dimensionality), name="encoder_output_input")
decoder_state_input_hidden = Input(shape=(dimensionality,), name="decoder_state_input_hidden")
decoder_state_input_cell = Input(shape=(dimensionality,), name="decoder_state_input_cell")
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
decoder_embedding_inf = decoder_embedding_layer(decoder_input_single)
decoder_outputs_inf, state_hidden_inf, state_cell_inf = decoder_lstm(decoder_embedding_inf, initial_state=decoder_states_inputs)
attention_output_inf = attention_layer([decoder_outputs_inf, encoder_output_input])
decoder_concat_inf = Concatenate(axis=-1, name="concat_layer_inf")([decoder_outputs_inf, attention_output_inf])
decoder_outputs_inf = decoder_dense(decoder_concat_inf)
decoder_model = Model([decoder_input_single, encoder_output_input] + decoder_states_inputs, [decoder_outputs_inf, state_hidden_inf, state_cell_inf])



#########################
# ------- DECODE -------
#########################

def decode_response(test_input):
    encoder_outs, state_h, state_c = encoder_model.predict(test_input, verbose=0)
    states_value = [state_h, state_c]
    target_seq = np.zeros((1, 1), dtype="int32")
    target_seq[0, 0] = target_features_dict[START_TOKEN]
    decoded_sentence = []
    used_tokens = {}
    while True:
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq, encoder_outs] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict.get(sampled_token_index, "")
        if sampled_token == "" or sampled_token == END_TOKEN: break
        used_tokens[sampled_token] = used_tokens.get(sampled_token, 0) + 1
        if used_tokens[sampled_token] > 3: break
        if len(decoded_sentence) > 0 and sampled_token == decoded_sentence[-1]: break
        decoded_sentence.append(sampled_token)
        if len(decoded_sentence) >= 30: break
        target_seq = np.zeros((1, 1), dtype="int32")
        target_seq[0, 0] = sampled_token_index
        states_value = [hidden_state, cell_state]
    return " ".join(decoded_sentence)



class Chatbot:
    negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "gb", "stop")

    def start_chat(self):
        user_response = (input("Hi, I'm a recipe chatbot. Feel free to ask me any recipe questions.\n"))
        if user_response.lower() in self.negative_responses:
            print("Okay, have a great day!")
        self.chat(user_response)
    def chat(self, reply):
        while not self.make_exit(reply):
            response = self.generate_response(reply)
            print(response)
            reply = input()
    def string_to_matrix(self, user_input):
        tokens = re.findall(token_pattern, clean_text(user_input))
        user_input_matrix = np.zeros((1, max_encoder_seq_length), dtype="int32")
        for timestep, token in enumerate(tokens):
            if timestep < max_encoder_seq_length and token in input_features_dict:
                user_input_matrix[0, timestep] = input_features_dict[token]
        return user_input_matrix
    def clean_chatbot_response(self, chatbot_response):
        chatbot_response = chatbot_response.replace(START_TOKEN, "")
        chatbot_response = chatbot_response.replace(END_TOKEN, "")
        chatbot_response = chatbot_response.strip()
        chatbot_response = re.sub(r"\s+([,.?!:/()'\-])", r"\1", chatbot_response)
        chatbot_response = re.sub(r"\b(\d+)\s*\.\s*0\b", r"\1", chatbot_response)
        chatbot_response = chatbot_response.replace("pt0s", "unknown time")
        chatbot_response = re.sub(r"\s+", " ", chatbot_response).strip()
        chatbot_response = remove_repeated_words(chatbot_response)
        return chatbot_response
    def generate_response(self, user_input):
        conversion_response = handle_conversion(user_input)
        if conversion_response is not None:
            return conversion_response
        recipe_response = handle_recipe_lookup(user_input)
        if recipe_response is not None:
            return recipe_response
        input_matrix = self.string_to_matrix(user_input)
        chatbot_response = decode_response(input_matrix)
        chatbot_response = self.clean_chatbot_response(chatbot_response)
        if chatbot_response == "":
            chatbot_response = "Sorry, I didn't understand."
        return chatbot_response
    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply.lower():
                print("Okay, have a great day!")
                return True
        return False



#############################
# ------- EVALUATION -------
#############################

def clean_target_for_eval(text):
    text = text.replace(START_TOKEN, "")
    text = text.replace(END_TOKEN, "")
    text = text.strip()
    text = re.sub(f"\s([,.?!:/()'\-])", r"\1", text)
    text = re.sub(r"\b(\d+)\s*\.\s*0\b", r"\1", text)
    text = text.replace("pt0s", "unknown time")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def evaluate_rouge(test_pairs, max_examples=100):
    bot = Chatbot()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_f = []
    rouge2_f = []
    rougeL_f = []
    rows = []

    eval_pairs = test_pairs[:max_examples]

    for input_text, target_text in eval_pairs:
        reference = clean_target_for_eval(target_text)
        prediction = bot.generate_response(input_text)
        scores = scorer.score(reference, prediction)
        rouge1_f.append(scores["rouge1"].fmeasure)
        rouge2_f.append(scores["rouge2"].fmeasure)
        rougeL_f.append(scores["rougeL"].fmeasure)
        rows.append(
            {
                "input_text": input_text,
                "reference": reference,
                "prediction": prediction,
                "rouge1_f": scores["rouge1"].fmeasure,
                "rouge2_f": scores["rouge2"].fmeasure,
                "rougeL_f": scores["rougeL"].fmeasure,
            }
        )
    results = {
        "ROUGE-1 F1": float(np.mean(rouge1_f)),
        "ROUGE-2 F1": float(np.mean(rouge2_f)),
        "ROUGE-L F1": float(np.mean(rougeL_f)),
    }
    return results, pd.DataFrame(rows)

results, results_df = evaluate_rouge(test_pairs, max_examples=100)
print("\nROUGE Evaluation Results")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

print("\nSample Predictions:")
print(results_df[["input_text", "reference", "prediction", "rougeL_f"]].head(10))
print("\n")

bot = Chatbot()
bot.start_chat()


