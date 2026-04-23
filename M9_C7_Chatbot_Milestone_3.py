from google.colab import drive
drive.mount('/content/drive')

import json
import re
import numpy as np
import tensorflow as keras

from keras.layers import Input, LSTM, Dense, Attention, Concatenate, Embedding
from keras.models import Model


SQUAD_COOKING_JSON = "/content/drive/MyDrive/Colab_Notebooks/data/squad_cooking_transformed.json"



MAX_ROWS = 10000
dimensionality = 128
embedding_dim = 128
batch_size = 32
epochs = 50

START_TOKEN = "starttoken"
END_TOKEN = "endtoken"



def clean_text(text):
    text = str(text).lower().strip()
    text = text.replace("Â°", " degrees ")
    text = text.replace("°", " degrees")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9,.?!:/()'\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def shorten_context(text, max_words=120):
    words = clean_text(text).split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def load_squad_cooking(path, max_rows=None, include_context=True):
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
            input_text = context + " " + question
        else:
            input_text = question


        target_text = START_TOKEN + " " + answer_text + " " + END_TOKEN

        pairs.append((input_text, target_text))

    return pairs

pairs = load_squad_cooking(
    SQUAD_COOKING_JSON,
    max_rows=MAX_ROWS,
    include_context=True
)

print("Number of pairs: ", len(pairs))
print("\nSample pairs: ")
print("INPUT : ", pairs[0][0])
print("OUTPUT : ", pairs[0][1])




input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()

token_pattern = r"[\w']+|[^\s\w]"

for input_text, target_text in pairs:
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
    initial_state=encoder_states
)


# Add attention
attention_layer = Attention(name="attention_layer")
attention_output = attention_layer([decoder_outputs, encoder_outputs])

decoder_concat = Concatenate(axis=-1, name="concat_layer")(
    [decoder_outputs, attention_output]
)


decoder_dense = Dense(num_decoder_tokens, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_concat)

training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
training_model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

training_model.summary()

history = training_model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)



# Inference Encoder Model
encoder_model = Model(
    encoder_inputs,
    [encoder_outputs, state_hidden, state_cell]
)


# Inference Decoder Model
decoder_input_single = Input(shape=(None,), name="decoder_input_single")
encoder_output_input = Input(shape=(None, dimensionality), name="encoder_output_input")
decoder_state_input_hidden = Input(shape=(dimensionality,), name="decoder_state_input_hidden")
decoder_state_input_cell = Input(shape=(dimensionality,), name="decoder_state_input_cell")

decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

decoder_embedding_inf = decoder_embedding_layer(decoder_input_single)
decoder_outputs_inf, state_hidden_inf, state_cell_inf = decoder_lstm(
    decoder_embedding_inf,
    initial_state=decoder_states_inputs
)

attention_output_inf = attention_layer([decoder_outputs_inf, encoder_output_input])
decoder_concat_inf = Concatenate(axis=-1, name="concat_layer_inf")(
    [decoder_outputs_inf, attention_output_inf]
)
decoder_outputs_inf = decoder_dense(decoder_concat_inf)

decoder_model = Model(
    [decoder_input_single, encoder_output_input] + decoder_states_inputs,
    [decoder_outputs_inf, state_hidden_inf, state_cell_inf]
)


def sample_with_temperature(preds, temperature=0.4):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def decode_response(test_input):
    encoder_outs, state_h, state_c = encoder_model.predict(test_input, verbose=0)
    states_value = [state_h, state_c]

    target_seq = np.zeros((1, 1), dtype="int32")
    target_seq[0, 0] = target_features_dict[START_TOKEN]

    decoded_sentence = []
    stop_condition = False

    while not stop_condition:
        output_tokens, hidden_state, cell_state = decoder_model.predict(
            [target_seq, encoder_outs] + states_value,
            verbose=0
        )

        sampled_token_index = sample_with_temperature(output_tokens[0, -1, :], temperature=0.4)
        sampled_token = reverse_target_features_dict.get(sampled_token_index, "")

        if sampled_token == "":
            break

        decoded_sentence.append(sampled_token)

        if sampled_token == END_TOKEN or len(decoded_sentence) > 20:
            stop_condition = True

        target_seq = np.zeros((1, 1), dtype="int32")
        target_seq[0, 0] = sampled_token_index

        states_value = [hidden_state, cell_state]

    return " ".join(decoded_sentence)




class ChatBot:
    negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")


    def start_chat(self):
        user_response = input("Hi, I'm a recipe chatbot. Do you have any recipe questions?\n")
        if user_response.lower() in self.negative_responses:
            print("Ok, have a great day!")
            return
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

    def generate_response(self, user_input):
        input_matrix = self.string_to_matrix(user_input)
        chatbot_response = decode_response(input_matrix)

        chatbot_response = chatbot_response.replace(START_TOKEN, "")
        chatbot_response = chatbot_response.replace(END_TOKEN, "")
        chatbot_response = chatbot_response.strip()

        chatbot_response = re.sub(r"\s+([,.?!:/()'\-])", r"\1", chatbot_response)

        if chatbot_response == "":
            chatbot_response = "Sorry, I didn't understand that."

        return chatbot_response

    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply.lower():
                print("Ok, have a great day!")
                return True
        return False


def test_examples():
    examples = [
        "How long should the mixture be allowed to gel in the freezer before adding the sour cream?",
        "What quantity of flour should be used in the recipe?",
        "How many bananas are needed?",
        "At what temperature should the cake be baked?",
        "How many eggs are needed?",
        "How many grams of flour are needed?"
    ]

    bot = ChatBot()

    for example in examples:
        print("\nQuestion: ", example)
        print("Answer: ", bot.generate_response(example))
        print("-" * 50)

test_examples()


chatbot = ChatBot()
chatbot.start_chat()





