from google.colab import drive
drive.mount('/content/drive')


import json
import re
import numpy as np
import tensorflow as keras

from keras.layers import Input, LSTM, Dense
from keras.models import Model

SQUAD_COOKING_JSON = "/content/drive/MyDrive/Colab_Notebooks/data/squad_cooking_transformed.json"


MAX_ROWS = 2000
dimensionality = 256
batch_size = 10
epochs = 30



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
            input_text = context + " " + question
        else:
            input_text = question


        target_text = "<START> " + answer_text + " <END>"

        pairs.append((input_text, target_text))

    return pairs

pairs = load_squad_cooking(SQUAD_COOKING_JSON, max_rows=MAX_ROWS, include_context=False)
print("Number of pairs: ", len(pairs))
print("\nSample pairs: ")
print("INPUT : ", pairs[0][0])
print("OUTPUT : ", pairs[0][1])




input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()

for input_text, target_text in pairs:
    input_docs.append(input_text)
    target_docs.append(target_text)

    input_words = re.findall(r"[\w']+|[^\s\w]", input_text)
    target_words = re.findall(r"[\w']+|[^\s\w]", target_text)

    for token in input_words:
        input_tokens.add(token)

    for token in target_words:
        target_tokens.add(token)


input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", txt)) for txt in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", txt)) for txt in target_docs])



print("num_encoder_tokens: ", num_encoder_tokens)
print("num_decoder_tokens: ", num_decoder_tokens)
print("max_encoder_seq_length: ", max_encoder_seq_length)
print("max_decoder_seq_length: ", max_decoder_seq_length)


input_features_dict = dict([(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict([(token, i) for i, token in enumerate(target_tokens)])
reverse_target_features_dict = dict((i, token) for token, i in target_features_dict.items())


encoder_input_data = np.zeros((len(input_docs), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
decoder_input_data = np.zeros((len(input_docs), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_target_data = np.zeros((len(input_docs), max_decoder_seq_length, num_decoder_tokens), dtype="float32")


for line, (input_text, target_text) in enumerate(zip(input_docs, target_docs)):
    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_text)):
        encoder_input_data[line, timestep, input_features_dict[token]] = 1.

    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", target_text)):
        decoder_input_data[line, timestep, target_features_dict[token]] = 1.

        if timestep > 0:
            decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(dimensionality, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
training_model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

training_model.summary()

history = training_model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)

training_model.save("training_model.h5")


from keras.models import load_model
training_model = load_model('training_model.h5')

encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

latent_dim = dimensionality

decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)



def decode_response(test_input):
    states_value = encoder_model.predict(test_input, verbose=0)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_features_dict["<START>"]] = 1.
    decoded_sentence = ""

    stop_condition = False
    while not stop_condition:
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if (sampled_token == "<END>" or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [hidden_state, cell_state]

    return decoded_sentence






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
            reply = (self.generate_response(reply) + "\n")


    def string_to_matrix(self, user_input):
        tokens = re.findall(r"[\w']+|[^\s\w]", clean_text(user_input))
        user_input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
        for timestep, token in enumerate(tokens):
            if timestep < max_encoder_seq_length and token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.

        return user_input_matrix

    def generate_response(self, user_input):
        input_matrix = self.string_to_matrix(user_input)
        chatbot_response = decode_response(input_matrix)

        chatbot_response = chatbot_response.replace("<START>", "")
        chatbot_response = chatbot_response.replace("<END>", "")
        chatbot_response = chatbot_response.strip()

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





