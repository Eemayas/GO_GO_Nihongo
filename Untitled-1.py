import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Embedding,
    Dropout,
    Input,
    dot,
    Activation,
    Concatenate,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers
from tensorflow.keras import initializers, regularizers, constraints

from sklearn.model_selection import train_test_split


import mojimoji
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import io

import nltk
import unicodedata
import sentencepiece as spm


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
print(tf.__version__)

tf.test.is_gpu_available()


from janome.tokenizer import Tokenizer
import mojimoji


tokenizer = Tokenizer()


def preprocess_japanese_text(text):

    text = mojimoji.zen_to_han(text)

    tokens = tokenizer.tokenize(text, wakati=True)

    preprocessed_text = " ".join(tokens)

    preprocessed_text = preprocessed_text.strip()

    return preprocessed_text


text = "今日は今日はとても良い日です。"
processed_text = preprocess_japanese_text(text)
print(processed_text)


import pandas as pd

df = pd.read_csv("./datas/Aniki.csv", index_col=False)

df = df[:1000]
df


import re
import pandas as pd
import tqdm


emoji_list_datas_path = "./datas//Emoji Sheets - Emoji Only.csv"
emoji_df = pd.read_csv(emoji_list_datas_path)


emoji_list = emoji_df["Emoji_List"].tolist()


pattern = "["


for cp in emoji_list:
    pattern += f"\\U{cp[1:]:0>8}"


pattern += "]"


emoji_pattern = re.compile(pattern, re.UNICODE)


def remove_emojis(text):
    if not isinstance(text, str):
        return text

    emoji_pattern = re.compile(pattern, re.UNICODE)

    text_no_emojis = emoji_pattern.sub(r"", text)
    return text_no_emojis


from tqdm import tqdm

tqdm.pandas(desc="Removing emojis from English text")
df["english"] = df["english"].progress_apply(remove_emojis)

tqdm.pandas(desc="Removing emojis from Japanese text")
df["japanese"] = df["japanese"].progress_apply(remove_emojis)


import mojimoji


def zen_to_han(text):
    return mojimoji.zen_to_han(text)


import unicodedata


def english_unicode_to_ascii(text):
    return "".join(
        ascii_text
        for ascii_text in unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


def japanese_unicode_to_ascii(text):
    return " ".join(ascii_text for ascii_text in unicodedata.normalize("NFKD", text))


japanese_unicode_to_ascii("こんにちは。 今日は"), english_unicode_to_ascii(
    "Hello world é "
)


import re


def clean_text(text):

    allowed_pattern = r"[^a-zA-Z\u4E00-\u9FFF\u3040-\u30FF\s]"

    cleaned_text = re.sub(allowed_pattern, " ", text)

    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    cleaned_text = cleaned_text.lower()

    return cleaned_text


text = "こんにちは、世界! This is an example sentence."
print(clean_text(text))

from janome.tokenizer import Tokenizer
import mojimoji


tokenizer = Tokenizer()


def preprocess_japanese_text(text):

    text = mojimoji.zen_to_han(text)

    text = clean_text(text)

    tokens = tokenizer.tokenize(text, wakati=True)

    preprocessed_text = " ".join(tokens)

    preprocessed_text = preprocessed_text.strip()

    return "start_ " + preprocessed_text + " _end"


text = "猫と犬がけんかしている。"
processed_text = preprocess_japanese_text(text)
print(processed_text)


from tensorflow.keras.preprocessing.text import Tokenizer


def preprocess_english_text(text):

    text = text.lower()
    text = english_unicode_to_ascii(text)
    text = clean_text(text)

    preprocessed_text = f"start_ {text} _end"

    return preprocessed_text


text = "Hello, how are you?"
token = preprocess_english_text(text)
token


from tqdm import tqdm


tqdm.pandas(desc="Preprocessing English text")
df["english_preprocessed"] = df["english"].progress_apply(preprocess_english_text)


tqdm.pandas(desc="Preprocessing Japanese text")
df["japanese_preprocessed"] = df["japanese"].progress_apply(preprocess_japanese_text)


def tokenize(lang):

    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=" ")

    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")
    return tensor, lang_tokenizer


tokenize(["this place is good", "こんにちは 今日は いい天気 。", "today is so cold"])


def create_dataset(ja, en):

    input_tensor, input_lang_tokenize = tokenize(ja)
    target_tensor, target_lang_tokenize = tokenize(en)

    return input_tensor, target_tensor, input_lang_tokenize, target_lang_tokenize


input_tensor, target_tensor, input_lang_tokenize, target_lang_tokenize = create_dataset(
    df["japanese_preprocessed"], df["english_preprocessed"]
)
(len(input_tensor), len(target_tensor))


import os
import pickle


kaggle = True
input_tokenizer_saving_path = (
    "/kaggle/working/output/input_tokenizer.pkl"
    if kaggle
    else "./output/input_tokenizer.pkl"
)
target_tokenizer_saving_path = (
    "/kaggle/working/output/target_tokenizer.pkl"
    if kaggle
    else "./output/target_tokenizer.pkl"
)


os.makedirs(os.path.dirname(input_tokenizer_saving_path), exist_ok=True)


with open(input_tokenizer_saving_path, "wb") as file:
    pickle.dump(input_lang_tokenize, file)

with open(target_tokenizer_saving_path, "wb") as file:
    pickle.dump(target_lang_tokenize, file)

print("Tokenizer saved as tokenizer.pkl")


with open(input_tokenizer_saving_path, "rb") as file:
    input_lang_tokenize_test = pickle.load(file)

with open(target_tokenizer_saving_path, "rb") as file:
    target_lang_tokenize_test = pickle.load(file)
print("Tokenizer loaded successfully")


preprocessed_input_jap = preprocess_japanese_text("こんにちは 今日は いい天気 。")
preprocessed_input_eng = preprocess_english_text("today is so cold")
input_sequence_jap = input_lang_tokenize.texts_to_sequences([preprocessed_input_jap])
input_sequence_eng = target_lang_tokenize.texts_to_sequences([preprocessed_input_eng])
(input_sequence_jap, input_sequence_eng)


preprocessed_input_jap = preprocess_japanese_text("こんにちは 今日は いい天気 。")
preprocessed_input_eng = preprocess_english_text("today is so cold")
input_sequence_jap = input_lang_tokenize_test.texts_to_sequences(
    [preprocessed_input_jap]
)
input_sequence_eng = target_lang_tokenize_test.texts_to_sequences(
    [preprocessed_input_eng]
)
(input_sequence_jap, input_sequence_eng)


def max_length(input_tensor, target_tensor):

    english_len = [len(i) for i in target_tensor]

    japanese_len = [len(i) for i in input_tensor]

    print("english length:", max(english_len))
    print("japanese length:", max(japanese_len))

    max_len_input = max(japanese_len)
    max_len_target = max(english_len)

    return max_len_input, max_len_target


max_length_input, max_length_target = max_length(input_tensor, target_tensor)
(max_length_input, max_length_target)


X_train, X_test, Y_train, Y_test = train_test_split(
    input_tensor, target_tensor, test_size=0.2, shuffle=True
)

X_test, X_val, Y_test, Y_val = train_test_split(
    X_test, Y_test, test_size=0.5, shuffle=True
)


print(len(X_train), len(Y_train), len(X_test), len(Y_test), len(X_val), len(Y_val))


def convert(lang, tensor):
    for t in tensor:
        if t != 0:

            print("%d----->%s" % (t, lang.index_word[t]))


print("input lang: index to word mapping")
convert(input_lang_tokenize, X_train[10])
print("output lang: index to word mapping")
convert(target_lang_tokenize, Y_train[10])


import tensorflow as tf


BUFFER_SIZE = len(X_train)
BATCH_SIZE = 64
dropout_rate = 0.3
embedding_dim = 300
units = 512


train_steps_per_epoch = len(X_train) // BATCH_SIZE
val_steps_per_epoch = len(X_val) // BATCH_SIZE

print(f"Train steps per epoch: {train_steps_per_epoch}")
print(f"Validation steps per epoch: {val_steps_per_epoch}")


vocab_inp_size = len(input_lang_tokenize.word_index) + 1
vocab_tar_size = len(target_lang_tokenize.word_index) + 1

print(f"Total unique words in the input: {len(input_lang_tokenize.word_index)}")
print(f"Total unique words in the target: {len(target_lang_tokenize.word_index)}")
print(f"Vocabulary input size: {vocab_inp_size}")
print(f"Vocabulary target size: {vocab_tar_size}")


train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(
    BATCH_SIZE, drop_remainder=True
)

print(f"File_name: config_B{BATCH_SIZE}_D{dropout_rate}_E{embedding_dim}_U{units}")


import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
class Seq2SeqModel(Model):
    def __init__(
        self,
        vocab_inp_size,
        vocab_tar_size,
        embedding_dim,
        units,
        batch_size,
        dropout_rate,
        **kwargs,
    ):
        super(Seq2SeqModel, self).__init__(**kwargs)
        self.vocab_inp_size = vocab_inp_size
        self.vocab_tar_size = vocab_tar_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate

        self.embedding_enc = Embedding(vocab_inp_size, embedding_dim)
        self.dropout_enc = Dropout(dropout_rate)
        self.first_lstm_enc = LSTM(
            units, return_sequences=True, recurrent_initializer="glorot_uniform"
        )
        self.final_lstm_enc = LSTM(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

        self.q_dense_layer = Dense(units, use_bias=False, name="q_dense_layer")
        self.k_dense_layer = Dense(units, use_bias=False, name="k_dense_layer")
        self.v_dense_layer = Dense(units, use_bias=False, name="v_dense_layer")
        self.output_dense_layer = Dense(
            units, use_bias=False, name="output_dense_layer"
        )

        self.embedding_dec = Embedding(vocab_tar_size, embedding_dim)
        self.dropout_dec = Dropout(dropout_rate)
        self.first_lstm_dec = LSTM(units, return_sequences=True)
        self.final_lstm_dec = LSTM(units, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_tar_size)

    def initialize_hidden_state(self):
        return (
            tf.zeros((self.batch_size, self.units)),
            tf.zeros((self.batch_size, self.units)),
        )

    def call_attention(self, query, memory, mask=None):
        q = self.q_dense_layer(query)
        print("Shape of q (query):", q.shape)

        k = self.k_dense_layer(memory)
        print("Shape of k (key):", k.shape)

        v = self.v_dense_layer(memory)
        print("Shape of v (value):", v.shape)

        depth = self.units**0.5
        q *= depth**-0.5

        logits = tf.matmul(q, k, transpose_b=True)
        print("Shape of logits (query x key):", logits.shape)

        if mask is not None:
            logits += mask * -1e9

        attention_weights = tf.nn.softmax(logits, axis=-1)
        print("Shape of attention_weights:", attention_weights.shape)

        attention_output = tf.matmul(attention_weights, v)
        print("Shape of attention_output:", attention_output.shape)

        return self.output_dense_layer(attention_output)

    def call_encoder(self, x, hidden):
        x = self.embedding_enc(x)
        print("Shape after embedding_enc (encoder input):", x.shape)

        x = self.dropout_enc(x)
        print("Shape after dropout_enc:", x.shape)

        x = self.first_lstm_enc(x, initial_state=hidden)
        print("Shape after first LSTM layer (encoder):", x.shape)

        output, state_h, state_c = self.final_lstm_enc(x)
        print("Shape of final LSTM output (encoder):", output.shape)
        print("Shape of final LSTM state_h (encoder):", state_h.shape)
        print("Shape of final LSTM state_c (encoder):", state_c.shape)

        state = [state_h, state_c]
        return output, state

    def call_decoder(self, x, hidden, enc_output, mask=None):
        x = self.embedding_dec(x)
        print("Shape after embedding_dec (decoder input):", x.shape)

        x = self.dropout_dec(x)
        print("Shape after dropout_dec:", x.shape)

        x = self.first_lstm_dec(x)
        print("Shape after first LSTM layer (decoder):", x.shape)

        output, state_h, state_c = self.final_lstm_dec(x)
        print("Shape of final LSTM output (decoder):", output.shape)

        attention_weights = self.call_attention(output, enc_output, mask=mask)
        print(
            "Shape of attention weights from call_attention:", attention_weights.shape
        )

        output = tf.concat([output, attention_weights], axis=-1)
        print("Shape after concatenation (decoder):", output.shape)

        output = tf.reshape(output, (-1, output.shape[2]))
        print("Shape after reshaping (decoder):", output.shape)

        output = self.fc(output)
        print("Shape after final dense layer (decoder):", output.shape)

        state = [state_h, state_c]
        return output, state

    def call(self, enc_input, dec_input, enc_hidden, mask=None):
        enc_output, enc_state = self.call_encoder(enc_input, enc_hidden)
        dec_output, dec_state = self.call_decoder(
            dec_input, enc_state, enc_output, mask
        )
        return dec_output, dec_state

    def get_build_config(self):
        """Returns the configuration needed to build the model."""
        return {
            "vocab_inp_size": self.vocab_inp_size,
            "vocab_tar_size": self.vocab_tar_size,
            "embedding_dim": self.embedding_dim,
            "units": self.units,
            "batch_size": self.batch_size,
            "dropout_rate": self.dropout_rate,
        }

    @classmethod
    def build_from_config(cls, config):
        """Reconstructs the model from its configuration."""
        return cls(**config)

    def get_config(self):
        config = super(Seq2SeqModel, self).get_config()
        config.update(
            {
                "vocab_inp_size": self.vocab_inp_size,
                "vocab_tar_size": self.vocab_tar_size,
                "embedding_dim": self.embedding_dim,
                "units": self.units,
                "batch_size": self.batch_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_loss(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_mean(loss)

    def calculate_accuracy(self, real, pred):
        real = tf.cast(real, dtype=tf.int32)
        pred_tokens = tf.argmax(pred, axis=-1, output_type=tf.int32)
        correct = tf.equal(real, pred_tokens)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        correct = tf.math.logical_and(correct, mask)
        return tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

    @tf.function
    def train_step(self, data):
        inp, targ = data
        loss = 0
        total_accuracy = 0

        enc_hidden = self.initialize_hidden_state()

        with tf.GradientTape() as tape:

            enc_output, enc_hidden = self.call_encoder(inp, enc_hidden)
            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([targ[:, 0]] * self.batch_size, axis=1)

            for t in range(1, targ.shape[1]):

                predictions, dec_hidden = self.call_decoder(
                    dec_input, dec_hidden, enc_output
                )

                loss += self.compute_loss(targ[:, t], predictions)

                total_accuracy += self.calculate_accuracy(targ[:, t], predictions)

                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = loss / int(targ.shape[1])
        batch_accuracy = total_accuracy / int(targ.shape[1])

        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {"loss": batch_loss, "accuracy": batch_accuracy}


optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.9, epsilon=1e-04, decay=1e-06
)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def calculate_accuracy(real, pred):
    real = tf.cast(real, dtype=tf.int32)
    pred_tokens = tf.argmax(pred, axis=-1, output_type=tf.int32)

    correct = tf.equal(real, pred_tokens)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    correct = tf.math.logical_and(correct, mask)

    accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
    return accuracy


@tf.function
def train_step_with_accuracy(inp, targ, enc_hidden):
    loss = 0
    total_accuracy = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = seq2seq_model.call_encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(
            [target_lang_tokenize.word_index["start_"]] * BATCH_SIZE, 1
        )

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden = seq2seq_model.call_decoder(
                dec_input, dec_hidden, enc_output
            )
            loss += loss_function(targ[:, t], predictions)

            batch_accuracy = calculate_accuracy(targ[:, t], predictions)
            total_accuracy += batch_accuracy

            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])
    batch_accuracy = total_accuracy / int(targ.shape[1])

    variables = seq2seq_model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss, batch_accuracy


seq2seq_model = Seq2SeqModel(
    vocab_inp_size, vocab_tar_size, embedding_dim, units, BATCH_SIZE, dropout_rate
)
seq2seq_model.compile(optimizer=optimizer)
