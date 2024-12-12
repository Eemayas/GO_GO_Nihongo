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


num_example = 300000


def create_lang_list(num_example):

    lines = io.open("datas/raw.txt", encoding="UTF-8").read().strip().split("\n")

    word_pairs = [[w for w in l.split("\t")] for l in lines[:num_example]]

    return zip(*word_pairs)


en, ja = create_lang_list(num_example)


ja_sentence = list()
for i in ja:
    ja_sentence.append(i.replace(" ", ""))

ja_text = list()

sp = spm.SentencePieceProcessor(
    model_file="./Pretrained tokenizer/spm.en.nopretok.model"
)

for text in ja_sentence:
    ja_text.append(" ".join(sp.EncodeAsPieces(text)).replace("▁", "").strip())


def moji(text):
    return mojimoji.zen_to_han(text)

print("カナダ →", moji("カナダ"))

def english_unicode_to_ascii(text):
    return "".join(
        ascii_text
        for ascii_text in unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


def japanese_unicode_to_ascii(text):
    return "".join(ascii_text for ascii_text in unicodedata.normalize("NFKD", text))

japanese_unicode_to_ascii("こんにちは。 今日は"), english_unicode_to_ascii(
    "Hello world é "
)


def replace_special_character_to_space_en(text):
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = text.strip()
    return text


def replace_special_character_to_space(text):
    text = re.sub(r"([?!。、¿])", r" \1", text)
    pattern = r"[^\u3041-\u309F\u30A1-\u30FF\uFF66-\uFF9F\u4E00-\u9FD0\u309B\u3099\uFF9E\u309C\u309A\uFF9F?!\s、。.,0-9]+"
    text = re.sub(pattern, "", text).rstrip().strip()
    text = re.sub(r'[" "]+', " ", text)
    text = text.replace("・", "")

    text = text.lower()

    return text


replace_special_character_to_space("hello")


def normalize_english(english_text, japanese_text):

    input_value = []
    target_value = []

    for en_text, ja_text in zip(english_text, japanese_text):

        en_text = english_unicode_to_ascii(en_text)
        en_text = replace_special_character_to_space_en(en_text)

        en_text = "start_ " + en_text + " _end"

        input_value.append(en_text)

        ja_text = japanese_unicode_to_ascii(ja_text)
        ja_text = replace_special_character_to_space(ja_text)
        ja_text = moji(ja_text)

        ja_text = "start_ " + ja_text + " _end"

        target_value.append(ja_text)

    return input_value, target_value


target_value, input_value = normalize_english(en, ja_text)


x = pd.Series(input_value)
y = pd.Series(target_value)

pd.DataFrame({"input": x, "target": y})


def tokenize(lang):

    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=" ")

    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")
    return tensor, lang_tokenizer


tokenize(["this place is good", "こんにちは 今日は いい天気 。", "today is so cold"])


def create_dataset(en, ja):

    input_tensor, input_lang_tokenize = tokenize(en)
    target_tensor, target_lang_tokenize = tokenize(ja)

    return input_tensor, target_tensor, input_lang_tokenize, target_lang_tokenize


input_tensor, target_tensor, input_lang_tokenize, target_lang_tokenize = create_dataset(
    x, y
)


def max_length(input_tensor, target_tensor):

    english_len = [len(i) for i in input_tensor]

    japanese_len = [len(i) for i in target_tensor]

    print("english length:", max(english_len))
    print("japanese length:", max(japanese_len))
    max_len_input = max(english_len)
    max_len_target = max(japanese_len)

    return max_len_input, max_len_target


max_length_input, max_length_target = max_length(input_tensor, target_tensor)


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


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size, dropout_rate):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.dropout = Dropout(dropout_rate)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.first_lstm = tf.keras.layers.LSTM(
            self.enc_units,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
        )

        self.final_lstm = tf.keras.layers.LSTM(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.first_lstm(x, initial_state=hidden)
        output, state_h, state_c = self.final_lstm(x)
        state = [state_h, state_c]

        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units)), tf.zeros(
            (self.batch_size, self.enc_units)
        )


class Attention(tf.keras.models.Model):

    def __init__(self, units: int, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.units = units

        self.q_dense_layer = Dense(units, use_bias=False, name="q_dense_layer")
        self.k_dense_layer = Dense(units, use_bias=False, name="k_dense_layer")
        self.v_dense_layer = Dense(units, use_bias=False, name="v_dense_layer")
        self.output_dense_layer = Dense(
            units, use_bias=False, name="output_dense_layer"
        )

    def call(self, input, memory):

        q = self.q_dense_layer(input)
        k = self.k_dense_layer(memory)
        v = self.v_dense_layer(memory)

        depth = self.units // 2
        q *= depth**-0.5

        logit = tf.matmul(q, k, transpose_b=True)

        attention_weight = tf.nn.softmax(logit)

        attention_output = tf.matmul(attention_weight, v)
        return self.output_dense_layer(attention_output)


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size, dropout_rate):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dropout = Dropout(dropout_rate)
        self.first_lstm = tf.keras.layers.LSTM(self.dec_units, return_sequences=True)
        self.final_lstm = tf.keras.layers.LSTM(
            self.dec_units, return_sequences=True, return_state=True
        )

        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = Attention(self.dec_units)

    def call(self, x, hidden, enc_output):
        x = self.embedding(x)
        x = self.dropout(x)

        x = self.first_lstm(x)
        output, state_h, state_c = self.final_lstm(x)
        state = [state_h, state_c]
        attention_weights = self.attention(output, enc_output)
        output = tf.concat([output, attention_weights], axis=-1)

        output = tf.reshape(output, (-1, output.shape[2]))

        output = self.fc(output)

        return output, state


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


BUFFER_SIZE = len(X_train)
BATCH_SIZE = 256

dropout_rate = 0.3

train_steps_per_epoch = len(X_train) // BATCH_SIZE
val_steps_per_epoch = len(X_val) // BATCH_SIZE
print("train step %d" % train_steps_per_epoch)
embedding_dim = 300
units = 128

vocab_inp_size = len(input_lang_tokenize.word_index) + 1
print("Total unique words in the input: %s" % len(input_lang_tokenize.word_index))
print("Total unique words in the target: %s" % len(target_lang_tokenize.word_index))
vocab_tar_size = len(target_lang_tokenize.word_index) + 1


train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
)


val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(
    BATCH_SIZE, drop_remainder=True
)

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, dropout_rate)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, dropout_rate)


checkpoint_dir = "./train_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims(
            [target_lang_tokenize.word_index["start_"]] * BATCH_SIZE, 1
        )

        for t in range(1, targ.shape[1]):

            predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


from tqdm import tqdm


latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print(f"Restored from {latest_checkpoint}")
else:
    print("Starting training from scratch.")

EPOCHS = 15

for epoch in range(EPOCHS):

    enc_hidden = encoder.initialize_hidden_state()
    train_loss = 0
    val_loss = 0

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for batch, (inp, targ) in enumerate(
        tqdm(train_dataset.take(train_steps_per_epoch), desc="Training", leave=False)
    ):
        train_batch_loss = train_step(inp, targ, enc_hidden)
        train_loss += train_batch_loss

    for batch, (val_inp, val_tar) in enumerate(
        tqdm(val_dataset.take(val_steps_per_epoch), desc="Validation", leave=False)
    ):
        val_batch_loss = train_step(val_inp, val_tar, enc_hidden)
        val_loss += val_batch_loss

    if (epoch + 1) % 1 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print(f"Epoch {epoch + 1} Train Loss: {train_loss / train_steps_per_epoch:.4f}")
    print(f"Epoch {epoch + 1} Validation Loss: {val_loss / val_steps_per_epoch:.4f}")


checkpoint.restore("train_checkpoints/ckpt-29")


encoder = checkpoint.encoder
decoder = checkpoint.decoder


def predict(sentence):
    inputs = tf.convert_to_tensor(sentence)
    result = ""
    inputs = tf.expand_dims(inputs, axis=0)
    hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
    enc_out, state = encoder(inputs, hidden)
    hidden_state = state
    dec_input = tf.expand_dims([target_lang_tokenize.word_index["start_"]], 0)
    for t in range(max_length_target):
        predictions, hidden_state = decoder(dec_input, hidden_state, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += target_lang_tokenize.index_word[predicted_id] + " "
        if (
            target_lang_tokenize.index_word[predicted_id] == "_end"
            or len(result) > max_length_target
        ):
            return result

        dec_input = tf.expand_dims([predicted_id], 0)
    return result


def create_reference(lang, tensor):
    all_sentence_list = []

    for word_list in tensor:
        sentence_list = []

        for t in word_list:
            if not t == 0:

                sentence_list.append(lang.index_word[t])
        all_sentence_list.append(sentence_list)
    return all_sentence_list


reference = create_reference(target_lang_tokenize, Y_test.tolist()[:30])

from tqdm import tqdm

hypothesis = []
for i in tqdm(X_test[:30]):
    hypothesis.append(predict(i))

for ref, hyp in zip(reference, hypothesis):
    ref = " ".join(ref[1:-1])
    ss = {"ref": ref, "hyp": hyp}

    print(ss)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

score = 0
smoothie = SmoothingFunction().method2
for i in range(len(reference)):
    score += sentence_bleu(
        [reference[i][1:-1]],
        hypothesis[i][:-5].strip().split(),
        smoothing_function=smoothie,
    )

score /= len(reference)
print("The bleu score is: " + str(score))


def preprocess_sentence(en_text):

    ja_text = japanese_unicode_to_ascii(en_text)
    ja_text = replace_special_character_to_space(ja_text)
    ja_text = moji(ja_text)

    ja_text = "start_ " + ja_text + " _end"
    return ja_text


def evaluate(sentence):

    attention_plot = np.zeros((max_length_target, max_length_input))
    sentence = preprocess_sentence(sentence).strip()
    inputs = [
        input_lang_tokenize.word_index[i]
        for i in sentence.split(" ")
        if i in input_lang_tokenize.word_index
    ]

    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_input, padding="post"
    )

    inputs = tf.convert_to_tensor(inputs)
    result = ""
    hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
    enc_out, state = encoder(inputs, hidden)
    hidden_state = state
    dec_input = tf.expand_dims([target_lang_tokenize.word_index["start_"]], 0)
    for t in range(max_length_target):
        predictions, hidden_state = decoder(dec_input, hidden_state, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()

        result += target_lang_tokenize.index_word[predicted_id] + " "
        if (
            target_lang_tokenize.index_word[predicted_id] == "_end"
            or len(result) > max_length_target
        ):
            return result, sentence

        dec_input = tf.expand_dims([predicted_id], 0)
    return (
        result,
        sentence,
    )


from nltk.translate.bleu_score import sentence_bleu


def result(sentence):
    result, sentence = evaluate(sentence)

    return result, sentence


result, sentence = result("彼は犬を飼っています")
print("Input: %s" % (sentence))
print("Predicted translation: {}".format(result))


result, sentence = result("これは何ですか!?")
print("Input: %s" % (sentence))
print("Predicted translation: {}".format(result))


result, sentence = result("実は、私がやりました。")
print("Input: %s" % (sentence))
print("Predicted translation: {}".format(result))


result, sentence = result("彼女を愛しています。")
print("Input: %s" % (sentence))
print("Predicted translation: {}".format(result))
