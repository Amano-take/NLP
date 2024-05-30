# %%
import gensim.downloader as gendl
corpus = gendl.load("text8")
corpus = list(corpus)

# %%
training_ratio = 0.9
total_size = 300
corpus = corpus[:total_size]
corpus = [word for sentence in corpus for word in sentence]
training_size = int(total_size * training_ratio)

def chunk_text(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# チャンクサイズを設定（例：100単語）
chunk_size = 15
# テキストをチャンクに分割
chunks = chunk_text(corpus, chunk_size)
# トレーニングデータを作成
train_data = []
for chunk in chunks:
    # 2文字単位のプレフィックスを作成
    prefixes = [word[:2] if len(word) >= 2 else word + " " for word in chunk]
    train_data.append((prefixes, chunk))

prefixes = [prefix for prefix, _ in train_data]
sentences = [sentence for _, sentence in train_data]
test_prefixes = prefixes[int(len(prefixes) * training_ratio):]
test_sentences = sentences[int(len(sentences) * training_ratio):]
prefixes = prefixes[:int(len(prefixes) * training_ratio)]
sentences = sentences[:int(len(sentences) * training_ratio)]


# %%

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
prefix_tokenizer = Tokenizer(char_level=False, filters='') 
prefix_tokenizer.fit_on_texts(prefixes)
prefix_vocab_size = len(prefix_tokenizer.word_index) + 1
print(prefix_vocab_size)
word_tokenizer = Tokenizer(char_level=False)
word_tokenizer.fit_on_texts(sentences)
word_vocab_size = len(word_tokenizer.word_index) + 1
print(word_vocab_size)

#%%
import numpy as np
prefix_sequences = prefix_tokenizer.texts_to_sequences(prefixes)
# パディング
max_prefix_len = max([len(seq) for seq in prefix_sequences])
prefix_sequences = pad_sequences(prefix_sequences, maxlen=max_prefix_len, padding='post')

# 出力文をトークン化
sentence_sequences = word_tokenizer.texts_to_sequences(sentences)
# パディング
max_sentence_len = max([len(seq) for seq in sentence_sequences])
sentence_sequences = pad_sequences(sentence_sequences, maxlen=max_sentence_len, padding='post')
X = np.array(prefix_sequences)
y = np.array(sentence_sequences)

# %%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed

# モデルの構築
model = Sequential()
model.add(Embedding(prefix_vocab_size, 32, input_length=max_prefix_len))
model.add(LSTM(64, return_sequences=True))
model.add(TimeDistributed(Dense(word_vocab_size, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# %%
from tensorflow.keras.callbacks import TensorBoard
import datetime
# ログディレクトリの設定
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

batch_size = 32

# データのバッチジェネレータを定義
def batch_generator(X, y, batch_size, total_chars):
    n_batches = len(X) // batch_size
    while True:  # データを無限に供給
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]
            y_batch = tf.keras.utils.to_categorical(y_batch, num_classes=total_chars)
            yield X_batch, y_batch

train_gen = batch_generator(X, y, batch_size, word_vocab_size)

# モデルのトレーニング
steps_per_epoch = len(X) // batch_size
# %%
len(X)
#%%
model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=50, verbose=1, callbacks=[tensorboard_callback])

# %%
def predict_next_words(model, prefix_tokenizer, word_tokenizer, text, max_prefix_len):
    token_list = prefix_tokenizer.texts_to_sequences([text])[0]
    original_length = len(token_list)
    token_list = pad_sequences([token_list], maxlen=max_prefix_len, padding='post')
    predicted = model.predict(token_list, verbose=0)
    predicted_indices = np.argmax(predicted, axis=-1)[0]
    predicted_words = [word_tokenizer.index_word.get(index, '') for index in predicted_indices[:original_length]]
    
    return ' '.join(predicted_words)

# 例：次の単語を予測
test_prefix = "th fi tw ch in"
predicted_sentence = predict_next_words(model, prefix_tokenizer, word_tokenizer, test_prefix, max_prefix_len)
print(f"Prefix: '{test_prefix}' -> Predicted sentence: '{predicted_sentence}'")

# %%
def calculate_accuracy(model, test_data, test_labels, prefix_tokenizer, word_tokenizer, max_prefix_len, max_sentence_len):
    # 正答数のカウント
    correct_predictions = 0
    total_predictions = 0

    for prefix, sentence in zip(test_data, test_labels):
        # tokenize
        token_list = prefix_tokenizer.texts_to_sequences([prefix])[0]
        original_length = len(token_list)
        # padding
        token_list = pad_sequences([token_list], maxlen=max_prefix_len, padding='post')
        # 予測
        predicted = model.predict(token_list, verbose=0)
        predicted_indices = np.argmax(predicted, axis=-1)[0]
        predicted_words = [word_tokenizer.index_word.get(index, '') for index in predicted_indices[:original_length]]
        print(f"Prefix: '{prefix}' -> Predicted sentence: '{' '.join(predicted_words)}' -> True sentence: '{' '.join(sentence)}'")
        # 単語レベルの正答数をカウント
        correct_predictions += sum([1 for pred, true in zip(predicted_words, sentence) if pred == true])
        # 単語数をカウント
        total_predictions += len(sentence)
    # 正答率の計算
    accuracy = correct_predictions / total_predictions
    return accuracy

accuracy = calculate_accuracy(model, test_prefixes, test_sentences, prefix_tokenizer, word_tokenizer, max_prefix_len, max_sentence_len)
print(f"Test Accuracy: {accuracy:.2f}")