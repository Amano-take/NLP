# %%
# コーパスは1701文の英文
import gensim.downloader as gendl
corpus = gendl.load("text8")
corpus = list(corpus)
# %%
# データサイズを制限

training_ratio = 0.9
total_size = 1500

corpus = corpus[:total_size]
training_size = int(total_size * training_ratio)
# %%
# データの前処理
flat_corpus = []
for sentence in corpus:
    temp = []
    for word in sentence:
        if len(word) >= 2:
            temp.append(word[:2])
        else:
            temp.append(word + " ")
    flat_corpus.append(temp)

training_data = flat_corpus[:training_size]
training_answer = corpus[:training_size]
test_data = flat_corpus[training_size:]
test_answer = corpus[training_size:]
true_test_data = ["a ", "fi", "tw", "ch", "in"]

# %%
# 出現する可能性のある最初の2文字のリストを作成
first_two_words = []
for f in range(26):
    for i in range(26):
        first_two_words.append(chr(97 + f) + chr(97 + i))
for f in range(26):
    first_two_words.append(chr(97 + f) + " ")


# %%
# n-gramの辞書を返すメソッド
from collections import defaultdict, Counter
def make_ngram_model(n, training_data, training_answer):
    ngram_model = defaultdict(Counter)
    for sentence, answer in zip(training_data, training_answer):
        for i in range(len(sentence) - n + 1):
            prefix = tuple(sentence[i:i+n])
            ngram_model[prefix][answer[i+n-1]] += 1
    ngram_model_predictions = defaultdict(str)
    for prefix in ngram_model:
        ngram_model_predictions[prefix] = ngram_model[prefix].most_common(1)[0][0]
    return ngram_model_predictions

# %%
# back-offを考慮したpredictメソッド

def back_off_ngram_predict(ngram_models, prefix):
    if prefix in ngram_models[len(prefix)]:
        return ngram_models[len(prefix)][prefix]
    else:
        return back_off_ngram_predict(ngram_models, prefix[1:])
# %% 
#　モデルの評価
def evaluate_ngram_model(ngram_models, test_data, test_answer):
    correct = 0
    for sentence, answer in zip(test_data, test_answer):
        for i in range(len(sentence)):
            prefix = tuple(sentence[max(0, i-len(ngram_models)+2):i+1])
            prediction = back_off_ngram_predict(ngram_models, prefix)
            if prediction == answer[i]:
                correct += 1
    return correct / sum([len(sentence) for sentence in test_answer])

# %%
# テストデータでの実行による評価
def show_ngram_model_predictions(ngram_models, test_sentence, test_answer):
    for i in range(len(test_sentence)):
        prefix = tuple(test_sentence[max(0, i-len(ngram_models)+2):i+1])
        prediction = back_off_ngram_predict(ngram_models, prefix)
        print(f"Prefix: {prefix}, Prediction: {prediction}, Answer: {test_answer[i]}")
# %%
# back-offを考慮したn-gramモデルの作成
n = 3
ngram_models = []
for i in range(0, n+1):
    ngram_models.append(make_ngram_model(i, training_data, training_answer))
# %%
# モデルの評価
evaluate_ngram_model(ngram_models, test_data, test_answer)

# %%
show_ngram_model_predictions(ngram_models, test_data[0], test_answer[0])


