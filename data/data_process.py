import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import os 
import sys

module_path = os.path.dirname(__file__)

# train_en = pd.read_csv(os.path.join(module_path,'train.csv'))
# train_de = pd.read_csv(os.path.join(module_path,'train_de.csv'))
# train_es = pd.read_csv(os.path.join(module_path,'train_es.csv'))
# train_fr = pd.read_csv(os.path.join(module_path,'train_fr.csv'))
# train_csv_list = [train_en, train_de, train_es, train_fr]
# train = pd.concat(train_csv_list)
train = pd.read_csv(os.path.join(module_path,'train.csv'))
test = pd.read_csv(os.path.join(module_path,'test.csv'))

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values

max_features = 100000
maxlen = 200
embed_size = 300

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=10)

def batch_iter(mode, batch_size, num_epochs, shuffle=True):
    if mode == "train":
        data = list(zip(X_tra, y_tra))
    elif mode == "valid":
        data = list(zip(X_val, y_val))
    elif mode == "test":
        data = x_test
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def get_embedding():
    EMBEDDING_FILE = os.path.join(module_path,'crawl-300d-2M.vec')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

if __name__ == '__main__':
    embedding_matrix = get_embedding()