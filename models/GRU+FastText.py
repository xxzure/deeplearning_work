# -*- coding: utf-8 -*-
"""
Created on Wed Jun  27 19:32:43 2018

@author: luzc
"""
###导入必要的包。主要是sklearn

import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

###读取文件
##字典crawl-300d-2M.vec，下载地址为：https://storage.googleapis.com/kaggle-datasets/14154/19053/crawl-300d-2M.vec.zip
EMBEDDING_FILE = 'C:/Users/luzc/Desktop/dlhm/data/crawl-300d-2M.vec'
##将csv文件读取为dataframe文件
train = pd.read_csv('C:/Users/luzc/Desktop/dlhm/data/train.csv')
test = pd.read_csv('C:/Users/luzc/Desktop/dlhm/data/test.csv')
submission = pd.read_csv('C:/Users/luzc/Desktop/dlhm/data/sample_submission.csv')

##取dataframe文件中的值
X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values


###处理数据
max_features = 30000
maxlen = 100
embed_size = 300

##将文本转化为向量
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf-8'))

##将向量转化为三位矩阵
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

###roc_auc评价
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
##使用roc_auc_score函数计算分数
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

###构建神经网络模型
def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)    
    model = Model(inputs=inp, outputs=outp)
    #编译模型
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
##获取神经网络模型
model = get_model()

###设置模型参数
batch_size = 32
epochs = 2

###模型训练
##得到训练集和测试机
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
##学习过程中的auc_roc
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
##训练模型
hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=2)

###模型在测试集上测试
y_pred = model.predict(x_test, batch_size=1024)
##生成csv文件
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)


