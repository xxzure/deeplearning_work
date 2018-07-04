# -*- coding: utf-8 -*-
# TextRCNN
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class TextRCNN(object):
    def __init__(self, W_list, num_classes=6, embedding_size=200, hidden_size=150):
        self.X = tf.placeholder(tf.int32, [None, embedding_size])
        self.Y = tf.placeholder(tf.float32, [None, num_classes])
        self.keep = tf.placeholder(tf.float32)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # embedding layer
        embedding = tf.Variable(initial_value=W_list,
                                dtype=tf.float32, trainable=True)
        embed = tf.nn.embedding_lookup(embedding, self.X)
        inputs = embed
        inputs = tf.nn.dropout(inputs, self.keep)

        # Bi-LSTM layer
        # input: [batch_size, max_time, input_size]
        # output: [batch_size,sequence_length,hidden_size]
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
        lstm_fw_cell = rnn.DropoutWrapper(
            lstm_fw_cell, output_keep_prob=self.keep)
        lstm_bw_cell = rnn.DropoutWrapper(
            lstm_bw_cell, output_keep_prob=self.keep)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32)

        # conv1d layer
        outputs= tf.nn.conv1d(outputs, 96,stride=1, padding='VALID')

        # concat layer
        # output: [batch_size,sequence_length,hidden_size*2]
        output_rnn=tf.concat(outputs,axis=2)  

        # get last state
        # output:[batch_size,hidden_size*2]
        output_rnn_last=output_rnn[:,-1,:] 

        # FC layer 
        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal(
                [self.hidden_size*2, self.num_classes], stddev=0.1))    #[hidden_size*2,label_size]
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes])) #[label_size]
            self.p = tf.matmul(output_rnn_last, W) + b
