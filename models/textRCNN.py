# -*- coding: utf-8 -*-
# TextRCNN
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import copy

class TextRCNN(object):
    def __init__(self, W_list, sequence_length = 200,num_classes=6, embedding_size=300, hidden_size=100,batch_size=256,initializer=tf.random_normal_initializer(stddev=0.1)):
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size= batch_size
        self.initializer=initializer
        self.activation = tf.nn.tanh
        self.sequence_length=sequence_length
        self.X = tf.placeholder(tf.int32, [None, sequence_length],name="input_x")
        self.Y = tf.placeholder(tf.float32, [None, num_classes],name="input_y_multilabel")
        self.keep = tf.placeholder(tf.float32)
        self.embedding_size = embedding_size
        self.embedding = tf.Variable(initial_value=W_list,
                                dtype=tf.float32, trainable=True)
        
        self.instantiate_weights()
        self.p = self.inference()


    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("weights"): # embedding matrix
           
            self.left_side_first_word= tf.get_variable("left_side_first_word",shape=[self.batch_size, self.embedding_size],initializer=self.initializer) #TODO
            self.right_side_last_word = tf.get_variable("right_side_last_word",shape=[self.batch_size, self.embedding_size],initializer=self.initializer) #TODO
            self.W_l=tf.get_variable("W_l",shape=[self.embedding_size, self.embedding_size],initializer=self.initializer)
            self.W_r=tf.get_variable("W_r",shape=[self.embedding_size, self.embedding_size],initializer=self.initializer)
            self.W_sl=tf.get_variable("W_sl",shape=[self.embedding_size, self.embedding_size],initializer=self.initializer)
            self.W_sr=tf.get_variable("W_sr",shape=[self.embedding_size, self.embedding_size],initializer=self.initializer)

            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*3, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])
            self.W_conv = tf.get_variable(
                    "W_conv", shape=[self.hidden_size * 3, self.hidden_size], initializer=self.initializer)
            self.b_conv = tf.get_variable(
                    "b_conv", shape=[self.hidden_size])
    
    def get_context_left(self,context_left,embedding_previous):
        """
        :param context_left:
        :param embedding_previous:
        :return: output:[None,embed_size]
        """
        left_c=tf.matmul(context_left,self.W_l) #context_left:[batch_size,embed_size];W_l:[embed_size,embed_size]
        left_e=tf.matmul(embedding_previous,self.W_sl)#embedding_previous;[batch_size,embed_size]
        left_h=left_c+left_e
        context_left=self.activation(left_h)
        return context_left

    def get_context_right(self,context_right,embedding_afterward):
        """
        :param context_right:
        :param embedding_afterward:
        :return: output:[None,embed_size]
        """
        right_c=tf.matmul(context_right,self.W_r)
        right_e=tf.matmul(embedding_afterward,self.W_sr)
        right_h=right_c+right_e
        context_right=self.activation(right_h)
        return context_right

    def conv_layer_with_recurrent_structure(self):
        """
        input:self.embedded_words:[None,sentence_length,embed_size]
        :return: shape:[None,sentence_length,embed_size*3]
        """
        #1. get splitted list of word embeddings
        embedded_words_split=tf.split(self.embedded_words,self.sequence_length,axis=1) #sentence_length个[None,1,embed_size]
        # print(embedded_words_split[0].shape)
        embedded_words_squeezed=[tf.squeeze(x,axis=1) for x in embedded_words_split]#sentence_length个[None,embed_size]
        embedding_previous=self.left_side_first_word
        context_left_previous=tf.zeros((self.batch_size,self.embedding_size))
        #2. get list of context left
        context_left_list=[]
        for i,current_embedding_word in enumerate(embedded_words_squeezed):#sentence_length个[None,embed_size]
            context_left=self.get_context_left(context_left_previous, embedding_previous) #[None,embed_size]
            context_left_list.append(context_left) #append result to list
            embedding_previous=current_embedding_word #assign embedding_previous
            context_left_previous=context_left #assign context_left_previous
        #3. get context right
        embedded_words_squeezed2=copy.copy(embedded_words_squeezed)
        embedded_words_squeezed2.reverse()
        embedding_afterward=self.right_side_last_word
        context_right_afterward = tf.zeros((self.batch_size, self.embedding_size))
        context_right_list=[]
        for j,current_embedding_word in enumerate(embedded_words_squeezed2):
            context_right=self.get_context_right(context_right_afterward,embedding_afterward)
            context_right_list.append(context_right)
            embedding_afterward=current_embedding_word
            context_right_afterward=context_right
        #4.ensemble left,embedding,right to output
        output_list=[]
        for index,current_embedding_word in enumerate(embedded_words_squeezed):
            representation=tf.concat([context_left_list[index],current_embedding_word,context_right_list[index]],axis=1)
            #print(i,"representation:",representation)
            output_list.append(representation) #shape:sentence_length个[None,embed_size*3]
        #5. stack list to a tensor
        #print("output_list:",output_list) #(3, 5, 8, 100)
        output=tf.stack(output_list,axis=1) #shape:[None,sentence_length,embed_size*3]
        #print("output:",output)
        return output

    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.max pooling, 4.FC layer 5.softmax """
        #1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.embedding,self.X) #shape:[batch_size,sentence_length,embed_size]
        
        #2. Bi-lstm layer
        output_conv = self.conv_layer_with_recurrent_structure() #shape:[batch_size,sentence_length,embed_size*3]

        # print("1",output_conv.shape)
        #3. non-linear layer
        output_conv = tf.matmul(tf.reshape(output_conv, [-1, self.hidden_size*3]), self.W_conv) + self.b_conv
        # print("2",output_conv.shape)
        output_conv = tf.reshape(output_conv, [-1, self.sequence_length, self.embedding_size]) # shape:[batch_size, sentence_length, embed_size]

        # print("3",output_conv.shape)
        #4. max pooling
        #print("output_conv:",output_conv) #(3, 5, 8, 100)
        output_pooling = tf.reduce_max(output_conv,axis=1) #shape:[batch_size,embed_size]
        #print("output_pooling:",output_pooling) #(3, 8, 100)
        # print("4",output_pooling.shape)
        #5. logits(use linear layer)
        with tf.name_scope("dropout"):
            h_drop=tf.nn.dropout(output_pooling, keep_prob=self.keep) #[batch_size,num_filters_total]

        with tf.name_scope("output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            # print(h_drop.shape)      # [512,300]
            # print(self.W_projection) # [300, 6]
            # print(self.b_projection) # [6]
            logits = tf.matmul(h_drop, self.W_projection) + self.b_projection  # [batch_size,num_classes]
        return logits


