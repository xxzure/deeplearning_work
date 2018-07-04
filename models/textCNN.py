import tensorflow as tf

class TextCNN():
    def __init__(self, W_list,num_classes=6,max_len=200, num_filters=64,batch_size=256):
        self.X = tf.placeholder(tf.int32, [None, max_len])
        self.Y = tf.placeholder(tf.float32, [None, num_classes])
        self.keep = tf.placeholder(tf.float32)
        self.batch_size=batch_size

        filter_sizes = [1, 2, 3, 5]
        pool_concat = []
        embedding = tf.Variable(initial_value=W_list, dtype=tf.float32, trainable=True)
        embed = tf.nn.embedding_lookup(embedding, self.X)
        inputs = tf.expand_dims(embed, -1)
        inputs = tf.nn.dropout(inputs, self.keep)

        for filter_size in filter_sizes:
            feature_length = max_len - filter_size + 1
            filter_shape = [filter_size, 300, 1, num_filters]
            W_inputs = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b_inputs = tf.Variable(tf.constant(0.0, shape=[num_filters]))
            conv_inputs = tf.nn.conv2d(inputs, W_inputs, strides=[1, 1, 1, 1], padding='VALID')
            h_inputs = tf.nn.relu(tf.nn.bias_add(conv_inputs, b_inputs))
            pool = tf.nn.max_pool(h_inputs, ksize=[1, feature_length, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            pool_concat.append(pool)

        pooled_concat = tf.concat(pool_concat, -1)
        hidden_unit = len(filter_sizes) * num_filters
        pooled_concat = tf.nn.dropout(pooled_concat, self.keep)
        pooled_concat = tf.reshape(pooled_concat, [-1, hidden_unit])


        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal([hidden_unit, num_classes], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
            self.p = tf.nn.xw_plus_b(pooled_concat, W, b)
