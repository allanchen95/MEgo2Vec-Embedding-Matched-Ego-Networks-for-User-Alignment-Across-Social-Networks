import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_nn_ops


class TextCNN(object):
    """
    A CNN for text subs.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, inputs, sequence_length,embedding_size, num_filters, filter_sizes, dropout_keep_prob,scope):

        # filter_sizes=[2,3]
        self.embedded_chars=inputs
        with tf.name_scope("cnn_subtract"+scope):
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    #filter_shape = [filter_size, embedding_size, 1, num_filters]
                    filter_shape = [filter_size, embedding_size, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv1d(
                        self.embedded_chars,
                        W,
                        stride=1,
                        padding="SAME",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    h_expand = tf.expand_dims(h, -1)
                    # Maxpooling over the outputs
                    pooled = gen_nn_ops._max_pool_v2(
                        h_expand,
                        ksize=[1, sequence_length, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            #self.test=conv
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add dropout

            self.h_drop = tf.nn.dropout(self.h_pool_flat, dropout_keep_prob)

