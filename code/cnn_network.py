import tensorflow as tf
import numpy as np
import datetime
from tensorflow.python.ops import gen_nn_ops
from text_cnn import TextCNN
from Utils import *
import os
np.set_printoptions(threshold=np.inf)
class textBiLSTM(object):
    def __init__(self, train_set, test_set, num_classes, word_vocab_size,
                 word_embedd_dim, char_vocab_size, char_embedd_dim, l2_reg_lambda, max_neighbor_size, first_num_filters, first_filter_size, second_num_filters ,second_filter_size, n_hidden):
        '''
        # Placeholders for input, output and dropout
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],name="sequence_lengths")
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_x_char = tf.placeholder(tf.int32, [None, sequence_length,max_char_per_word], name="input_x_char")
        #in this step we basically concatentate all the characters of the words. We need to have a separate layer.
        #self.input_x_char_flat = tf.reshape(self.input_x_char,[-1,max_char_per_word*sequence_length],name="input_x_char_flat")
        '''

        self.char_embedd_dim = char_embedd_dim
        self.max_neighbor_size = max_neighbor_size
        self.first_num_filters = first_num_filters
        self.first_filter_size = first_filter_size
        self.second_num_filters = second_num_filters
        self.second_filter_size = second_filter_size

        self.train_set = train_set
        self.test_set = test_set

        self.input1_wname = tf.placeholder(tf.int32, [None, None])
        self.input_wname_len1 = tf.placeholder(tf.int32, [None, None])
        self.input1_waff = tf.placeholder(tf.int32, [None, None])
        self.input_waff_len1 = tf.placeholder(tf.int32, [None, None])
        self.input1_wedu = tf.placeholder(tf.int32, [None, None])
        self.input_wedu_len1 = tf.placeholder(tf.int32, [None, None])
        self.input1_wpub = tf.placeholder(tf.int32, [None, None])
        self.input_wpub_len1 = tf.placeholder(tf.int32, [None, None])

        self.input1_name = tf.placeholder(tf.int32, [None, None, None])
        self.input1_name_maxlen = tf.placeholder(tf.int32)
        self.input1_aff = tf.placeholder(tf.int32, [None, None, None])
        self.input1_aff_maxlen = tf.placeholder(tf.int32)
        self.input1_edu = tf.placeholder(tf.int32, [None, None, None])
        self.input1_edu_maxlen = tf.placeholder(tf.int32)
        self.input1_pub = tf.placeholder(tf.int32, [None, None, None])
        self.input1_pub_maxlen = tf.placeholder(tf.int32)

        self.neighbor_wname1 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_wname_len1 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_waff1 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_waff_len1 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_wedu1 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_wedu_len1 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_wpub1 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_wpub_len1 = tf.placeholder(tf.int32, [None, None])

        self.neighbor_name1 = tf.placeholder(tf.int32, [None, None, None])
        self.neighbor_name1_maxlen = tf.placeholder(tf.int32)
        self.neighbor_aff1 = tf.placeholder(tf.int32, [None, None, None])
        self.neighbor_aff1_maxlen = tf.placeholder(tf.int32)
        self.neighbor_edu1 = tf.placeholder(tf.int32, [None, None, None])
        self.neighbor_edu1_maxlen = tf.placeholder(tf.int32)
        self.neighbor_pub1 = tf.placeholder(tf.int32, [None, None, None])
        self.neighbor_pub1_maxlen = tf.placeholder(tf.int32)

        self.input2_wname = tf.placeholder(tf.int32, [None, None])
        self.input_wname_len2 = tf.placeholder(tf.int32, [None, None])
        self.input2_waff = tf.placeholder(tf.int32, [None, None])
        self.input_waff_len2 = tf.placeholder(tf.int32, [None, None])
        self.input2_wedu = tf.placeholder(tf.int32, [None, None])
        self.input_wedu_len2 = tf.placeholder(tf.int32, [None, None])
        self.input2_wpub = tf.placeholder(tf.int32, [None, None])
        self.input_wpub_len2 = tf.placeholder(tf.int32, [None, None])

        self.input2_name = tf.placeholder(tf.int32, [None, None, None])
        self.input2_name_maxlen = tf.placeholder(tf.int32)
        self.input2_aff = tf.placeholder(tf.int32, [None, None, None])
        self.input2_aff_maxlen = tf.placeholder(tf.int32)
        self.input2_edu = tf.placeholder(tf.int32, [None, None, None])
        self.input2_edu_maxlen = tf.placeholder(tf.int32)
        self.input2_pub = tf.placeholder(tf.int32, [None, None, None])
        self.input2_pub_maxlen = tf.placeholder(tf.int32)

        self.neighbor_wname2 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_wname_len2 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_waff2 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_waff_len2 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_wedu2 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_wedu_len2 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_wpub2 = tf.placeholder(tf.int32, [None, None])
        self.neighbor_wpub_len2 = tf.placeholder(tf.int32, [None, None])

        self.neighbor_name2 = tf.placeholder(tf.int32, [None, None, None])
        self.neighbor_name2_maxlen = tf.placeholder(tf.int32)
        self.neighbor_aff2 = tf.placeholder(tf.int32, [None, None, None])
        self.neighbor_aff2_maxlen = tf.placeholder(tf.int32)
        self.neighbor_edu2 = tf.placeholder(tf.int32, [None, None, None])
        self.neighbor_edu2_maxlen = tf.placeholder(tf.int32)
        self.neighbor_pub2 = tf.placeholder(tf.int32, [None, None, None])
        self.neighbor_pub2_maxlen = tf.placeholder(tf.int32)
        
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope("word_embedding"):
            # plus 1 becuase 0 is for random word
            # init word dict for embedding
            word_0=tf.zeros([1, word_embedd_dim], tf.float32)
            word_other=tf.random_uniform([word_vocab_size, word_embedd_dim], -1, 1)
            word_all=tf.concat([word_0,word_other],0)
            self.W_word = tf.Variable(word_all, trainable=True)
            self.word_embedding_placeholder = tf.placeholder(tf.float32, [word_vocab_size + 1, word_embedd_dim])
            word_embedding_init = self.W_word.assign(self.word_embedding_placeholder)
            ##output is #[batch_size, sequence_length, word_embedd_dim]

            # look up embedding
            self.embedded_words_name1 = tf.nn.embedding_lookup(self.W_word, self.input1_wname)
            self.embedded_words_aff1 = tf.nn.embedding_lookup(self.W_word, self.input1_waff)
            self.embedded_words_edu1 = tf.nn.embedding_lookup(self.W_word, self.input1_wedu)
            self.embedded_words_pub1 = tf.nn.embedding_lookup(self.W_word, self.input1_wpub)

            self.embedded_words_name2 = tf.nn.embedding_lookup(self.W_word, self.input2_wname)
            self.embedded_words_aff2 = tf.nn.embedding_lookup(self.W_word, self.input2_waff)
            self.embedded_words_edu2 = tf.nn.embedding_lookup(self.W_word, self.input2_wedu)
            self.embedded_words_pub2 = tf.nn.embedding_lookup(self.W_word, self.input2_wpub)

            self.neighbor_embedded_words_name1 = tf.nn.embedding_lookup(self.W_word, self.neighbor_wname1)
            self.neighbor_embedded_words_aff1 = tf.nn.embedding_lookup(self.W_word, self.neighbor_waff1)
            self.neighbor_embedded_words_edu1 = tf.nn.embedding_lookup(self.W_word, self.neighbor_wedu1)
            self.neighbor_embedded_words_pub1 = tf.nn.embedding_lookup(self.W_word, self.neighbor_wpub1)

            self.neighbor_embedded_words_name2 = tf.nn.embedding_lookup(self.W_word, self.neighbor_wname2)
            self.neighbor_embedded_words_aff2 = tf.nn.embedding_lookup(self.W_word, self.neighbor_waff2)
            self.neighbor_embedded_words_edu2 = tf.nn.embedding_lookup(self.W_word, self.neighbor_wedu2)
            self.neighbor_embedded_words_pub2 = tf.nn.embedding_lookup(self.W_word, self.neighbor_wpub2)

        with tf.name_scope("char_embedding"):
            # plus 1 becuase 0 is for unknown char
            # init char dict for embedding 
            char_0=tf.zeros([1, char_embedd_dim], tf.float32)
            char_other=tf.random_uniform([char_vocab_size, char_embedd_dim], -1, 1)
            char_all=tf.concat([char_0,char_other],0)
       
            self.W_char = tf.Variable(char_all, trainable=True)
            self.char_embedding_placeholder = tf.placeholder(tf.float32, [char_vocab_size + 1, char_embedd_dim])
            char_embedding_init = self.W_char.assign(self.char_embedding_placeholder)

            # embedding lookup
            self.input1_name_flat = tf.reshape(self.input1_name,[-1, self.input1_name_maxlen * tf.reduce_max(self.input_wname_len1)])
            self.embedded_char_name1 = tf.nn.embedding_lookup(self.W_char, self.input1_name_flat)  # shape [batch_size,max_char_per_word*sequence_length,char_embedd_dim]
            self.embedded_char_name1_dropout = tf.nn.dropout(self.embedded_char_name1, self.dropout_keep_prob)

            self.input1_aff_flat = tf.reshape(self.input1_aff,[-1, self.input1_aff_maxlen * tf.reduce_max(self.input_waff_len1)])
            self.embedded_char_aff1 = tf.nn.embedding_lookup(self.W_char, self.input1_aff_flat)  # shape [batch_size,max_char_per_word*sequence_length,char_embedd_dim]
            self.embedded_char_aff1_dropout = tf.nn.dropout(self.embedded_char_aff1, self.dropout_keep_prob)

            self.input1_edu_flat = tf.reshape(self.input1_edu,[-1, self.input1_edu_maxlen * tf.reduce_max(self.input_wedu_len1)])
            self.embedded_char_edu1 = tf.nn.embedding_lookup(self.W_char, self.input1_edu_flat)  # shape [batch_size,max_char_per_word*sequence_length,char_embedd_dim]
            self.embedded_char_edu1_dropout = tf.nn.dropout(self.embedded_char_edu1, self.dropout_keep_prob)

            self.input1_pub_flat = tf.reshape(self.input1_pub,[-1, self.input1_pub_maxlen * tf.reduce_max(self.input_wpub_len1)])
            self.embedded_char_pub1 = tf.nn.embedding_lookup(self.W_char, self.input1_pub_flat)  # shape [batch_size,max_char_per_word*sequence_length,char_embedd_dim]
            self.embedded_char_pub1_dropout = tf.nn.dropout(self.embedded_char_pub1, self.dropout_keep_prob)

            self.neighbor_name1_flat = tf.reshape(self.neighbor_name1, [-1, self.neighbor_name1_maxlen * tf.reduce_max(self.neighbor_wname_len1)])
            self.neighbor_embedded_char_name1 = tf.nn.embedding_lookup(self.W_char, self.neighbor_name1_flat)
            self.neighbor_embedded_char_name1_dropout = tf.nn.dropout(self.neighbor_embedded_char_name1,self.dropout_keep_prob)

            self.neighbor_aff1_flat = tf.reshape(self.neighbor_aff1, [-1, self.neighbor_aff1_maxlen * tf.reduce_max(self.neighbor_waff_len1)])
            self.neighbor_embedded_char_aff1 = tf.nn.embedding_lookup(self.W_char, self.neighbor_aff1_flat)
            self.neighbor_embedded_char_aff1_dropout = tf.nn.dropout(self.neighbor_embedded_char_aff1,self.dropout_keep_prob)

            self.neighbor_edu1_flat = tf.reshape(self.neighbor_edu1, [-1, self.neighbor_edu1_maxlen * tf.reduce_max(self.neighbor_wedu_len1)])
            self.neighbor_embedded_char_edu1 = tf.nn.embedding_lookup(self.W_char, self.neighbor_edu1_flat)
            self.neighbor_embedded_char_edu1_dropout = tf.nn.dropout(self.neighbor_embedded_char_edu1,self.dropout_keep_prob)

            self.neighbor_pub1_flat = tf.reshape(self.neighbor_pub1, [-1, self.neighbor_pub1_maxlen * tf.reduce_max(self.neighbor_wpub_len1)])
            self.neighbor_embedded_char_pub1 = tf.nn.embedding_lookup(self.W_char, self.neighbor_pub1_flat)
            self.neighbor_embedded_char_pub1_dropout = tf.nn.dropout(self.neighbor_embedded_char_pub1,self.dropout_keep_prob)

            self.input2_name_flat = tf.reshape(self.input2_name,[-1, self.input2_name_maxlen * tf.reduce_max(self.input_wname_len2)])
            self.embedded_char_name2 = tf.nn.embedding_lookup(self.W_char, self.input2_name_flat)  # shape [batch_size,max_char_per_word*sequence_length,char_embedd_dim]
            self.embedded_char_name2_dropout = tf.nn.dropout(self.embedded_char_name2, self.dropout_keep_prob)

            self.input2_aff_flat = tf.reshape(self.input2_aff,[-1, self.input2_aff_maxlen * tf.reduce_max(self.input_waff_len2)])
            self.embedded_char_aff2 = tf.nn.embedding_lookup(self.W_char, self.input2_aff_flat)  # shape [batch_size,max_char_per_word*sequence_length,char_embedd_dim]
            self.embedded_char_aff2_dropout = tf.nn.dropout(self.embedded_char_aff2, self.dropout_keep_prob)

            self.input2_edu_flat = tf.reshape(self.input2_edu,[-1, self.input2_edu_maxlen * tf.reduce_max(self.input_wedu_len2)])
            self.embedded_char_edu2 = tf.nn.embedding_lookup(self.W_char, self.input2_edu_flat)  # shape [batch_size,max_char_per_word*sequence_length,char_embedd_dim]
            self.embedded_char_edu2_dropout = tf.nn.dropout(self.embedded_char_edu2, self.dropout_keep_prob)

            self.input2_pub_flat = tf.reshape(self.input2_pub,[-1, self.input2_pub_maxlen * tf.reduce_max(self.input_wpub_len2)])
            self.embedded_char_pub2 = tf.nn.embedding_lookup(self.W_char, self.input2_pub_flat)  # shape [batch_size,max_char_per_word*sequence_length,char_embedd_dim]
            self.embedded_char_pub2_dropout = tf.nn.dropout(self.embedded_char_pub2, self.dropout_keep_prob)

            self.neighbor_name2_flat = tf.reshape(self.neighbor_name2, [-1, self.neighbor_name2_maxlen * tf.reduce_max(self.neighbor_wname_len2)])
            self.neighbor_embedded_char_name2 = tf.nn.embedding_lookup(self.W_char, self.neighbor_name2_flat)
            self.neighbor_embedded_char_name2_dropout = tf.nn.dropout(self.neighbor_embedded_char_name2,self.dropout_keep_prob)

            self.neighbor_aff2_flat = tf.reshape(self.neighbor_aff2, [-1, self.neighbor_aff2_maxlen * tf.reduce_max(self.neighbor_waff_len2)])
            self.neighbor_embedded_char_aff2 = tf.nn.embedding_lookup(self.W_char, self.neighbor_aff2_flat)
            self.neighbor_embedded_char_aff2_dropout = tf.nn.dropout(self.neighbor_embedded_char_aff2,self.dropout_keep_prob)

            self.neighbor_edu2_flat = tf.reshape(self.neighbor_edu2, [-1, self.neighbor_edu2_maxlen * tf.reduce_max(self.neighbor_wedu_len2)])
            self.neighbor_embedded_char_edu2 = tf.nn.embedding_lookup(self.W_char, self.neighbor_edu2_flat)
            self.neighbor_embedded_char_edu2_dropout = tf.nn.dropout(self.neighbor_embedded_char_edu2,self.dropout_keep_prob)

            self.neighbor_pub2_flat = tf.reshape(self.neighbor_pub2, [-1, self.neighbor_pub2_maxlen * tf.reduce_max(self.neighbor_wpub_len2)])
            self.neighbor_embedded_char_pub2 = tf.nn.embedding_lookup(self.W_char, self.neighbor_pub2_flat)
            self.neighbor_embedded_char_pub2_dropout = tf.nn.dropout(self.neighbor_embedded_char_pub2,self.dropout_keep_prob)

        # model procedure 
        # multi-view embedding

        name1 = self.name_embedding(self.embedded_words_name1, self.embedded_char_name1_dropout, tf.reduce_max(self.input_wname_len1), self.input1_name_maxlen)
        aff1 = self.attribute_embedding(self.embedded_words_aff1, self.embedded_char_aff1_dropout, tf.reduce_max(self.input_waff_len1), self.input1_aff_maxlen)
        edu1 = self.attribute_embedding(self.embedded_words_edu1, self.embedded_char_edu1_dropout, tf.reduce_max(self.input_wedu_len1), self.input1_edu_maxlen)
        pub1 = self.attribute_embedding(self.embedded_words_pub1, self.embedded_char_pub1_dropout, tf.reduce_max(self.input_wpub_len1), self.input1_pub_maxlen)


        neighbor_name1 = self.name_embedding(self.neighbor_embedded_words_name1, self.neighbor_embedded_char_name1_dropout, tf.reduce_max(self.neighbor_wname_len1), self.neighbor_name1_maxlen)
        neighbor_aff1 = self.attribute_embedding(self.neighbor_embedded_words_aff1, self.neighbor_embedded_char_aff1_dropout, tf.reduce_max(self.neighbor_waff_len1), self.neighbor_aff1_maxlen)
        neighbor_edu1 = self.attribute_embedding(self.neighbor_embedded_words_edu1, self.neighbor_embedded_char_edu1_dropout, tf.reduce_max(self.neighbor_wedu_len1), self.neighbor_edu1_maxlen)
        neighbor_pub1 = self.attribute_embedding(self.neighbor_embedded_words_pub1, self.neighbor_embedded_char_pub1_dropout, tf.reduce_max(self.neighbor_wpub_len1), self.neighbor_pub1_maxlen)

        name2 = self.name_embedding(self.embedded_words_name2, self.embedded_char_name2_dropout, tf.reduce_max(self.input_wname_len2), self.input2_name_maxlen)
        aff2 = self.attribute_embedding(self.embedded_words_aff2, self.embedded_char_aff2_dropout, tf.reduce_max(self.input_waff_len2), self.input2_aff_maxlen)
        edu2 = self.attribute_embedding(self.embedded_words_edu2, self.embedded_char_edu2_dropout, tf.reduce_max(self.input_wedu_len2), self.input2_edu_maxlen)
        pub2 = self.attribute_embedding(self.embedded_words_pub2, self.embedded_char_pub2_dropout, tf.reduce_max(self.input_wpub_len2), self.input2_pub_maxlen)

        neighbor_name2 = self.name_embedding(self.neighbor_embedded_words_name2, self.neighbor_embedded_char_name2_dropout, tf.reduce_max(self.neighbor_wname_len2), self.neighbor_name2_maxlen)
        neighbor_aff2 = self.attribute_embedding(self.neighbor_embedded_words_aff2, self.neighbor_embedded_char_aff2_dropout, tf.reduce_max(self.neighbor_waff_len2), self.neighbor_aff2_maxlen)
        neighbor_edu2 = self.attribute_embedding(self.neighbor_embedded_words_edu2, self.neighbor_embedded_char_edu2_dropout, tf.reduce_max(self.neighbor_wedu_len2), self.neighbor_edu2_maxlen)
        neighbor_pub2 = self.attribute_embedding(self.neighbor_embedded_words_pub2, self.neighbor_embedded_char_pub2_dropout, tf.reduce_max(self.neighbor_wpub_len2), self.neighbor_pub2_maxlen)

        # attention node embedding
        node1 = self.get_node_hidden(name1, tf.reduce_max(self.input_wname_len1),
                                         aff1, tf.reduce_max(self.input_waff_len1),
                                         edu1, tf.reduce_max(self.input_wedu_len1),
                                         pub1, tf.reduce_max(self.input_wpub_len1),
                                         n_hidden, self.dropout_keep_prob)

        neighbor_node1 = self.get_node_hidden(neighbor_name1, tf.reduce_max(self.neighbor_wname_len1),
                                            neighbor_aff1, tf.reduce_max(self.neighbor_waff_len1),
                                            neighbor_edu1, tf.reduce_max(self.neighbor_wedu_len1),
                                            neighbor_pub1, tf.reduce_max(self.neighbor_wpub_len1),
                                            n_hidden, self.dropout_keep_prob)


        node2 = self.get_node_hidden(name2, tf.reduce_max(self.input_wname_len2),
                                         aff2, tf.reduce_max(self.input_waff_len2),
                                         edu2, tf.reduce_max(self.input_wedu_len2),
                                         pub2, tf.reduce_max(self.input_wpub_len2),
                                         n_hidden, self.dropout_keep_prob)

        neighbor_node2 = self.get_node_hidden(neighbor_name2, tf.reduce_max(self.neighbor_wname_len2),
                                            neighbor_aff2, tf.reduce_max(self.neighbor_waff_len2),
                                            neighbor_edu2, tf.reduce_max(self.neighbor_wedu_len2),
                                            neighbor_pub2, tf.reduce_max(self.neighbor_wpub_len2),
                                            n_hidden, self.dropout_keep_prob)
        
        # social convolution
        self.neighbor_size = tf.placeholder(tf.int32, [None])
        self.neighbors = tf.placeholder(tf.int32, [None, None])

        self.neighbor_a = tf.placeholder(tf.int32, [None, None, None])
        self.input_y = tf.placeholder(tf.int32, [None])

        
        shape = tf.shape(self.neighbor_a)
        
        reshape_node1 = tf.reshape(neighbor_node1, [shape[0], self.max_neighbor_size, 2 * n_hidden*3+2*32])
        reshape_node2 = tf.reshape(neighbor_node2, [shape[0], self.max_neighbor_size, 2 * n_hidden*3+2*32])

        reshape_node1_drop = tf.nn.dropout(reshape_node1, self.dropout_keep_prob)
        reshape_node2_drop = tf.nn.dropout(reshape_node2, self.dropout_keep_prob)

        all_node1,all_node2 = self.graph_all_attention(node1,node2,reshape_node1_drop, reshape_node2_drop, self.neighbor_a, self.neighbor_size,2*n_hidden*3+2*32, 'all_attention')
        
        # structure embedding
        structure_embed = self.structure_cnn(self.neighbor_a,'structure_cnn')


        # full-connected layer
        hidden1 = tf.concat([node1,all_node1], 1)
        hidden2 = tf.concat([node2,all_node2], 1)
        hidden = tf.abs(tf.subtract(hidden1, hidden2))
        self.diff = tf.concat([hidden,structure_embed],1)
        
        with tf.name_scope("output"):

            W1 = tf.get_variable(
                "W1",
                shape=[(2 * n_hidden * 3+2*32) * 2+64, 100],
                initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[100]), name="b1")

            fc1 = tf.nn.relu(tf.nn.xw_plus_b(self.diff, W1, b1, name="fc1"), name="relu")

            fc2 = tf.nn.dropout(fc1, self.dropout_keep_prob)

            W2 = tf.get_variable(
                "W2",
                shape=[100, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")

            self.probability = tf.nn.xw_plus_b(fc2, W2, b2, name="probability")
            self.predictions = tf.argmax(self.probability, 1, name="predictions")
        
        '''
        with tf.name_scope("output"):
            fc_1 = tf.contrib.layers.fully_connected(self.diff, 50, activation_fn=tf.nn.relu)
            fc_2 = tf.nn.dropout(fc_1, self.dropout_keep_prob)
            self.probability = tf.contrib.layers.fully_connected(fc_2, num_classes, activation_fn=None)
            self.predictions = tf.argmax(self.probability, 1, name="predictions")
        '''
        with tf.name_scope("loss"):
            tr_variables = tf.trainable_variables()
            l2_loss = 0
            for v in tr_variables:
                l2_loss += tf.nn.l2_loss(v)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.probability, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss


        self._lr = tf.Variable(0.0, trainable=False)
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self._lr)
        '''
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        '''

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
        # with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        
        checkpoint_dir = "./checkpoints_test2/"
        self.save_path = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #self.saver = tf.train.Saver(tf.global_variables())
        self.saver = tf.train.Saver(max_to_keep=1)
    

    def name_embedding(self, wattr, attr, sequence_length, max_char_per_word):
        # Add CNN get filters and char embedding for name
        with tf.name_scope("name"):
            with tf.name_scope("name_conv_maxPool"):
                filter_shape = [self.first_filter_size, self.char_embedd_dim, self.first_num_filters]
                W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_conv")
                b_conv = tf.Variable(tf.constant(0.1, shape=[self.first_num_filters]), name="b_conv")

                conv = tf.nn.conv1d(attr, W_conv, stride=1, padding="SAME", name="conv")  # will have dimensions [batch_size,out_width,first_num_filters] out_width is a function of max_words,first_filter_size and stride_size
                # out_width for same padding iwth stride 1  given by (max_char_per_word * sequence_length)
                # h = tf.nn.bias_add(conv, b_conv,name="add bias")#does not change dimensions
                h_expand = tf.expand_dims(conv, -1)
                pooled = gen_nn_ops._max_pool_v2(
                    h_expand,
                    # [batch, height, width, channels]
                    ksize=[1, max_char_per_word, 1, 1],
                    # On the batch size dimension and the channels dimension, ksize is 1 because we don't want to take the maximum over multiple examples, or over multiples channels.
                    strides=[1, max_char_per_word, 1, 1],
                    padding='VALID',
                    name="pooled")

                char_pool_flat = tf.reshape(pooled, [-1, sequence_length, self.first_num_filters], name="char_pool_flat")
                char_features_dropout = tf.nn.dropout(char_pool_flat, self.dropout_keep_prob,name="char_features_dropout")
        return char_features_dropout

    def attribute_embedding(self, wattr, attr, sequence_length, max_char_per_word):
        # Add CNN get filters and combine with word
        with tf.name_scope("node_id"):
            with tf.name_scope("attr_conv_maxPool"):
                filter_shape = [self.first_filter_size, self.char_embedd_dim, self.first_num_filters]
                W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_conv")
                b_conv = tf.Variable(tf.constant(0.1, shape=[self.first_num_filters]), name="b_conv")

                conv = tf.nn.conv1d(attr, W_conv, stride=1, padding="SAME", name="conv")  # will have dimensions [batch_size,out_width,first_num_filters] out_width is a function of max_words,first_filter_size and stride_size
                # h = tf.nn.bias_add(conv, b_conv,name="add bias")
                h_expand = tf.expand_dims(conv, -1)
                pooled = gen_nn_ops._max_pool_v2(
                    h_expand,
                    # [batch, height, width, channels]
                    ksize=[1, max_char_per_word, 1, 1],
                    # On the batch size dimension and the channels dimension, ksize is 1 because we don't want to take the maximum over multiple examples, or over multiples channels.
                    strides=[1, max_char_per_word, 1, 1],
                    padding='VALID',
                    name="pooled")

                char_pool_flat = tf.reshape(pooled, [-1, sequence_length, self.first_num_filters], name="char_pool_flat")
                # [batch, sequence_length, word_embedd_dim+first_num_filters]
                word_char_features = tf.concat([wattr, char_pool_flat],axis=2)  # we mean that the feature with index 2 i/e first_num_filters is variable
                word_char_features_dropout = tf.nn.dropout(word_char_features, self.dropout_keep_prob,name="word_char_features_dropout")
        return word_char_features_dropout

    def get_node_hidden(self, name, name_len, aff, aff_len, edu, edu_len, pub, pub_len, n_hidden, dropout_keep_prob):
        with tf.variable_scope("scope", reuse=tf.AUTO_REUSE):
            name_cnn = TextCNN(name, name_len, self.char_embedd_dim, self.first_num_filters, self.second_filter_size, dropout_keep_prob, scope='name').h_drop
            #name_cnn = TextCNN(name, name_len, n_hidden, n_hidden, dropout_keep_prob, scope='name').h_drop
            
            aff_cnn = TextCNN(aff, aff_len, n_hidden, self.second_num_filters, self.second_filter_size, dropout_keep_prob, scope='aff').h_drop
            edu_cnn = TextCNN(edu, edu_len, n_hidden, self.second_num_filters, self.second_filter_size, dropout_keep_prob, scope='edu').h_drop
            pub_cnn = TextCNN(pub, pub_len, n_hidden, self.second_num_filters, self.second_filter_size, dropout_keep_prob, scope='pub').h_drop

            # Attribute-Attention
            
            w1 = tf.get_variable("W1", (32 * 2, 1), dtype=tf.float32,initializer=tf.random_normal_initializer())
            #w1 = tf.get_variable("W1", (n_hidden * 2, 1), dtype=tf.float32, initializer=tf.random_normal_initializer())
            w2 = tf.get_variable("W2", (n_hidden * 2, 1), dtype=tf.float32, initializer=tf.random_normal_initializer())
            w3 = tf.get_variable("W3", (n_hidden * 2, 1), dtype=tf.float32, initializer=tf.random_normal_initializer())
            w4 = tf.get_variable("W4", (n_hidden * 2, 1), dtype=tf.float32, initializer=tf.random_normal_initializer())

            b1 = tf.get_variable('b1', (1,), dtype=tf.float32, initializer=tf.random_normal_initializer())
            b2 = tf.get_variable('b3', (1,), dtype=tf.float32, initializer=tf.random_normal_initializer())
            b3 = tf.get_variable('b3', (1,), dtype=tf.float32, initializer=tf.random_normal_initializer())
            b4 = tf.get_variable('b4', (1,), dtype=tf.float32, initializer=tf.random_normal_initializer())

            attention1 =tf.tanh(tf.nn.bias_add(tf.matmul(name_cnn, w1), b1))
            attention2 =tf.tanh(tf.nn.bias_add(tf.matmul(aff_cnn, w2), b2))
            attention3 = tf.tanh(tf.nn.bias_add(tf.matmul(edu_cnn, w3), b3))
            attention4 = tf.tanh(tf.nn.bias_add(tf.matmul(pub_cnn, w4), b4))

            attention = tf.concat([attention1, attention2, attention3, attention4], 1)
            softmax=tf.nn.softmax(attention)
            a1, a2, a3, a4 = tf.split(softmax, num_or_size_splits=4, axis=1)
            concat_hidden = tf.concat([a1 * name_cnn, a2 * aff_cnn, a3 * edu_cnn, a4 * pub_cnn], 1)
            # concat_hidden = a1 * name_cnn + a2 * aff_cnn + a3 * edu_cnn + a4 * pub_cnn
            weight = tf.concat([a1, a2, a3, a4], 1)
            
        return concat_hidden

    def graph_all_attention(self,ax,bx,x1, x2, a, neighbor_size,embed_size, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            att_kernel = tf.get_variable("a_k", (embed_size*4, 1), dtype=tf.float32, initializer=tf.random_normal_initializer())
            att_b = tf.get_variable('att_b', (1,), dtype=tf.float32, initializer=tf.random_normal_initializer())
            # x1 x2.shape:[batch_size,self.max_neighbor_size,embed_size]
            # x.shape:[batch_size,embed_size]
            a_shape = tf.shape(a)

            size=a_shape[0]

            #relation-attention

            reshape_ax = tf.reshape(tf.tile(ax, [1, self.max_neighbor_size]),[size, self.max_neighbor_size, embed_size])  # [batch_size,self.max_neighbor_size,embed]

            combination1 = tf.abs(tf.subtract(reshape_ax, x1))  # [batch_size,self.max_neighbor_size,embed]

            reshape_bx = tf.reshape(tf.tile(bx, [1, self.max_neighbor_size]),[size, self.max_neighbor_size, embed_size])  # [batch_size,self.max_neighbor_size,embed]

            combination2 = tf.abs(tf.subtract(reshape_bx, x2))  # [batch_size,self.max_neighbor_size,embed]

            combination_slices = tf.abs(tf.subtract(combination1, combination2))

            #difference-attention

            diff_transf_x = tf.abs(tf.subtract(x1, x2)) # [batch_size,self.max_neighbor_size,embed]

            #feature-attention

            fea_transf_x = tf.concat([x1, x2], 2) # [batch_size,self.max_neighbor_size,embed*2]

            #combine all

            combination_all=tf.concat([diff_transf_x,combination_slices,fea_transf_x], 2) # [batch_size,self.max_neighbor_size,embed*4]

            # Attention head
            dense = tf.nn.bias_add(tf.einsum('aij,jk->aik', combination_all, att_kernel),att_b)  # [batch_size,self.max_neighbor_size,1]

            # add nonlinearty
            relu = tf.nn.leaky_relu(dense, alpha=0.2)

            mask = tf.sequence_mask(neighbor_size, self.max_neighbor_size, dtype=tf.float32)
            comparison = tf.equal(mask, tf.constant(0, dtype=tf.float32))
            masked = tf.expand_dims(tf.where(comparison, tf.ones_like(mask) * -10e9, tf.zeros_like(mask)), -1)
            # [batch_size,self.max_neighbor_size,1]

            # attention
            attention = relu + masked
            softmax = tf.nn.softmax(attention, 1)
            node1 = tf.reduce_sum(x1 * softmax, 1)
            node2 = tf.reduce_sum(x2 * softmax, 1)

        return node1, node2

    def structure_cnn(self, a,scope):
        # Add CNN get structure information:

        with tf.name_scope(scope):  # a.shape:[batch_size,max_node_num,max_node_num]
            filter_size=self.max_neighbor_size*self.max_neighbor_size
            input_channel=1
            output_channel=64
            a=tf.cast(a,tf.float32)
            reshape_a=tf.reshape(a,[-1,self.max_neighbor_size*self.max_neighbor_size,1])

            filter_shape = [filter_size,input_channel,output_channel]
            #W_conv.set_shape([1, input_channel, output_channel])
            W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_conv1d")

            b_conv = tf.Variable(tf.constant(0.1, shape=[output_channel]), name="b_conv1d")

            conv = tf.nn.conv1d(reshape_a,
                                W_conv,
                                stride=14,
                                padding="VALID",
                                name='conv1d')
            h = tf.nn.relu(tf.nn.bias_add(conv,b_conv,name="bias"))
            output = tf.reduce_sum(h, 1)

        return output



    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def predict(self, session, eval_set):
        y_true = []
        prob_pred = []
	     
        string = "test batch count:{}".format(eval_set.batch_count)
        #print string
        for i in range(eval_set.batch_count):
            ret = eval_set.get_batch(i)
            feed_dict = {
                self.input1_wname: ret[0],
                self.input_wname_len1: ret[1],
                self.input1_waff: ret[2],
                self.input_waff_len1: ret[3],
                self.input1_wedu: ret[4],
                self.input_wedu_len1: ret[5],
                self.input1_wpub: ret[6],
                self.input_wpub_len1: ret[7],

                self.input1_name: ret[8],
                self.input1_name_maxlen: ret[9],
                self.input1_aff: ret[10],
                self.input1_aff_maxlen: ret[11],
                self.input1_edu: ret[12],
                self.input1_edu_maxlen: ret[13],
                self.input1_pub: ret[14],
                self.input1_pub_maxlen: ret[15],

                self.neighbor_wname1: ret[16],
                self.neighbor_wname_len1: ret[17],
                self.neighbor_waff1: ret[18],
                self.neighbor_waff_len1: ret[19],
                self.neighbor_wedu1: ret[20],
                self.neighbor_wedu_len1: ret[21],
                self.neighbor_wpub1: ret[22],
                self.neighbor_wpub_len1: ret[23],

                self.neighbor_name1: ret[24],
                self.neighbor_name1_maxlen: ret[25],
                self.neighbor_aff1: ret[26],
                self.neighbor_aff1_maxlen: ret[27],
                self.neighbor_edu1: ret[28],
                self.neighbor_edu1_maxlen: ret[29],
                self.neighbor_pub1: ret[30],
                self.neighbor_pub1_maxlen: ret[31],

                self.input2_wname: ret[32],
                self.input_wname_len2: ret[33],
                self.input2_waff: ret[34],
                self.input_waff_len2: ret[35],
                self.input2_wedu: ret[36],
                self.input_wedu_len2: ret[37],
                self.input2_wpub: ret[38],
                self.input_wpub_len2: ret[39],

                self.input2_name: ret[40],
                self.input2_name_maxlen: ret[41],
                self.input2_aff: ret[42],
                self.input2_aff_maxlen: ret[43],
                self.input2_edu: ret[44],
                self.input2_edu_maxlen: ret[45],
                self.input2_pub: ret[46],
                self.input2_pub_maxlen: ret[47],

                self.neighbor_wname2: ret[48],
                self.neighbor_wname_len2: ret[49],
                self.neighbor_waff2: ret[50],
                self.neighbor_waff_len2: ret[51],
                self.neighbor_wedu2: ret[52],
                self.neighbor_wedu_len2: ret[53],
                self.neighbor_wpub2: ret[54],
                self.neighbor_wpub_len2: ret[55],

                self.neighbor_name2: ret[56],
                self.neighbor_name2_maxlen: ret[57],
                self.neighbor_aff2: ret[58],
                self.neighbor_aff2_maxlen: ret[59],
                self.neighbor_edu2: ret[60],
                self.neighbor_edu2_maxlen: ret[61],
                self.neighbor_pub2: ret[62],
                self.neighbor_pub2_maxlen: ret[63],

                self.neighbor_size: ret[64],
                self.neighbors: ret[65],
                self.neighbor_a: ret[66],

                self.input_y: ret[67],
                self.dropout_keep_prob: 1.0
            }
            probability, predictions, loss= session.run([self.probability, self.predictions, self.loss], feed_dict)
            if (i % 60 == 0):
                print "Vaild or test loss {:.4f}".format(loss)
            prob_pred.extend(predictions)
            y_true.extend(ret[67])

        return y_true, prob_pred

    def train(self, session, num_epochs, lr_init, max_decay_epoch, dropout_keep_prob, evaluate_every):
        # print('write graph begin')
        # tf.train.write_graph(session.graph_def, self.graph_save_dir, 'simple_dynamic_lstm.pb', as_text=True)
        # print('write graph end')

        batch_count = self.train_set.batch_count
        print "batch count = ", batch_count
        best_f1 = 0.0
        test_auc = 0.0
        lr_decay = 0.86
        lr_end = 1e-5

        for epoch in range(num_epochs):
            '''
            # change lr
            tmp = lr_decay ** max(epoch + 1 - max_decay_epoch, 0.0)
            decay_lr = lr_init * tmp
            if decay_lr < lr_end:
                break
            print "lr: ",decay_lr
            '''
            self.assign_lr(session, lr_init)

            shuffle_indices = np.random.permutation(np.arange(batch_count))

            for i in range(batch_count):
                ret = self.train_set.get_batch(shuffle_indices[i])
                feed_dict = {
                    self.input1_wname: ret[0],
                    self.input_wname_len1: ret[1],
                    self.input1_waff: ret[2],
                    self.input_waff_len1: ret[3],
                    self.input1_wedu: ret[4],
                    self.input_wedu_len1: ret[5],
                    self.input1_wpub: ret[6],
                    self.input_wpub_len1: ret[7],

                    self.input1_name: ret[8],
                    self.input1_name_maxlen: ret[9],
                    self.input1_aff: ret[10],
                    self.input1_aff_maxlen: ret[11],
                    self.input1_edu: ret[12],
                    self.input1_edu_maxlen: ret[13],
                    self.input1_pub: ret[14],
                    self.input1_pub_maxlen: ret[15],

                    self.neighbor_wname1: ret[16],
                    self.neighbor_wname_len1: ret[17],
                    self.neighbor_waff1: ret[18],
                    self.neighbor_waff_len1: ret[19],
                    self.neighbor_wedu1: ret[20],
                    self.neighbor_wedu_len1: ret[21],
                    self.neighbor_wpub1: ret[22],
                    self.neighbor_wpub_len1: ret[23],

                    self.neighbor_name1: ret[24],
                    self.neighbor_name1_maxlen: ret[25],
                    self.neighbor_aff1: ret[26],
                    self.neighbor_aff1_maxlen: ret[27],
                    self.neighbor_edu1: ret[28],
                    self.neighbor_edu1_maxlen: ret[29],
                    self.neighbor_pub1: ret[30],
                    self.neighbor_pub1_maxlen: ret[31],

                    self.input2_wname: ret[32],
                    self.input_wname_len2: ret[33],
                    self.input2_waff: ret[34],
                    self.input_waff_len2: ret[35],
                    self.input2_wedu: ret[36],
                    self.input_wedu_len2: ret[37],
                    self.input2_wpub: ret[38],
                    self.input_wpub_len2: ret[39],

                    self.input2_name: ret[40],
                    self.input2_name_maxlen: ret[41],
                    self.input2_aff: ret[42],
                    self.input2_aff_maxlen: ret[43],
                    self.input2_edu: ret[44],
                    self.input2_edu_maxlen: ret[45],
                    self.input2_pub: ret[46],
                    self.input2_pub_maxlen: ret[47],

                    self.neighbor_wname2: ret[48],
                    self.neighbor_wname_len2: ret[49],
                    self.neighbor_waff2: ret[50],
                    self.neighbor_waff_len2: ret[51],
                    self.neighbor_wedu2: ret[52],
                    self.neighbor_wedu_len2: ret[53],
                    self.neighbor_wpub2: ret[54],
                    self.neighbor_wpub_len2: ret[55],

                    self.neighbor_name2: ret[56],
                    self.neighbor_name2_maxlen: ret[57],
                    self.neighbor_aff2: ret[58],
                    self.neighbor_aff2_maxlen: ret[59],
                    self.neighbor_edu2: ret[60],
                    self.neighbor_edu2_maxlen: ret[61],
                    self.neighbor_pub2: ret[62],
                    self.neighbor_pub2_maxlen: ret[63],

                    self.neighbor_size: ret[64],
                    self.neighbors: ret[65],
                    self.neighbor_a: ret[66],

                    self.input_y: ret[67],
                    self.dropout_keep_prob: dropout_keep_prob
                }

                # print "y"
                # print ret[32]
                _, step, loss_value= session.run([self.train_op, self.global_step, self.loss], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                current_step = tf.train.global_step(session, self.global_step)
                if current_step % evaluate_every == 0:
                    print("{}: epoch {}, step {}, batch {}, loss {:.2f}".format(time_str, epoch, step, i, loss_value))
                    y_true, prop_pred = self.predict(session, self.test_set)

                    # print y_true
                    # print prop_pred
                    precision, recall, f1= cal_f1(y_true, prop_pred)
                    print("Validation f1 {:.4f}".format(f1))

                    if best_f1 < f1:
                        best_f1 = f1
                        self.saver.save(session, self.save_path + 'all.ckpt', global_step=current_step)
                        #y_true, prop_pred = self.predict(session, self.test_set)
                        auc = cal_auc(y_true, prop_pred)
                        #precision, recall, f1= cal_f1(y_true, prop_pred)
                        print("Test auc {:.4f}, precision {:.4f}, recall {:.4f}, f1 {:.4f}".format(auc, precision, recall,f1))






