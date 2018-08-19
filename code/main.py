import numpy as np
import tensorflow as tf
import cnn_network as cnn
import os
from DatasetProcessor import *
import cPickle as pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():

    # Data loading params
    tf.flags.DEFINE_float("test_sample_percentage", 0.2 , "Percentage of the test data")
    tf.flags.DEFINE_string("data_dir", "../data_new", "Data source")
    tf.flags.DEFINE_string("network_path", "../data_new/network_A.p", "Whole_network")  
    tf.flags.DEFINE_string("train_network_path", "../relation-att/train_network.p", "Train_network")
    tf.flags.DEFINE_string("test_network_path", "../relation-att/test_network.p", "Test_network")      

    # Model Hyperparameters
    tf.flags.DEFINE_integer("max_neighbor_size", 14, "Max neighbor num for one instance (default: 14)")
    tf.flags.DEFINE_integer("word_embedding_dim", 64, "Dimensionality of word embedding (default: 64)")
    tf.flags.DEFINE_integer("char_embedding_dim", 32, "Dimensionality of character embedding (default: 32)")
    tf.flags.DEFINE_integer("hidden_size", 96, "Dimensionality of word+char embedding (default: 96)")
    tf.flags.DEFINE_integer("first_num_filters", 32, "Number of filters per filter size for first cnn layer (default: 32)")
    tf.flags.DEFINE_integer("second_num_filters", 96, "Number of filters per filter size for second cnn layer (default: 96)")
    tf.flags.DEFINE_integer("first_filter_size", 3, "Comma-separated filter sizes for first cnn layer (default: 3)")
    tf.flags.DEFINE_string("second_filter_size", "2,3", "Comma-separated filter sizes for second cnn char (default: '2,3')")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_float("learning_rate", 0.001, "learning_rate (default: 0.001)") 
    tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 50)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 100)")
    tf.flags.DEFINE_integer("max_decay_epoch", 5, "Decay epoch num for changing learning_rate (default: 5)")
    tf.flags.DEFINE_integer("grad_clip", 5, "Gradient clip (default: 5)")
    tf.flags.DEFINE_integer("class_num", 2, "Class num (default: 2)")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    ### Load data
    print("Loading data...")

    dir = FLAGS.data_dir
 
    feature_path = dir + "/simplified_feature.p"
    raw_features = np.asarray(pickle.load(open(feature_path, "rb")))

    pairs_path = dir + "/new_pairs.p"
    pairs = np.asarray(pickle.load(open(pairs_path, "rb")))

    
    #network = np.asarray(pickle.load(open(FLAGS.network_path, "rb")))
    #print "network:",len(network)
    
    
    test_network = np.asarray(pickle.load(open(FLAGS.test_network_path, "rb")))

    train_network = np.asarray(pickle.load(open(FLAGS.train_network_path, "rb")))
    
    vocabulary_path = dir + "/vocabulary.p"
    vocabulary = pickle.load(open(vocabulary_path, "rb"))

    charvoca_path = dir + '/char-voca_new.p'
    charvaca = pickle.load(open(charvoca_path, "rb"))

    # modify feature lengths
    def cut_len(feature,maxlen,flag):
        text = map(lambda line: re.split('[\s\-\(\)"\[\]]+', line.strip().lower()), feature)
        if(flag==1):
            text= map(lambda line: line[0:maxlen], text)
        return np.asarray(text)

    name = cut_len(raw_features[:, 0], 1000,0)
    aff = cut_len(raw_features[:,1], 100,1)
    edu = cut_len(raw_features[:,2], 1000,0)
    pub = cut_len(raw_features[:,3], 1000,0)
    print "cut completed!"

    '''
    data_size = len(network)
    ind = np.random.permutation(data_size)

    n_test = int(FLAGS.test_sample_percentage * data_size)
    #n_valid = int(0.2 * data_size)
    test_indices = ind[:n_test]
    #valid_indices = ind[n_test:n_test + n_valid]
    train_indices = ind[n_test:]

    train_network = network[train_indices]
    test_network = network[test_indices]


    #pickle.dump(train_network, open("train_network.p", "wb"))
    #pickle.dump(test_network, open("test_network.p", "wb"))
    #valid_data = network[valid_indices]
    '''
    print "train",len(train_network)
    print "test",len(test_network)
    
    train = DatasetProcessor(name, aff, edu, pub, vocabulary, charvaca, pairs, train_network, FLAGS.max_neighbor_size, FLAGS.batch_size)

    test = DatasetProcessor(name, aff, edu, pub, vocabulary, charvaca, pairs, test_network, FLAGS.max_neighbor_size, FLAGS.batch_size)

    print('data loaded.')

    with tf.Graph().as_default():
        tf_config = tf.ConfigProto(allow_soft_placement=True,
                                   log_device_placement=False)
        tf_config.gpu_options.allow_growth = True

        sess = tf.Session(config=tf_config)
        with sess.as_default():
            # with tf.device('/cpu:0'):
            model = cnn.textBiLSTM(train_set = train,
                                   test_set = test,
                                   num_classes = FLAGS.class_num,
                                   word_vocab_size = len(vocabulary),
                                   word_embedd_dim = FLAGS.word_embedding_dim,
                                   char_vocab_size = len(charvaca),
                                   char_embedd_dim = FLAGS.char_embedding_dim,
                                   l2_reg_lambda = FLAGS.l2_reg_lambda,
                                   max_neighbor_size = FLAGS.max_neighbor_size,
                                   first_num_filters = FLAGS.first_num_filters,
                                   first_filter_size = FLAGS.first_filter_size,
                                   second_num_filters = FLAGS.second_num_filters,
                                   second_filter_size = list(map(int, FLAGS.second_filter_size.split(","))),
                                   n_hidden = FLAGS.hidden_size)

            sess.run(tf.global_variables_initializer())
            
            model.train(sess,num_epochs = FLAGS.num_epochs, lr_init=FLAGS.learning_rate, max_decay_epoch = FLAGS.max_decay_epoch, dropout_keep_prob = FLAGS.dropout_keep_prob, evaluate_every = FLAGS.evaluate_every)
            '''            
            ckpt = tf.train.get_checkpoint_state('/home/ubuntu/daniel/mgan/aminer-linkedin/all-att/checkpoints_cikm/')
            if ckpt and ckpt.model_checkpoint_path:
            	print ckpt.model_checkpoint_path
            	model.saver.restore(sess, ckpt.model_checkpoint_path)
            	model.predict(sess, test)
            '''		

if __name__ == "__main__":
    main()
