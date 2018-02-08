from common import *
from senet_builder_feature_vec import FeatureBuilder
import tensorflow as tf
import numpy as np
import os
import nltk, sys


class RNNModel:
    '''
    Load existing model and classify the
    '''

    def classify(self, X, weights, biases):
        X = tf.reshape(X, [-1, self.n_inputs])
        X_in = tf.matmul(X, weights['in']) + biases['in']
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])
        cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        results = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return results

    def __init__(self, vec_len, model_path):
        self.lr = 0.001  # learning rate
        self.training_iters = 4000  # 100000  # train step upper bound
        self.batch_size = 1
        self.n_inputs = vec_len  # MNIST data input (img shape: 28*28)
        self.n_steps = 1  # time steps
        self.n_hidden_units = 128  # neurons in hidden layer
        self.n_classes = 2  # classes (0/1 digits)

    def get_result(self, feature_vecs):
        result = []
        x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        weights = {
            'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units]), name='w_in'),
            'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]), name='w_out')
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ]), name='b_in'),
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]), name='b_out')
        }

        with tf.Session() as sess:
            pred = self.classify(x, weights, biases)
            pred_label_index = tf.argmax(pred, 1)  # Since we use one-hot represent the predicted label, index = label
            saver = tf.train.Saver()
            saver.restore(sess, RNNMODEL + os.sep + "rnn.ckpt")
            for i in range(len(feature_vecs)):
                batch_xs = feature_vecs[i].reshape([self.batch_size, self.n_steps, self.n_inputs])
                pre_res = sess.run(pred_label_index, feed_dict={x: batch_xs})
                result.append(pre_res)
        parsed_result = []
        for res in result:
            if res == 0:
                parsed_result.append('yes')
            else:
                parsed_result.append('no')
        return parsed_result


class Heuristics:
    def __init__(self):
        pass

    def isAcronym(self, acr, phrase):
        token1 = nltk.word_tokenize(acr)
        token2 = nltk.word_tokenize(phrase)
        if len(token1) != 1 or len(acr) != len(token2):
            return False
        for i in range(0, len(acr)):
            if acr[i].lower() != token2[i][0].lower():
                return False
        return True

    def pharse_processing(self, phrase):
        tokens = phrase.split()
        bigrams = list(zip(tokens, tokens[1:]))
        return tokens, bigrams

    def isHyper(self, w1, w2):
        tk1, bg1 = self.pharse_processing(w1)
        tk2, bg2 = self.pharse_processing(w2)
        pos_tk1 = nltk.pos_tag(tk1)
        pos_tk2 = nltk.pos_tag(tk2)

        hypernon = ""
        # Check bigrams first then check the last tokens
        if len(bg1) > 0 and len(bg2) > 0 and bg1[-1] == bg2[-1]:
            hypernon = " ".join(list(bg1[-1]))
        elif tk1[-1] == tk2[-1]:
            hypernon = tk1[-1]
        else:
            # Try to match as much tokens as possible from left to right
            length = min(len(tk1), len(tk2))
            tmp = []
            for i in range(0, length):
                if tk1[i] == tk2[i]:
                    tmp.append(pos_tk1[i])

            if len(tmp) > 0 and tmp[-1][1] == 'NN':
                tmp = [tk[0] for tk in tmp]
                hypernon = " ".join(tmp)
        if len(hypernon) > 0:
            return True
        else:
            return False

    def classify(self, word_pair):
        word1 = word_pair[0]
        word2 = word_pair[1]
        if self.isAcronym(word1, word2) or self.isAcronym(word2, word1) or self.isHyper(word1, word2) or self.isHyper(
                word2, word1):
            return 'yes'
        return 'no'


if __name__ == "__main__":
    """
    Due to memory limitation of GPU, classification can be fork as much process as the vector builder.
    Thus each classifier will process the feature vec files whose number can be mod by the classifier 
    partition num
    """
    try:
        partition_num = int(sys.argv[1])
    except:
        partition_num = 1

    try:
        total_partition_num = int(sys.argv[2])
    except:
        total_partition_num = 1
    nltk_requires = ["punkt", "averaged_perceptron_tagger"]
    for nltk_require in nltk_requires:
        nltk.download(nltk_require)

    feature_vec_dir = os.path.join(DATA_DIR, "dataset", "fv_backup")
    all_feature_vec_files = [x for x in os.listdir(feature_vec_dir) if x.startswith("fv_")]
    fv_file_for_partition = []
    for f in feature_vec_dir:
        file_num = int(f.split("_")[1])
        if file_num % partition_num == 0:
            fv_file_for_partition.append(os.path.join(feature_vec_dir, f))

    fb = FeatureBuilder()
    fb.read_feature_vecs(fv_file_for_partition)
    vec_len = len(fb.data_set[0][0])

    rnn = RNNModel(vec_len, RNNMODEL + os.sep + 'rnn.ckpt')
    hu = Heuristics()
    hu_res = []
    print("Start classification ...")
    for pair in fb.data_set:
        hu_res.append(hu.classify(pair[1]))
    rnn_res = rnn.get_result(np.array([x[0] for x in fb.data_set]))
    res = []
    print(len(hu_res), len(rnn_res))
    for i in range(len(fb.data_set)):
        if hu_res[i] == 'yes':
            res.append('yes-h')
        else:
            res.append(rnn_res[i] + "-m")
    print("Writing result to disk ...")
    with open(os.path.join(RESULT_DIR, "extension_res_{}.text".format(partition_num)), 'w', encoding='utf8') as fout:
        for i in range(len(rnn_res)):
            words = fb.data_set[i][1]
            w1 = words[0]
            w2 = words[1]
            label = res[i]
            fout.write(",".join([w1, w2, label]) + "\n")
    print("Finished")
