import tensorflow as tf
from common import *
import numpy as np


class RNN:
    def __init__(self, vec_len):
        self.lr = 0.001  # learning rate
        self.training_iters = 4000  # 100000  # train step upper bound
        self.batch_size = 1
        self.n_inputs = vec_len  # MNIST data input (img shape: 28*28)
        self.n_steps = 1  # time steps
        self.n_hidden_units = 128  # neurons in hidden layer
        self.n_classes = 2  # classes (0/1 digits)

    def train_test(self, data):
        # x y placeholder
        x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        y = tf.placeholder(tf.float32, [None, self.n_classes])

        weights = {
            # shape ( vec_len, 128)
            'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            # shape (128, 2)
            'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        }
        biases = {
            # shape (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            # shape (2, )
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
        }

        with tf.Session() as sess:
            pred = self.classify(x, weights, biases)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
            train_op = tf.train.AdamOptimizer(self.lr).minimize(cost)
            pred_label_index = tf.argmax(pred, 1)  # Since we use one-hot represent the predicted label, index = label
            correct_pred = tf.equal(pred_label_index, tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            init = tf.global_variables_initializer()

            a_acc = a_recall = a_pre = a_f1 = 0
            result_file = RESULT_DIR + os.sep + "RNN_result{}.txt".format(len(os.listdir(RESULT_DIR)))
            with open(result_file, "w", encoding='utf8') as fout:
                for index, (train_set, test_set) in enumerate(data.ten_fold()):
                    print("Start fold {}".format(index))
                    sess.run(init)
                    step = 0
                    while step * self.batch_size < self.training_iters:
                        batch_xs, batch_ys, train_word_pairs = train_set.next_batch(self.batch_size)
                        batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
                        sess.run([train_op], feed_dict={
                            x: batch_xs,
                            y: batch_ys,
                        })
                        step += 1

                    print("Start testing...")
                    res = []
                    print(len(test_set.data))
                    for i in range(len(test_set.data)):
                        batch_xs, batch_ys, test_word_pairs = test_set.next_batch(self.batch_size)
                        batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
                        is_correct = sess.run(correct_pred, feed_dict={x: batch_xs, y: batch_ys})
                        res.append((batch_ys, is_correct, test_word_pairs))
                    re, pre, f1, accuracy = self.eval(res)
                    self.write_res(res, fout)
                    a_recall += re
                    a_pre += pre
                    a_f1 += f1
                    a_acc += accuracy
                avg_str = "Average recall:{}, precision:{}, f1:{}, accuracy:{}".format(a_recall / 10, a_pre / 10,
                                                                                       a_f1 / 10, a_acc / 10)
                fout.write(avg_str)
                print(avg_str)

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

    def eval(self, results):
        tn = 0
        tp = 0
        fn = 0
        fp = 0
        for label, correctness, word_pairs in results:
            if label[0][0] == 1:  # positive
                if correctness[0]:
                    tp += 1
                else:
                    fp += 1
            else:
                if correctness[0]:
                    tn += 1
                else:
                    fn += 1

        if tp + fn == 0:
            recall = 1
        else:
            recall = tp / (tp + fn)

        accuracy = (tp + tn) / (tp + tn + fn + fp)

        if tp + fp == 0:
            precision = 1
        else:
            precision = tp / (tp + fp)
        f1 = 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        print("True Negative:{}, True Positive:{}, False Negative:{}, False Positive:{}".format(tn, tp, fn, fp))
        print("recall: {}".format(recall))
        print("precision: {}".format(precision))
        print("f1:{}".format(f1))
        print("accuracy:{}".format(accuracy))
        return recall, precision, f1, accuracy

    def write_res(self, res, writer):
        writer.write("label, correctness, w1, w2\n")
        for label, correctness, word_pairs in res:
            tran_label = "Yes"
            if np.argmax(label.ravel()) == 1:
                tran_label = "No";
            correct_output = 'Incorrect'

            if correctness[0] == True:
                correct_output = 'Correct'

            res_str = "{}\t{}\t{}\t\t{}".format(tran_label, correct_output, word_pairs[0][0], word_pairs[0][1])
            writer.write(res_str + "\n")
