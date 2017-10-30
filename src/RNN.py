import tensorflow as tf

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
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            init = tf.global_variables_initializer()

            for index, (train_set, test_set) in enumerate(data.ten_fold()):
                print("Start fold {}".format(index))

                sess.run(init)
                step = 0
                while step * self.batch_size < self.training_iters:
                    batch_xs, batch_ys = train_set.next_batch(self.batch_size)
                    batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
                    sess.run([train_op], feed_dict={
                        x: batch_xs,
                        y: batch_ys,
                    })
                    if step % 2000 == 0:
                        print("Training step {}:".format(step))
                        print(sess.run(accuracy, feed_dict={
                            x: batch_xs,
                            y: batch_ys,
                        }))
                    step += 1

                print("Start testing...")
                res = []
                print(len(test_set.data))
                for i in range(len(test_set.data)):
                    batch_xs, batch_ys = train_set.next_batch(self.batch_size)
                    batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
                    is_correct = sess.run(correct_pred, feed_dict={x: batch_xs, y: batch_ys})
                    res.append((batch_ys, is_correct))
                print(res)
                self.eval(res)
                break

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
        for label, correctness in results:
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

        print("True Negative:{}, True Positive:{}, False Negative:{}, False Negative:{}".format(tn, tp, fn, tp))
        print("recall: {}".format(recall))
        print("accuracy: {}".format(accuracy))
        print("precision: {}".format(precision))
