from common import *
import tensorflow as tf
from feature_extractors import FeatureExtractor
import numpy as np
from GoogleScraper import scrape_with_config, GoogleSearchError
import urllib
from lxml import html
from preprocess import Preprocessor
import os
from threading import Thread
import functools
from nltk import PorterStemmer
import nltk, sys


class FeatureBuilder:
    def __init__(self):
        self.data_set = []

    def timeout(timeout):
        def deco(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]

                def newFunc():
                    try:
                        res[0] = func(*args, **kwargs)
                    except Exception as e:
                        res[0] = e

                t = Thread(target=newFunc)
                t.daemon = True
                try:
                    t.start()
                    t.join(timeout)
                except Exception as je:
                    print('error starting thread')
                    raise je
                ret = res[0]
                if isinstance(ret, BaseException):
                    raise ret
                return ret

            return wrapper

        return deco

    @timeout(60)
    def get_page_content(self, link):
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers = {'User-Agent': user_agent, }
        try:
            preprocessor = Preprocessor()
            request = urllib.request.Request(link, None, headers)
            with urllib.request.urlopen(request) as url:
                html_page = url.read()
            tree = html.fromstring(html_page)
            text = tree.xpath('//p/text()')
            text = " ".join(text)
            tokens = preprocessor.tokenize(text)
            tokens = preprocessor.remove_stop_word(tokens)
            tokens = preprocessor.lemmatizing(tokens)
            clean_sents_list = preprocessor.clean_nonAlpha(tokens)
            return clean_sents_list
        except Exception as e:
            print(e)

    def search(self, dir, keyword):
        BING_STACKOVERFLOW = BING_WORD_DIR_ROOT + os.sep + "bing_stackoverflow_word"
        BING_REGULAR = BING_WORD_DIR_ROOT + os.sep + "bing_word"
        BING_SENTENCE_QUERY = BING_WORD_DIR_ROOT + os.sep + "bing_setenceQuery_word"

        if dir == BING_STACKOVERFLOW:
            query = keyword.strip("\n\t\r") + " site:stackoverflow.com definition"
        elif dir == BING_SENTENCE_QUERY:
            query = "what is {} in computer science".format(keyword.strip("\n\t\r"))
        else:
            query = keyword

        config = {
            'use_own_ip': 'True',
            'keyword': query,
            'search_engines': 'bing',
            'num_pages_for_keyword': 1,
            'scrape_method': 'http',
            'do_caching': 'True'

        }

        try:
            sqlalchemy_session = scrape_with_config(config)
        except GoogleSearchError as e:
            print(e)
        res = ""
        for serp in sqlalchemy_session.serps:
            for link in serp.links:
                doc = self.get_page_content(link.link)
                if doc:
                    for str in doc:
                        sent = " ".join(str)
                        res += " " + sent
        print("Scraping context for word:" + keyword)
        if not os.path.isfile(dir + keyword + ".txt") and len(res) > 0:
            try:
                with open(dir + os.sep + keyword + ".txt", 'w', encoding='utf8') as fout:
                    fout.write(res)
            except Exception as e:
                print(e)

        return res

    def get_features_vecs(self, pairs, write_to_file_path=""):
        try:
            if write_to_file_path != "":
                fin = open(write_to_file_path, 'w')

            for i, pair in enumerate(pairs):
                try:
                    print("Preparing feature vec, progress {}/{}".format(i, len(pairs)))
                    words1 = pair[0].strip(" \n")
                    words2 = pair[1].strip(" \n")
                    vector = []
                    vector.extend(self.build_feature_vector(words1, words2))
                    self.data_set.append(
                        (vector, (words1, words2)))
                    if write_to_file_path != "":
                        entry = "{},{},{}\n".format(words1, words2, ",".join(map(str, vector)))
                        fin.write(entry)
                except Exception as e:
                    print(e)
        except Exception as fun_e:
            print(fun_e)
        finally:
            if write_to_file_path != "":
                fin.close()

    def build_feature_vector(self, words1, words2):
        define1 = ""
        define2 = ""

        for dir in BING_WORD_DIR:
            try:
                with open(dir + os.sep + words1 + ".txt", encoding='utf8') as f1:
                    define1 += f1.read()
            except Exception as e:
                define1 += self.search(dir, words1)

        for dir in BING_WORD_DIR:
            try:
                with open(dir + os.sep + words2 + ".txt", encoding='utf8') as f2:
                    define2 += f2.read()
            except Exception as e:
                define2 += self.search(dir, words2)
        return FeatureExtractor().get_feature(words1, define1, words2, define2)


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
        tokens = phrase.split(" ")
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


class PairBuilder:
    def __read_words(self, file_path):
        res = set()
        with open(file_path, encoding='utf8') as fin:
            for line in fin.readlines():
                phrase = line.strip("\n\t\r ")
                res.add(phrase)
        return list(res)

    def __get_all_relationships(self):
        constras = os.path.join(VOCAB_DIR, "contrast.txt")
        hyper = os.path.join(VOCAB_DIR, "hyper.txt")
        related = os.path.join(VOCAB_DIR, "related.txt")
        synonym = os.path.join(VOCAB_DIR, "synonym.txt")
        one_pair_in_line = [constras, related, synonym]
        multi_pair_in_line = [hyper]
        rel = set()
        for f in one_pair_in_line:
            with open(f) as fin:
                for line in fin.readlines():
                    line = line.strip("\n\t\r ")
                    word_pair = line.split(",")
                    relation = (word_pair[0], word_pair[1])
                    rel.add(relation)
        for f in multi_pair_in_line:
            with open(f) as fin:
                for line in fin.readlines():
                    line = line.strip("\n\t\r ")
                    hyper, rest = line.split(":")
                    for w_r in rest.split(","):
                        rel.add((hyper[0], w_r))
        return rel

    def __init__(self, expension_list_txt):
        self.exp_list = self.__read_words(expension_list_txt)
        self.relations = self.__get_all_relationships()

    def get_pairs(self):
        pairs = []
        vocab = self.__read_words(os.path.join(VOCAB_DIR, "vocabulary.txt"))
        for w_v in vocab:
            for w_e in self.exp_list:
                if (w_v, w_e) not in self.relations and (w_e, w_v) not in self.relations:
                    pairs.append((w_v, w_e))
        return pairs


if __name__ == "__main__":
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

    pair_builder = PairBuilder(os.path.join(DATA_DIR, "dataset", "requirement_extension_vocab.txt"))
    pairs = pair_builder.get_pairs()
    work_size = len(pairs)
    chunk_siz = int(work_size / total_partition_num)
    work_partition_start = (partition_num - 1) * chunk_siz
    work_partition_end = min(partition_num * chunk_siz, work_size)
    pairs = pairs[work_partition_start: work_partition_end]
    print("Total pairs to process:{}, the working interval for partition {} is from {} to {}".format(
        work_size, partition_num, work_partition_start, work_partition_end))
    print("Start building feature vectors ...")
    fb = FeatureBuilder()
    feature_vec_backup_file = os.path.join(DATA_DIR, "dataset", "fv_backup", "fv_{}".format(partition_num))
    fb.get_features_vecs(pairs, feature_vec_backup_file)
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
    for i in range(len(pairs)):
        if hu_res[i] == 'yes':
            res.append('yes-h')
        else:
            res.append(rnn_res[i] + "-m")
    print("Writing result to disk ...")
    with open(os.path.join(RESULT_DIR, "extension_res_{}.text".format(partition_num)), 'w', encoding='utf8') as fout:
        for i in range(len(pairs)):
            words = pairs[i]
            w1 = words[0]
            w2 = words[1]
            label = res[i]
            fout.write(",".join([w1, w2, label]) + "\n")
    print("Finished")
