import os
import random
import re
import numpy as np
from common import *
from nltk.stem.porter import PorterStemmer

from feature_extractors import FeatureExtractor


class DataPrepare:
    def __init__(self, p2v_model, fake=False):
        self.p2v_model = p2v_model
        self.keyword_path = VOCAB_DIR + os.sep + "vocabulary.txt"
        self.keys = []
        with open(self.keyword_path, 'r', encoding='utf-8') as kwin:
            for line in kwin:
                self.keys.append(line.strip(" \n\r\t"))

        self.golden_pair_files = ["synonym.txt", "contrast.txt", "related.txt"]
        golden_pairs = self.build_golden()
        self.data_set = []
        neg_pairs = self.build_neg_with_random_pair(golden_pairs)
        labels = [[0., 1.], [1., 0.]]
        print("Candidate neg pairs:{}, Golden pairs:{}".format(len(neg_pairs), len(golden_pairs)))
        cnt_n = cnt_p = 0
        for i, plist in enumerate([neg_pairs, golden_pairs]):
            label = labels[i]
            for pair in plist:
                try:
                    words1 = pair[0].strip(" \n")
                    words2 = pair[1].strip(" \n")
                    vector = []
                    vector.extend(self.build_feature_vector(words1, words2))
                    self.data_set.append(
                        (vector, label, (words1, words2)))  # This will be parsed by next_batch() in dataset object
                    if i == 0:
                        cnt_n += 1
                    else:
                        cnt_p += 1
                except Exception as e:
                    print(e)
        print("Negative pairs:{} Golden Pairs:{}".format(cnt_n, cnt_p))
        random.shuffle(self.data_set)
        # self.write_file()

    def write_file(self):
        with open(DATA_DIR + os.sep + "dataset.data", encoding='utf8') as fout:
            for entry in self.data_set:
                fout.write(entry)

    def build_neg_with_w2v(self, golden_pairs):
        neg_pairs = []
        for word in self.keys:
            phrase = self.words2phrase(word)
            close_phrases = self.p2v_model.w2v_model.most_similar(phrase, topn=20)
            for close_phrase in close_phrases:
                close_word = self.phrase2words(close_phrase)
                if (word, close_word) not in golden_pairs:
                    neg_pairs.append((word, close_word))
        return neg_pairs

    def build_neg_with_random_pair(self, golden_pairs):
        def get_random_word(gold, num):
            res = []
            cnt = 0
            key_size = len(self.keys)
            while cnt < num:
                neg_pair = (gold, self.keys[random.randint(0, key_size - 1)])
                neg_verse = (neg_pair[1], neg_pair[0])
                if neg_pair not in golden_pairs and neg_verse not in golden_pairs:
                    res.append(neg_pair)
                    cnt += 1
            return res

        neg_pairs = []
        for pair in golden_pairs:
            try:
                g1_negs = get_random_word(pair[0], 1)
                # g2_negs = get_random_word(pair[1], 3)
                neg_pairs.extend(g1_negs)
                # neg_pairs.extend(g2_negs)
            except Exception as e:
                pass
        return neg_pairs

    def build_golden(self):
        pair_set = set()
        for g_pair_name in self.golden_pair_files:
            path = VOCAB_DIR + os.sep + g_pair_name
            with open(path, encoding='utf8') as fin:
                for line in fin.readlines():
                    words1, words2 = line.strip(" \n").split(",")
                    pair_set.add((words1, words2))

        with open(VOCAB_DIR + os.sep + "hyper.txt") as fin:
            for line in fin.readlines():
                words1, rest = line.strip(" \n").split(":")
                for word in rest.strip(" ").split(","):
                    pair_set.add((words1.strip(" \n"), word.strip("\n")))
        print("Golden pair number:{}".format(len(pair_set)))
        pair_set = self.remove_pair_with_same_pre_post(pair_set)
        return pair_set

    def remove_pair_with_same_pre_post(self, pair_set):
        def __stem_Tokens(words):
            porter_stemmer = PorterStemmer()
            return [porter_stemmer.stem(x) for x in words.split(" ")]

        cnt = 0
        filtered = []
        for p in pair_set:
            w1 = __stem_Tokens(p[0])
            w2 = __stem_Tokens(p[1])
            flag = False
            for tk in w1:
                if tk in w2:
                    flag = True
            if flag:
                cnt += 1
                continue
            filtered.append(p)
        print("Totally {} pairs have been removed".format(cnt))
        return filtered

    def words2phrase(self, words):
        return words.replace(" ", "_")

    def phrase2words(self, phrase):
        return phrase.replace("_", " ")

    def clean_word(self, word):
        word = re.sub(r'\([^)]*\)', '', word)
        tokens = word.split(" ")
        tokens = [token.lower() for token in tokens if len(token) > 0 and not token.isupper()]
        return " ".join(tokens)

    def build_feature_vector(self, words1, words2):
        """
        :return:
        """
        define1 = ""
        define2 = ""
        try:
            with open(BING_WORD_DIR + os.sep + words1 + ".txt", encoding='utf8') as f1:
                define1 = f1.read()
        except Exception as e:
            print(e)

        try:
            with open(BING_WORD_DIR + os.sep + words2 + ".txt", encoding='utf8') as f2:
                define2 = f2.read()
        except Exception as e:
            print(e)

        return FeatureExtractor().get_feature(words1, define1, words2, define2)

    def get_vec_length(self):
        first = self.data_set[0][0]
        return len(first)

    def ten_fold(self):
        train_test_pair = []
        folds = []
        slice_size = int(len(self.data_set) / 10)
        if slice_size == 0:
            raise Exception("Not enough data to do 10 fold")
        start_cut_index = 0;
        for i in range(0, 10):
            end_cut_index = min(start_cut_index + slice_size, len(self.data_set))
            folds.append(self.data_set[start_cut_index: end_cut_index])
            start_cut_index = end_cut_index

        for i in range(0, 10):
            test_entries = folds[i]
            train_entries = []
            for fd in folds[:i]:
                train_entries.extend(fd)
            for fd in folds[i + 1:]:
                train_entries.extend(fd)

            train_set = DataSet(train_entries)
            test_set = DataSet(test_entries)
            train_test_pair.append((train_set, test_set))
        return train_test_pair


class DataSet:
    def __init__(self, entry_list):
        self.cur_batch_start = 0
        self.data = entry_list

    def next_batch(self, batch_size):
        """
        Get next batch of the data. If the times of requesting new batch larger than the dataset,
        shuffle the dataset and do it again
        :param batch_size:
        :return:
        """
        start = self.cur_batch_start
        self.cur_batch_start += batch_size
        if self.cur_batch_start > len(self.data):
            random.shuffle(self.data)
            start = 0
            self.cur_batch_start = batch_size
            assert batch_size <= len(self.data)
        end = self.cur_batch_start
        batch_data = self.data[start:end]
        # Provide the vector, label and the readable words
        return np.array([x[0] for x in batch_data]), np.array([x[1] for x in batch_data]), [x[2] for x in batch_data]

    def all(self):
        return np.array([x[0] for x in self.data]), np.array([x[1] for x in self.data]), [x[2] for x in self.data]
