import os
import random
import re
import numpy as np
from common import VOCAB_DIR

from feature_extractors import FeatureExtractor


class DataPrepare:
    def __init__(self, p2v_model, fake=False):
        self.p2v_model = p2v_model
        self.keyword_path = VOCAB_DIR + os.sep + "vocabulary.txt"
        self.golden_pair_files = ["synonym.txt", "contrast.txt", "related.txt"]
        golden_pairs = self.build_golden()
        self.data_set = []
        pairs = self.build_pairs()

        for pair in pairs:
            try:
                words1 = pair[0]
                words2 = pair[1]
                phrase1 = self.words2phrase(words1)
                phrase2 = self.words2phrase(words2)
                p1_vec = self.p2v_model.w2v_model[pair[0]]
                p2_vec = self.p2v_model.w2v_model[pair[1]]
                similarity = self.p2v_model.w2v_model.similarity(phrase1, phrase2)
                vector = []
                vector.extend(p1_vec)
                vector.extend(p2_vec)
                vector.append(similarity)
                vector.extend(self.build_feature_vector(words1, words2))
                if pair in golden_pairs:
                    label = [1., 0.]
                else:
                    label = [0., 1.]
                self.data_set.append((vector, label))
            except KeyError as e:
                pass
                # print(e)

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
                    pair_set.add((words1, word))
        return pair_set

    def build_pairs(self):
        """
        for each keyword, find its closest N phrases and produce N pairs.
        :return: The binned N pairs
        """
        pairs = set()
        with open(self.keyword_path, encoding='utf8') as kf:
            for words in kf.readlines():
                words = words.strip(" \n")
                cleaned_words = self.clean_word(words)
                if len(cleaned_words) == 0:
                    continue
                # Transfer the words into a phrase eg. new york -> new_york, since p2vmodel use this format
                phrase = self.words2phrase(cleaned_words)
                try:
                    close_phrases = self.p2v_model.w2v_model.most_similar(phrase, topn=50)
                    for close_phrase in close_phrases:
                        close_words = self.phrase2words(close_phrase[0])
                        pairs.add((cleaned_words, close_words))
                except KeyError:
                    pass
                    # print("key error for " + phrase + " | " + words)
        return pairs

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
        return FeatureExtractor().get_feature(words1, define1, words2, define2)

    def get_vec_length(self):
        first = self.data_set[0][0]
        return len(first)

    def ten_fold(self):
        train_test_pair = []
        test_size = int(len(self.data_set) / 5)
        for i in range(1, 10):
            random.shuffle(self.data_set)
            test_entries = self.data_set[:test_size]
            train_entries = self.data_set[test_size:]
            train_set = DataSet(train_entries)
            test_set = DataSet(test_entries)
            train_test_pair.append((train_set, test_set))
        return train_test_pair


class DataSet:
    def __init__(self, entry_list):
        self.cur_batch_start = 0
        self.data = entry_list

    def next_batch(self, batch_size):
        start = self.cur_batch_start
        self.cur_batch_start += batch_size
        if self.cur_batch_start > len(self.data):
            random.shuffle(self.data)
            start = 0
            self.cur_batch_start = batch_size
            assert batch_size <= len(self.data)
        end = self.cur_batch_start
        batch_data = self.data[start:end]
        return np.array([x[0] for x in batch_data]), np.array([x[1] for x in batch_data])


