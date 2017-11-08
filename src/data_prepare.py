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
        neg_pairs = []
        for pair in golden_pairs:
            try:
                words1 = pair[0].stirp(" \n")
                words2 = pair[1].stirp(" \n")
                phrase1 = self.words2phrase(words1)
                phrase2 = self.words2phrase(words2)
                close_phrases1 = self.p2v_model.w2v_model.most_similar(phrase1, topn=3)
                close_phrases2 = self.p2v_model.w2v_model.most_similar(phrase2, topn=3)
                for close_phrase1, close_phrase2 in zip(close_phrases1, close_phrases2):
                    close_words1 = self.phrase2words(close_phrase1[0])
                    close_words2 = self.phrase2words(close_phrase2[0])
                    if close_words1 != phrase2:
                        neg_pairs.append((phrase1, close_words1))
                    if close_words2 != phrase1:
                        neg_pairs.append((phrase2, close_words1))
            except Exception:
                pass

        labels = [[0., 1.], [1., 0.]]
        for i, plist in enumerate([neg_pairs, golden_pairs]):
            label = labels[i]
            for pair in plist:
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
                    self.data_set.append(
                        (vector, label, (words1, words2)))  # This will be parsed by next_batch() in dataset object
                except KeyError as e:
                    pass
        random.shuffle([(x[1], x[2]) for x in self.data_set])
        print(self.data_set)

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
        return pair_set

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
