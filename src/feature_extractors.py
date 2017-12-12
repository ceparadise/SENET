from nltk.stem.porter import PorterStemmer


class FeatureExtractor:
    def __init__(self):
        self.func_pip = [self.common_token_len,
                         self.same_postfix,
                         self.same_prefix,
                         self.token_num_same,
                         self.include_each_other
                         ]

    def __stem_Tokens(self, words):
        porter_stemmer = PorterStemmer()
        return [porter_stemmer.stem(x) for x in words.split(" ")]

    def common_token_len(self, w1, d1, w2, d2):
        """
        Number of common tokens
        :return:
        """
        w1_tk = set(self.__stem_Tokens(w1))
        w2_tk = set(self.__stem_Tokens(w2))
        common_len = len(w1_tk.intersection(w2_tk))
        return common_len

    def same_prefix(self, w1, d1, w2, d2):
        w1_tk = self.__stem_Tokens(w1)
        w2_tk = self.__stem_Tokens(w2)
        if w1_tk[0] == w2_tk[0]:
            return 1
        else:
            return 0

    def same_postfix(self, w1, d1, w2, d2):
        w1_tk = self.__stem_Tokens(w1)
        w2_tk = self.__stem_Tokens(w2)
        if w1_tk[-1] == w2_tk[-1]:
            return 1
        else:
            return 0

    def get_feature(self, word1, define1, words2, define2):
        feature_vec = []
        for func in self.func_pip:
            feature_vec.append(func(word1, define1, words2, define2))
        return feature_vec

    def token_num_same(self, w1, d1, w2, d2):
        # Check if two words have same length
        d1_tk = self.__stem_Tokens(w1)
        d2_tk = self.__stem_Tokens(w2)
        return len(d1_tk) == len(d2_tk)

    def include_each_other(self, w1, d1, w2, d2):
        # Count how many time each word appear on each other's definition
        return d1.count(w2) + d2.count(w1)
