from nltk.stem.porter import PorterStemmer


class FeatureExtractor:
    def __init__(self):
        self.func_pip = [self.common_token_len,
                         self.same_postfix,
                         self.same_prefix
                         ]

    def __stem_Tokens(self, words):
        porter_stemmer = PorterStemmer()
        return [porter_stemmer.stem(x) for x in words.split(" ")]

    def common_token_len(self, w1, d1, w2, d2):
        """
        Number of common tokens
        :return:
        """
        d1_tk = set(self.__stem_Tokens(w1))
        d2_tk = set(self.__stem_Tokens(w2))
        common_len = len(d1_tk.intersection(d2_tk))
        return common_len

    def same_prefix(self, w1, d1, w2, d2):
        d1_tk = self.__stem_Tokens(w1)
        d2_tk = self.__stem_Tokens(w2)
        if d1_tk[0] == d2_tk[0]:
            return 1
        else:
            return 0

    def same_postfix(self, w1, d1, w2, d2):
        d1_tk = self.__stem_Tokens(w1)
        d2_tk = self.__stem_Tokens(w2)
        if d1_tk[-1] == d2_tk[-1]:
            return 1
        else:
            return 0

    def get_feature(self, word1, define1, words2, define2):
        feature_vec = []
        for func in self.func_pip:
            feature_vec.append(func(word1, define1, words2, define2))
        return feature_vec
