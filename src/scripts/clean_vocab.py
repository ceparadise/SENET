"""
Clean the files in vocab/origin and generate usable vocabulary files in the vocab directory
"""
import nltk
from nltk.corpus import words
from common import *
import re


class WordCleaner:
    def __init__(self):
        self.en_words = words.words()
        nltk_requires = ["words"]
        for nltk_require in nltk_requires:
            nltk.download(nltk_require)

    @staticmethod
    def to_file_name_format(word):
        """
        Convert the word to file name format which are used to retrive document in file system. Windows don't
        distinguish upper and lower cases, this cause error in linux
        :param word:
        :return:
        """
        word = WordCleaner().clean_line(word)
        word = re.sub("-", " ", word)
        return word.lower()

    def clean_line(self, line):
        line = line.strip("\n\t\r ")
        line = re.sub("[\/\\\\]+", "-", line)
        line = re.sub('\(.*?\)', '', line)
        line = re.sub('\[.*?\]', '', line)
        line = re.sub("[^\w\',\"\-&\s\:]", "", line)

        line = re.sub("\s+", " ", line)
        line = re.sub("\s*-\s*", "-", line)
        return line

    def get_clean_phrases(self, line, seperator="[:,]"):
        """
        Get the cleaned phrases from a line
        :param line:
        :param seperator
        :return: a list of phrases
        """
        cleaned_line = self.clean_line(line)
        phrases = re.split(seperator, cleaned_line)
        phrases = [p.strip("\n\t\r ") for p in phrases]
        return phrases

    def get_vocab(self, file_path, seperator="[:,]"):
        vocab = set()
        with open(file_path, encoding="utf8") as fin:
            for line in fin.readlines():
                clean_phrases = self.get_clean_phrases(line, seperator)
                for phrase in clean_phrases:
                    vocab.add(phrase)
        return vocab

    def is_acronym(self, phrase):
        """
        At least one token start with uppercase and end with uppercase
        :param phrase:
        :return:
        """
        tokens = phrase.split(" ")
        if (len(tokens) == 1):
            if len(tokens[0].split("-")) > 1:
                return False
            if tokens[0] not in self.en_words and len(tokens[0]) < 5:
                return True
        for tk in tokens:
            if re.match("^[A-Z].*[A-Z]$", tk) != None:
                return True
        return False


if __name__ == "__main__":
    relation_files = ["contrast", "hyper", "related", "synonym", "acronym"]
    wc = WordCleaner()

    all_vocab = set()
    acr_vocab = set()
    for file_name_root in relation_files:
        in_file_path = os.path.join(VOCAB_DIR, "origin", file_name_root + "_origin.txt")
        out_file_path = os.path.join(VOCAB_DIR, file_name_root + ".txt")

        all_vocab |= wc.get_vocab(in_file_path)
        with open(in_file_path, encoding="utf8") as fin, open(out_file_path, 'w', encoding='utf8') as fout:
            for line in fin.readlines():
                cleaned = wc.clean_line(line)
                fout.write(cleaned + "\n")
        if file_name_root == "acronym":
            acr_tmp_vocab = wc.get_vocab(in_file_path)
            acr_vocab_file = os.path.join(VOCAB_DIR, file_name_root + "_vocab.txt")
            for acr_ph in acr_tmp_vocab:
                if wc.is_acronym(acr_ph):
                    acr_vocab.add(acr_ph)

    vocab_file = os.path.join(VOCAB_DIR, "origin", "vocabulary_origin.txt")
    vocab_clean = os.path.join(VOCAB_DIR, "vocabulary.txt")
    all_vocab |= wc.get_vocab(vocab_file)
    with open(vocab_clean, 'w', encoding='utf8') as vocab_out, \
            open(acr_vocab_file, 'w', encoding='utf8') as arc_vocab_in:
        for ph in all_vocab:
            if wc.is_acronym(ph):
                acr_vocab.add(ph)
            else:
                vocab_out.write(ph + "\n")
        for ph in acr_vocab:
            arc_vocab_in.write(ph + "\n")

    print("Cleaned ...")
