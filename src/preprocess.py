#!usr/bin/env python
from nltk import word_tokenize
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import wikipedia
from wikipedia.exceptions import DisambiguationError
from wikipedia.exceptions import PageError


class Preprocessor:
    def __init__(self, dir_path):
        self.dir_paht = dir_path
        self.vocab_file = os.path.join(dir_path, "vocabulary.txt")
        self.vocab_with_wiki = os.path.join(dir_path, "vocabulary_wiki.txt")
        self.w2v_raw_dir = os.path.join(data_dir, "SE_Books_txt")

    def __wiki_context_extract(self, word):
        try:
            summary = wikipedia.summary(word)
            summary = summary.replace("\n" , " ")
            if summary:
                return summary
            else:
                return ""
        except Exception as e:
            return ""

    def add_wiki_content(self):
        count = 0
        with open(self.vocab_file, encoding="utf8") as fin, open(self.vocab_with_wiki, 'w', encoding="utf8") as fout:
            for line in fin:
                word = line.strip("\n\t\r")
                print("processing {} ...".format(word))
                wiki_doc = self.__wiki_context_extract(word)
                if len(wiki_doc) > 0:
                    count += 1
                fout.write("{}\t{}\n".format(word, wiki_doc))
        print("{} word have wiki doc associated".format(count))

    def read_all_files(self):
        """
        Read all files in a directory and make the content in each file to be lowercase. All files will be merged to a
        single string
        :param dir_path: The path to the directory
        :return: A string representing all files
        """
        self.add_wiki_content()
        file_names = os.listdir(self.w2v_raw_dir)
        file_paths = [os.path.join(self.w2v_raw_dir, file_name) for file_name in file_names]
        file_paths.append(self.vocab_with_wiki)
        raw_txt = ''
        for file_path in file_paths:
            if not file_path.endswith("txt"):
                continue
            print("Processing " + file_path)
            with open(file_path, 'r', encoding='utf8') as input:
                for line in input:
                    raw_txt += line.lower()
        return raw_txt

    def tokenize(self, raw_txt):
        """
        Tokenize a string into a list
        :param raw_txt: A string
        :return: A list of tokens
        """
        raw_txt = sent_tokenize(raw_txt)
        tokenized_sentences = []
        for sentence in raw_txt:
            sentence_token = word_tokenize(sentence)
            tokenized_sentences.append(sentence_token)
        return tokenized_sentences

    def clean_numbers(self, tokenized_sentences):
        # clean numbers and puncutations
        clean_numbers_sentences = []
        for sentence_token in tokenized_sentences:
            temp = []
            for token in sentence_token:
                if token.isalpha():
                    temp.append(token)
            clean_numbers_sentences.append(temp)
        return clean_numbers_sentences

    def stemming(self, tokenized_sentences):
        porter_stemmer = PorterStemmer()
        stem_sentences = []
        for sentence_token in tokenized_sentences:
            tmp = []
            for token in sentence_token:
                tmp.append(porter_stemmer.stem(token))
            stem_sentences.append(tmp)
        return stem_sentences

    def lemmatizing(self, tokenized_sentences):
        lmtzr = WordNetLemmatizer()
        lem_sentences = []
        for sentence_token in tokenized_sentences:
            tmp = []
            for token in sentence_token:
                tmp.append(lmtzr.lemmatize(token))
            lem_sentences.append(tmp)
        return lem_sentences

    def remove_stop_word(self, tokenized_sentences):
        """
        Remove stop words from a list of tokens
        :param tokenized_sentences: A list of tokens
        :return: A list of tokens with out stop words
        """
        stop = set(stopwords.words('english'))
        re_stop_sentences = []
        for sentence_token in tokenized_sentences:
            temp = []
            for token in sentence_token:
                if token not in stop:
                    temp.append(token)
            re_stop_sentences.append(temp)
        return re_stop_sentences

    def write_to_file(self, output_path, sentences):
        with open(output_path, 'w', encoding='utf8') as outfile:
            for line in sentences:
                line = ' '.join(line).strip(' \n') + '\n'
                if len(line) >= 10:
                    outfile.write(line)


if __name__ == "__main__":
    print("Start ...")
    nltk.download("stopwords")
    nltk.download("wordnet")
    data_dir = os.path.abspath(os.pardir) + os.sep + "data" + os.sep
    preprocessor = Preprocessor(data_dir)
    docuemnts = preprocessor.read_all_files()
    tokens = preprocessor.tokenize(docuemnts)
    tokens = preprocessor.remove_stop_word(tokens)
    tokens = preprocessor.clean_numbers(tokens)
    tokens = preprocessor.lemmatizing(tokens)  # tokens =preprocessor.stemming(tokens)
    preprocessor.write_to_file(data_dir + "w2v.data", tokens)
    print("Finished ...")
