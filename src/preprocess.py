#!usr/bin/env python
from nltk import word_tokenize
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import wikipedia
from common import *


class Preprocessor:
    def __init__(self):
        self.vocab_file = os.path.join(VOCAB_DIR, "vocabulary.txt")
        self.vocab_with_wiki = os.path.join(VOCAB_DIR, "vocabulary_wiki.txt")

    def __wiki_context_extract(self, word):
        try:
            summary = wikipedia.summary(word)
            summary = summary.replace("\n", " ")
            if summary:
                return summary
            else:
                return ""
        except Exception as e:
            return ""

    def add_wiki_content(self):
        count = 0
        if os.path.isfile(self.vocab_with_wiki):
            return
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
        single string. The file path can be too long for windows machine.
        :param dir_path: The path to the directory
        :return: A string representing all files
        """

        self.add_wiki_content()
        file_paths = []
        for raw_dir in RAW_DIR_LIST:
            file_names = os.listdir(raw_dir)
            file_paths.extend([os.path.join(raw_dir, file_name) for file_name in file_names])
        file_paths.append(self.vocab_with_wiki)

        raw_txt = ''
        for file_path in file_paths:
            if not file_path.endswith("txt"):
                continue
            print("Processing " + file_path)
            try:
                with open(file_path, 'r', encoding='utf8') as input:
                    for line in input:
                        raw_txt += line.lower()
            except Exception as e:
                pass
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
        print("Writing to disk ...")
        with open(output_path, 'w', encoding='utf8') as outfile:
            for line in sentences:
                line = ' '.join(line).strip(' \n') + '\n'
                if len(line) >= 10:
                    outfile.write(line)


'''
class PUREDataExtractor:
    def __init__(self):
        xmls = [os.path.join(PURE_REQ_DIR, fname) for fname in os.listdir(PURE_REQ_DIR) if fname.endswith(".xml")]
        for xml_path in xmls:
            print("processing " + xml_path)
            tree = ET.parse(xml_path)
            txt = ""
            for element in tree.iter():
                if element.tag.endswith("title") or element.tag.endswith("text_body"):
                    if (element.text == None):
                        continue
                    txt += element.text.strip("\n\t\r") + "\n"
            txt_path = xml_path[:-len(".xml")] + ".txt"
            with open(txt_path, 'w', encoding='utf8') as of:
                of.write(txt)
'''

if __name__ == "__main__":
    print("Start ...")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("punkt")

    preprocessor = Preprocessor()
    docuemnts = preprocessor.read_all_files()
    tokens = preprocessor.tokenize(docuemnts)
    tokens = preprocessor.remove_stop_word(tokens)
    tokens = preprocessor.clean_numbers(tokens)
    tokens = preprocessor.lemmatizing(tokens)  # tokens =preprocessor.stemming(tokens)
    preprocessor.write_to_file(W2V_DIR + os.sep + "w2v.data", tokens)
    print("Finished ...")
