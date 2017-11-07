"""
Convert the collected files into definition
"""
from common import *
import os
from lxml import html

from preprocess import Preprocessor


def get_word_from_fName(fName):
    parts = fname.split("_")[:-1]
    return " ".join(parts)


fileNames = os.listdir(GOOGLE_DOWNLOAD)
word_dict = dict()
preprocessor = Preprocessor()
for fname in fileNames:
    print(fname)
    fpath = os.path.join(GOOGLE_DOWNLOAD, fname)
    word = get_word_from_fName(fname)
    with open(fpath, 'r', encoding="utf-8") as fin:
        text = []
        try:
            tree = html.fromstring(fin.read())
            text = tree.xpath('//p/text()')
        except Exception as e:
            pass
        text = " ".join(text)
        tokens = preprocessor.tokenize(text)
        tokens = preprocessor.remove_stop_word(tokens)
        clean_sents_list = preprocessor.clean_nonAlpha(tokens)
        if word not in word_dict.keys():
            word_dict[word] = []
        word_dict[word].extend(clean_sents_list)

def_path = os.path.join(VOCAB_DIR, "definition.txt")
with open(def_path, 'w', encoding='utf8') as fout:
    for word in word_dict.keys():
        str = ""
        for sent in word_dict[word]:
            str += " ".join(sent) + " "
        fout.write(word + "\t" + str + "\n")
