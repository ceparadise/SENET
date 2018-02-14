"""
Clean the vocabulary file
"""
from common import *
import re

vocab_file = VOCAB_DIR + os.sep + "vocabulary_origin.txt"
vocab_clean = VOCAB_DIR + os.sep + "vocabulary.txt"

with open(vocab_file, 'r', encoding='utf8') as fin, open(vocab_clean, 'w', encoding='utf8') as fout:
    for line in fin:
        line = line.strip("\n\t\r ")
        line = re.sub('\(.*?\)', '', line)
        line = re.sub("\s+"," ", line)
        line = re.sub("\s*-\s*","-", line)
        tokens = line.split(" ")
        tokens = [x for x in tokens if len(x) > 0]
        word = " ".join(tokens)
        # Remove all acronym
        if word.isupper():
            continue
        fout.write(word + "\n")
print("Cleaned ...")