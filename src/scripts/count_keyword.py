from gensim.models.word2vec import Word2Vec
from nltk.stem.wordnet import WordNetLemmatizer
import os
from common import *

keywords = set()
pair_set = set()
wd = set()
lmtzr = WordNetLemmatizer()
word_files = ['contrast.txt', 'related.txt', 'synonym.txt']
with open(VOCAB_DIR + os.sep + "vocabulary.txt", encoding="utf8") as fin:
    for line in fin:
        wd.add(line.strip(" \n\t\r"))
    print(len(wd))
for word_file in word_files:
    with open(VOCAB_DIR + os.sep + word_file, encoding="utf8") as fin:
        for line in fin:
            line = line.strip("\n\t\r")
            words = line.split(",")
            keywords.add(words[0])
            keywords.add(words[1])
            pair_set.add((words[0], words[1]))

with open(VOCAB_DIR + os.sep + "hyper.txt", encoding="utf8") as fin:
    for line in fin:
        parent, rest = line.split(":")
        keywords.add(parent)
        rest_words = rest.split(",")
        for words in rest_words:
            keywords.add(words)
            pair_set.add((parent, words))
print("Distint pair #:{}".format(len(pair_set)))

model = Word2Vec.load(W2V_DIR + os.sep + "w2v.model")
in_vocab = set()
un_vocab = set()
vocab = model.wv.vocab.keys()
for key in keywords:
    key = key.lower()
    tokens = key.strip(' \n\t').split()
    tokens = [lmtzr.lemmatize(tk) for tk in tokens]
    key = "_".join(tokens)
    if key in vocab or key + "\n" in vocab:
        in_vocab.add(key)
    else:
        un_vocab.add(key)

with open("./appearedWords.txt", 'w') as out:
    for word in in_vocab:
        out.write(word + "\n")

with open("./notAppearedWords.txt", "w") as out:
    for word in un_vocab:
        out.write(word + "\n")

print("vocab size {} , Total keywords = {} and {} keywords are in the vocab".format(len(vocab), len(keywords),
                                                                                    len(in_vocab)))
