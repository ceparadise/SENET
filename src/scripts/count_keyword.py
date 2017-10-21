from gensim.models.word2vec import Word2Vec

keywords = []
with open("../data/vocabulary.txt", encoding = "utf8") as fin:
    for line in fin:
        keywords.append(line.strip("\n\t\r"))
count = 0
model = Word2Vec.load("../data/w2v.model")
in_vocab = set()
vocab = model.wv.vocab.keys()
for key in keywords:
    if key in vocab  or key + "\n" in vocab:
        in_vocab.add(key)

print("vocab size {} , and {} keywords are in the vocab".format(len(vocab), len(in_vocab)))
