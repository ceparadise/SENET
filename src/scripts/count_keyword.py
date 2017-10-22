from gensim.models.word2vec import Word2Vec

keywords = []
with open("../../data/vocabulary.txt", encoding="utf8") as fin:
    for line in fin:
        keywords.append(line.strip("\n\t\r"))
count = 0
model = Word2Vec.load("../../data/w2v.model")
in_vocab = set()
un_vocab = set()
vocab = model.wv.vocab.keys()
for key in keywords:
    key = key.strip(' \n\t').replace(' ', '_')
    if key in vocab or key + "\n" in vocab:
        in_vocab.add(key)
        count += 1
    else:
        un_vocab.add(key)

with open("./appearedWords.txt", 'w') as out:
    for word in in_vocab:
        out.write(word + "\n")

with open("./notAppearedWords.txt", "w") as out:
    for word in un_vocab:
        out.write(word + "\n")

print("vocab size {} , and {} keywords are in the vocab".format(len(vocab), len(in_vocab), count))
