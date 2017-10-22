from gensim.models.word2vec import Word2Vec
from nltk.stem.wordnet import WordNetLemmatizer

def lemmatizing(tokens):
    lmtzr = WordNetLemmatizer()
    lem_sentences = []
    for sentence_token in tokens:
        tmp = []
        for token in sentence_token:
            tmp.append(lmtzr.lemmatize(token))
        lem_sentences.append(tmp)
    return lem_sentences

keywords = set()
wd = set()
word_files = ['contrast.txt', 'related.txt', 'synonym.txt']
with open("../../data/vocabulary.txt", encoding="utf8") as fin:
    for line in fin:
        wd.add(line.strip(" \n\t\r"))
    print(len(wd))
for word_file in word_files:
    with open("../../data/" + word_file, encoding="utf8") as fin:
        for line in fin:
            line = line.strip("\n\t\r")
            words = line.split(",")
            keywords.add(words[0])
            keywords.add(words[1])

with open("../../data/hyper.txt", encoding="utf8") as fin:
    for line in fin:
        parent, rest = line.split(":")
        keywords.add(parent)
        rest_words = rest.split(",")
        for words in rest_words:
            keywords.add(parent)

model = Word2Vec.load("../../data/w2v.model")
in_vocab = set()
un_vocab = set()
vocab = model.wv.vocab.keys()
for key in keywords:
    key = key.lower()
    tokens = key.strip(' \n\t').split()
    tokens = lemmatizing(tokens)
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
