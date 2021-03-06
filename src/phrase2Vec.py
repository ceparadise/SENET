from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases
from common import  W2V_DIR
import os


class Phrase2Vec:
    def __init__(self, force_build=False):
        print("Force rebuild = {}", force_build)

        self.model_path = W2V_DIR + os.sep + "w2v.model";
        self.training_data_path = W2V_DIR + os.sep + "w2v.data"

        # If file not exists, train a new one
        if not os.path.isfile(self.model_path) or force_build == True:
            with open(self.training_data_path, encoding='utf-8') as f:
                sentences = f.readlines()
                sentence_stream = [sentence.split(" ") for sentence in sentences]
                phrasedTxt = Phrases(sentence_stream, min_count=1, threshold=3)
                self.w2v_model = Word2Vec(phrasedTxt[sentence_stream], size=100, window=5, min_count=2, workers=4)
                self.w2v_model.save(self.model_path)
        self.w2v_model = Word2Vec.load(self.model_path)


if __name__ == "__main__":
    print("Building the word2vec ...")
    p2v = Phrase2Vec(force_build=True)
    print("Finish building")
