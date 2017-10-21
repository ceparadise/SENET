from phrase2Vec import Phrase2Vec
from data_prepare import DataPrepare
from RNN import RNN

"""
1. phrase2vec train model if not exist
2. Iter keyword build pairs with top k threshold
3. for each pair, find the definiton
4. Build features set for the pairs
5. Turn each pair+features to vector as X
6. Add label to each vector as Y
7. Write the vectors to a file.
"""
if __name__ == '__main__':
    print("Preparing phrase2vec model...")
    p2v_model = Phrase2Vec()
    print("Phrase2Vector model loaded...")
    data = DataPrepare(p2v_model)
    print("Experiment data is ready, size ", len(data.data_set))

    #for train_set, test_set in data_prepare.ten_fold():
    rnn = RNN(data.get_vec_length())
    rnn.train_test(data)
