from RNN import RNN
from FeedForwardNN import FNN
from Maxent import Maxent
from data_prepare import DataPrepare

if __name__ == "__main__":
    for i in range(0, 10):
        data = DataPrepare(remove_same_pre_post=False)
        rnn = RNN(data.get_vec_length())
        fnn = FNN()
        maxent = Maxent()
        rnn.train_test(data, half_seen=False)
        fnn.train_test(data, half_seen=False)
        maxent.run(data, half_seen=False)
