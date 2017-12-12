import nltk
from data_prepare import DataPrepare
from common import *


def eval(results):
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for label, correctness in zip(results[0], results[1]):
        if label:  # positive
            if correctness:
                tp += 1
            else:
                fp += 1
        else:
            if correctness:
                tn += 1
            else:
                fn += 1

    if tp + fn == 0:
        recall = 1
    else:
        recall = tp / (tp + fn)

    accuracy = (tp + tn) / (tp + tn + fn + fp)

    if tp + fp == 0:
        precision = 1
    else:
        precision = tp / (tp + fp)
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    print("True Negative:{}, True Positive:{}, False Negative:{}, False Positive:{}".format(tn, tp, fn, fp))
    print("recall: {}".format(recall))
    print("precision: {}".format(precision))
    print("f1:{}".format(f1))
    return recall, precision, f1


def write_res(res, writer):
    writer.write("label, correctness, w1, w2\n")
    for label, correctness, word_pairs in zip(res[0], res[1], res[2]):
        tran_label = "Yes"
        if not label:
            tran_label = "No";
        correct_output = 'Incorrect'
        if correctness:
            correct_output = 'Correct'

        res_str = "{}\t{}\t{}\t\t{}".format(tran_label, correct_output, word_pairs[0], word_pairs[1])
        writer.write(res_str + "\n")


print("Preparing phrase2vec model...")
# p2v_model = Phrase2Vec()
p2v_model = None
print("Phrase2Vector model loaded...")
data = DataPrepare(p2v_model)
result_file = RESULT_DIR + os.sep + "result{}.txt".format(len(os.listdir(RESULT_DIR)))
a_recall = a_pre = a_f1 = 0
with open(result_file, "w", encoding='utf8') as fout:
    for raw_train_set, raw_test_set in data.ten_fold():
        train_x, train_y, train_word_pair = raw_train_set.all()
        test_x, test_y, test_word_pair = raw_test_set.all()
        train_set = []
        for x, y in zip(train_x, train_y):
            feature_set = dict()
            for i, feature in enumerate(x):
                feature_set["f" + str(i)] = feature
            if y[1] == 1:
                label = False
            else:
                label = True
            train_set.append((feature_set, label))

        test_set = []
        test_gold = []
        for x, y in zip(test_x, test_y):
            feature_set = dict()
            for i, feature in enumerate(x):
                feature_set["f" + str(i)] = feature
            test_set.append((feature_set))
            if y[1] == 1:
                test_gold.append(False)
            else:
                test_gold.append(True)

        algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
        maxent = nltk.MaxentClassifier.train(train_set, algorithm, trace=0, max_iter=100)
        res = [[], [], []]
        for i, case in enumerate(test_set):
            label = maxent.classify(case)
            correctness = label == test_gold[i]
            res[0].append(label)
            res[1].append(correctness)
            res[2].append(test_word_pair[i])
        re, pre, f1 = eval(res)
        write_res(res, fout)
        a_recall += re
        a_pre += pre
        a_f1 += f1
    avg_str = "Average recall:{}, precision:{}, f1:{}".format(a_recall / 10, a_pre / 10, a_f1 / 10)
    fout.write(avg_str)
    print(avg_str)
