from common import *

if __name__ == "__main__":
    word_info = {}
    query_list = []
    with open(os.path.join(DATA_DIR, "dataset", "requirement_extension_vocab.txt")) as fin:
        for line in fin:
            line = line.strip("\n\t\r ")
            query_list.append(line)

    with open("extension_result.total") as fin:
        for line in fin:
            line = line.strip("\n\t\r ")
            parts = line.split(",")
            w1 = parts[0]
            w2 = parts[1]
            score_str = parts[2].strip("\[\]").split()[0]
            score = float(score_str)
            if w1 not in word_info:
                word_info[w1] = []
            else:
                word_info[w1].append((w2, score))

            if w2 not in word_info:
                word_info[w2] = []
            else:
                word_info[w2].append((w1, score))

    k = 10
    thresholds = [.66, .76, .86]

    with open("top_" + str(k) + ".res", "w") as fout:
        for wd in query_list:
            if wd in word_info:
                recommands = word_info[wd]
                recommands = sorted(recommands, key=lambda x: x[1], reverse=True)
                fout.write(wd + "\t" + str(recommands[:k]) + "\n")
            else:
                print("{} not in the result...".format(wd))

    for thresold in thresholds:
        with open("thre_" + str(thresold) + ".res", "w") as fout:
            for wd in query_list:
                if wd in word_info:
                    recommands = word_info[wd]
                    filtered = [x for x in recommands if x[1] >= thresold]
                    fout.write(wd + "\t" + str(filtered) + "\n")
                else:
                    print("{} not in the result...".format(wd))

    print("Finished")
