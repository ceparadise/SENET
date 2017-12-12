import common
import os

with open(common.W2V_DIR + os.sep + "w2v_2.data", 'a', encoding='utf8') as fout:
    for fname in os.listdir(common.BING_WORD_DIR):
        print(fname)
        with open(common.BING_WORD_DIR + os.sep + fname, encoding='utf8') as fin:
            fout.write(fin.read())
print("finished")
