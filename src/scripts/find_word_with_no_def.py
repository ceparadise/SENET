import os
from common import *

count_def = 0
count_nodef = 0
with open(DATA_DIR + os.sep + "vocabulary_wiki.txt") as fin:
    for line in fin:
        key, defin = line.strip("\n").split("\t")
        if len(defin) == 0:
            count_def += 1
            print(key)
        else:
            count_nodef += 1
            print([key, defin])

print("words with def ={}, without={}".format(count_def, count_nodef))
