from scripts.common import *
import os

with open(DATA_DIR + os.sep + "vocabulary_wiki.txt") as fin:
    for line in fin:
        parts = line.split("\t")
        if len(parts == 1):
            print(parts[0])