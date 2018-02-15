import os
import time
from common import *
import subprocess

now = time.time()

folder_names = ["bing_sentenceQuery_word", "bing_stackoverflow_word", "bing_word"]
folder_paths = [os.path.join(DATA_DIR, "..", "src", "google_scraper", x) for x in folder_names]

for folder in folder_paths:
    files = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    for filename in files:
        if (now - os.stat(filename).st_mtime) < 60 * 60 * 72:
            print("remove {}".format(filename))
            os.remove(filename)
