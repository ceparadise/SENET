import os

PROJECT_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
DATA_DIR = PROJECT_ROOT + os.sep + "data"

BING_WORD_DIR = PROJECT_ROOT + os.sep + "src" + os.sep + "google_scraper" + os.sep + "bing_stackoverflow_word"

RAW_DIR = DATA_DIR + os.sep + "raw"
SE_BOOK_DIR = RAW_DIR + os.sep + "SE_Books_txt"
PURE_REQ_DIR = RAW_DIR + os.sep + "PURE_Requirement"
GOOGLE_DOWNLOAD = RAW_DIR + os.sep + 'Google_Word_Download'
RAW_DIR_LIST = [SE_BOOK_DIR, PURE_REQ_DIR, BING_WORD_DIR]

VOCAB_DIR = DATA_DIR + os.sep + "vocab"
W2V_DIR = DATA_DIR + os.sep + "w2v"
RESULT_DIR = DATA_DIR + os.sep + "results"

PACKAGES_DIR = PROJECT_ROOT + os.sep + "packages"
