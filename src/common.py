import os
from nltk.corpus import wordnet

PROJECT_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
DATA_DIR = PROJECT_ROOT + os.sep + "data"
RNNMODEL = DATA_DIR + os.sep + "RNNModel"

BING_WORD_DIR_ROOT = PROJECT_ROOT + os.sep + "src" + os.sep + "google_scraper"
BING_STACKOVERFLOW = BING_WORD_DIR_ROOT + os.sep + "bing_stackoverflow_word"
BING_REGULAR = BING_WORD_DIR_ROOT + os.sep + "bing_word"
BING_SENTENCE_QUERY = BING_WORD_DIR_ROOT + os.sep + "bing_setenceQuery_word"
BING_WORD_DIR = [BING_STACKOVERFLOW, BING_REGULAR]  # , BING_SENTENCE_QUERY]

RAW_DIR = DATA_DIR + os.sep + "raw"
SE_BOOK_DIR = RAW_DIR + os.sep + "SE_Books_txt"
PURE_REQ_DIR = RAW_DIR + os.sep + "PURE_Requirement"
GOOGLE_DOWNLOAD = RAW_DIR + os.sep + 'Google_Word_Download'
RAW_DIR_LIST = [SE_BOOK_DIR, PURE_REQ_DIR, BING_WORD_DIR]

VOCAB_DIR = DATA_DIR + os.sep + "vocab"
W2V_DIR = DATA_DIR + os.sep + "w2v"
RESULT_DIR = DATA_DIR + os.sep + "results"

PACKAGES_DIR = PROJECT_ROOT + os.sep + "packages"


def report_ration(data_set):
    pos_cnt = 0
    neg_cnt = 0
    for entry in data_set:
        if entry[1] == [1, 0]:
            pos_cnt += 1
        elif entry[1] == [0, 1]:
            neg_cnt += 1
        else:
            raise Exception
    return pos_cnt, neg_cnt


def unbalance_dataset(dataset, neg_vs_pos_ration):
    "neg_vs_pos_ration means how many times of neg samples should be against the pos ones"
    pos_entries = [x for x in dataset if x[1] == [1, 0]]
    neg_entries = [x for x in dataset if x[1] == [0, 1]]
    final_set = []
    cur_index = 0
    for pos_entry in pos_entries:
        final_set.append(pos_entry)
        next_index = min(cur_index + neg_vs_pos_ration, len(neg_entries))
        final_set.extend(neg_entries[cur_index:next_index])
        cur_index = next_index
        if cur_index >= len(neg_entries):
            break
    return final_set


def get_synsets(words):
    syn_sets = set()
    for word in words:
        word_morphy = wordnet.morphy(word)
        if word_morphy == None:
            word_morphy = word
        for syn in wordnet.synsets(word_morphy):
            syn_sets.add(syn)
    return syn_sets


def get_related_set(words):
    related_set = set()
    for word in words:
        word_morphy = wordnet.morphy(word)
        if word_morphy == None:
            word_morphy = word
        for syn in wordnet.synsets(word_morphy):
            for l in syn.lemmas():
                related_set.add(l.name())
                if l.antonyms():
                    related_set.add(l.antonyms()[0].name())
    return related_set
