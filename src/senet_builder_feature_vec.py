from common import *
from feature_extractors import FeatureExtractor
from GoogleScraper import scrape_with_config, GoogleSearchError
import urllib
from lxml import html
from preprocess import Preprocessor
import os
from threading import Thread
import functools
import nltk, sys
from clean_vocab import WordCleaner


class FeatureBuilder:
    def __init__(self):
        self.data_set = []

    def timeout(timeout):
        def deco(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]

                def newFunc():
                    try:
                        res[0] = func(*args, **kwargs)
                    except Exception as e:
                        res[0] = e

                t = Thread(target=newFunc)
                t.daemon = True
                try:
                    t.start()
                    t.join(timeout)
                except Exception as je:
                    print('error starting thread')
                    raise je
                ret = res[0]
                if isinstance(ret, BaseException):
                    raise ret
                return ret

            return wrapper

        return deco

    @timeout(60)
    def get_page_content(self, link):
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers = {'User-Agent': user_agent, }
        try:
            preprocessor = Preprocessor()
            request = urllib.request.Request(link, None, headers)
            with urllib.request.urlopen(request) as url:
                html_page = url.read()
            tree = html.fromstring(html_page)
            text = tree.xpath('//p/text()')
            text = " ".join(text)
            tokens = preprocessor.tokenize(text)
            tokens = preprocessor.remove_stop_word(tokens)
            tokens = preprocessor.lemmatizing(tokens)
            clean_sents_list = preprocessor.clean_nonAlpha(tokens)
            return clean_sents_list
        except Exception as e:
            print(e)

    def search(self, dir, keyword):
        BING_STACKOVERFLOW = BING_WORD_DIR_ROOT + os.sep + "bing_stackoverflow_word"
        BING_REGULAR = BING_WORD_DIR_ROOT + os.sep + "bing_word"
        BING_SENTENCE_QUERY = BING_WORD_DIR_ROOT + os.sep + "bing_sentenceQuery_word"

        if dir == BING_STACKOVERFLOW:
            query = keyword.strip("\n\t\r") + " site:stackoverflow.com definition"
        elif dir == BING_SENTENCE_QUERY:
            query = "what is {} in computer science".format(keyword.strip("\n\t\r"))
        else:
            query = keyword

        config = {
            'use_own_ip': 'True',
            'keyword': query,
            'search_engines': 'bing',
            'num_pages_for_keyword': 1,
            'scrape_method': 'http',
            'do_caching': 'True'

        }

        res = ""
        print("Scraping context for word:" + keyword)
        try:
            sqlalchemy_session = scrape_with_config(config)
            for serp in sqlalchemy_session.serps:
                for link in serp.links:
                    try:
                        doc = self.get_page_content(link.link)
                        if doc:
                            for str in doc:
                                sent = " ".join(str)
                                res += " " + sent
                    except Exception as sql_e:
                        print(sql_e)
        except GoogleSearchError as e:
            print(e)

        file_path = os.path.join(dir, WordCleaner.to_file_name_format(keyword) + ".txt")
        if not os.path.isfile(file_path) and len(res) > 0:
            try:
                with open(file_path, 'w', encoding='utf8') as fout:
                    fout.write(res)
            except Exception as e:
                print(e)

        return res

    def write_features_vecs(self, pairs, write_to_file_path):
        with open(write_to_file_path, 'w') as fout:
            for i, pair in enumerate(pairs):
                try:
                    print("Preparing feature vec, progress {}/{}".format(i, len(pairs)))
                    words1 = pair[0].strip(" \n")
                    words2 = pair[1].strip(" \n")
                    vector = []
                    vector.extend(self.build_feature_vector(words1, words2))
                    self.data_set.append(
                        (vector, (words1, words2)))
                    if write_to_file_path != "":
                        entry = "{}|{}|{}\n".format(words1, words2, ",".join(map(str, vector)))
                        fout.write(entry)
                except Exception as e:
                    print(e)

    def read_feature_vecs(self, file_list):
        for file in file_list:
            with open(file) as fin:
                lines = fin.readlines()
                if len(lines) > 0:
                    for line in lines:
                        parts = line.split("|")
                        w1 = parts[0]
                        w2 = parts[1]
                        vec = []
                        for f in parts[2].strip("\n\t\r").split(","):
                            if f == 'True' or f == 'False':
                                vec.append(bool(f))
                            else:
                                vec.append(float(f))
                        self.data_set.append((vec, (w1, w2)))

    def build_feature_vector(self, words1, words2):
        define1 = ""
        define2 = ""

        for dir in BING_WORD_DIR:
            try:
                with open(dir + os.sep + WordCleaner.to_file_name_format(words1) + ".txt", encoding='utf8') as f1:
                    define1 += f1.read()
            except Exception as e:
                define1 += self.search(dir, words1)

        for dir in BING_WORD_DIR:
            try:
                with open(dir + os.sep + WordCleaner.to_file_name_format(words2) + ".txt", encoding='utf8') as f2:
                    define2 += f2.read()
            except Exception as e:
                define2 += self.search(dir, words2)
        return FeatureExtractor().get_feature(words1, define1, words2, define2)


class PairBuilder:
    def __read_words(self, file_path):
        res = set()
        with open(file_path, encoding='utf8') as fin:
            for line in fin.readlines():
                phrase = line.strip("\n\t\r ")
                res.add(phrase)
        return list(res)

    def __get_all_relationships(self):
        constras = os.path.join(VOCAB_DIR, "contrast.txt")
        hyper = os.path.join(VOCAB_DIR, "hyper.txt")
        related = os.path.join(VOCAB_DIR, "related.txt")
        synonym = os.path.join(VOCAB_DIR, "synonym.txt")
        one_pair_in_line = [constras, related, synonym]
        multi_pair_in_line = [hyper]
        rel = set()
        for f in one_pair_in_line:
            with open(f) as fin:
                for line in fin.readlines():
                    line = line.strip("\n\t\r ")
                    word_pair = line.split(",")
                    relation = (word_pair[0], word_pair[1])
                    rel.add(relation)
        for f in multi_pair_in_line:
            with open(f) as fin:
                for line in fin.readlines():
                    line = line.strip("\n\t\r ")
                    hyper, rest = line.split(":")
                    for w_r in rest.split(","):
                        rel.add((hyper[0], w_r))
        return rel

    def __init__(self, expension_list_txt):
        self.exp_list = self.__read_words(expension_list_txt)
        self.relations = self.__get_all_relationships()

    def get_pairs(self):
        pairs = []
        vocab = self.__read_words(os.path.join(VOCAB_DIR, "vocabulary.txt"))
        for w_v in vocab:
            for w_e in self.exp_list:
                if (w_v, w_e) not in self.relations and (w_e, w_v) not in self.relations:
                    pairs.append((w_v, w_e))
        return pairs


if __name__ == "__main__":
    """
    This script crreate feature vectors and write to disk. Next phrase will read the feature vecs and 
    apply them to classifier
    """
    try:
        partition_num = int(sys.argv[1])
    except:
        partition_num = 1

    try:
        total_partition_num = int(sys.argv[2])
    except:
        total_partition_num = 1
    nltk_requires = ["punkt", "averaged_perceptron_tagger"]
    for nltk_require in nltk_requires:
        nltk.download(nltk_require)

    pair_builder = PairBuilder(os.path.join(DATA_DIR, "dataset", "requirement_extension_vocab.txt"))
    pairs = pair_builder.get_pairs()
    work_size = len(pairs)
    chunk_siz = int(work_size / total_partition_num)
    work_partition_start = (partition_num - 1) * chunk_siz
    work_partition_end = min(partition_num * chunk_siz, work_size)
    pairs = pairs[work_partition_start: work_partition_end]
    print("Total pairs to process:{}, the working interval for partition {} is from {} to {}".format(
        work_size, partition_num, work_partition_start, work_partition_end))
    print("Start building feature vectors ...")
    fb = FeatureBuilder()
    feature_vec_backup_file = os.path.join(DATA_DIR, "dataset", "fv_backup", "fv_{}".format(partition_num))
    fb.write_features_vecs(pairs, feature_vec_backup_file)
