import sqlite3
import urllib

from lxml import html
from preprocess import Preprocessor
import os

conn = sqlite3.connect('google_scraper.db')
c = conn.cursor()

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers = {'User-Agent': user_agent, }

query_type = 'sentence'  # change type here

if query_type == 'sentence':
    file_dir = './bing_sentenceQuery_word/'
    query_template = "what is {} in computer science"
elif query_type == "stack_overflow":
    file_dir = './bing_stackoverflow_word/'
    query_template = "{} site:stackoverflow.com definition"

from threading import Thread
from threading import Lock
import functools
import time


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
def get_page_content(link):
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
        pass


def worker(sub_query_link, thread_num):
    cur_visited_num = 0
    for q_file in sub_query_link:
        for link in sub_query_link[q_file]:
            print("querying {} with link {}".format(q_file, link))
            try:
                if os.path.isfile(q_file):
                    mode = 'a'
                else:
                    mode = 'w'
                    cur_visited_num += 1
                    if cur_visited_num % 5 == 0:
                        print("T{}: word processed = {}".format(thread_num, cur_visited_num))

                page_content = get_page_content(link)
                if page_content:
                    with open(q_file, mode, encoding='utf8') as fout:
                        for str in page_content:
                            str = " ".join(str)
                            fout.write(str + "\n")
                        fout.write("\n")
                time.sleep(1)
            except Exception as e:
                print(e)


word_num = c.execute("SELECT count(DISTINCT(query)) FROM serp JOIN link ON link.serp_id = serp.id");
for row in word_num:
    print("Total words:" + str(row[0]))
join_result = c.execute("SELECT query, link, title,snippet FROM serp JOIN link ON link.serp_id = serp.id")
link_num = 0
count_lock = Lock()
word_link = dict()
for row in join_result:
    query = row[0]
    if query_type == "sentence":
        begin_index = len("what is ")
        end_len = len(" in computer science")
    elif query_type == "stack_overflow":
        begin_index = 0
        end_len = len(" site:stackoverflow.com definition")

    query = query[begin_index:-end_len]
    q_file = os.path.join(file_dir, query + ".txt")
    if os.path.isfile(q_file):
        continue
    link = row[1]
    if q_file not in word_link:
        word_link[q_file] = []
    word_link[q_file].append(link)

threads = []
for thread_num in range(0, 4):
    d = {key: value for i, (key, value) in enumerate(word_link.items()) if i % 4 == thread_num}
    print("Thread {} have {} words to fetch".format(thread_num, len(d)))
    t = Thread(target=worker, args=(d, thread_num))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
print("Finished")
