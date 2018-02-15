#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GoogleScraper import scrape_with_config, GoogleSearchError
from common import *

if os.path.isfile("google_scraper.db"):
    os.remove("google_scraper.db")
keys = set()
query_type = 'stack_overflow'
if query_type == 'sentence':
    file_dir = './bing_sentenceQuery_word/'
    query_template = "what is {} in computer science"
elif query_type == "stack_overflow":
    file_dir = './bing_stackoverflow_word/'
    query_template = "{} site:stackoverflow.com definition"
elif query_type == "word":
    file_dir = './bing_word/'


for vocab_file in ["vocabulary.txt", "acronym_vocab.txt"]:
    with open(os.path.join(VOCAB_DIR, vocab_file), 'r', encoding='utf8') as fin:
        for word in fin:
            word = word.strip("\n\t\r ")
            file_path = os.path.join(file_dir, word + ".txt")
            if not os.path.isfile(file_path):
                query = query_template.format(word)
                keys.add(query)
print("query to run:", keys)

config = {
    'use_own_ip': 'True',
    'keywords': keys,
    'search_engines': ['bing', ],
    'num_pages_for_keyword': 1,
    'scrape_method': 'http',
    'do_caching': 'True'
}
try:
    search = scrape_with_config(config)
except GoogleSearchError as e:
    print(e)
