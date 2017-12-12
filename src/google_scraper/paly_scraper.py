#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GoogleScraper import scrape_with_config, GoogleSearchError
from common import *
import sqlite3

keys = set()
with open(VOCAB_DIR + os.sep + "vocabulary.txt", 'r', encoding='utf8') as fin:
    for word in fin:
        keys.add(word.strip("\n\t\r") + " site:stackoverflow.com definition")
while True:
    try:
        conn = sqlite3.connect('google_scraper.db')
        c = conn.cursor()
        for i, row in enumerate(c.execute("SELECT query FROM serp")):
            cur_key = row[0]
            if cur_key in keys:
                keys.remove(cur_key)
        keys = list(keys)
    except Exception as e:
        print(e)

    # config = {
    #     'use_own_ip': True,
    #     'keywords': keys,
    #     'search_engines': ['bing'],
    #     'num_pages_for_keyword': 1,
    #     'scrape_method': 'selenium',
    #     'sel_browser': 'chrome',
    # }

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
