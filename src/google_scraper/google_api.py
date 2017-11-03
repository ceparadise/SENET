import search_google.api
from common import *

with open("api.key") as key_in, open(VOCAB_DIR + os.sep + 'vocabulary.txt') as word_in:
    lines = key_in.readlines();
    cse_id = lines[0].split('=')[1].strip('\n')
    api_key = lines[1].split('=')[1].strip('\n')

    buildargs = {
        'serviceName': 'customsearch',
        'version': 'v1',
        'developerKey': api_key
    }
    for word in word_in:
        succ = False
        for i in range(0, 10):
            try:
                word = word.strip(' \n\t\r')
                cseargs = {
                    'q': word,
                    'cx': cse_id,
                    'fileType': ''
                }
                results = search_google.api.results(buildargs, cseargs)
                results.download_links(GOOGLE_DOWNLOAD)
            except Exception as e:
                print(e)
                continue
            else:
                succ = True
                break
        if not succ:
            print("failure happen when processing " + word)
