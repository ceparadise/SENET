from common import *
import os
import re
from nltk.corpus import stopwords

black_list_sent = ["learn share knowledge build career"]
stopWords = set(stopwords.words('english'))
def clean_context_file(content):
    final_lines = []
    content = content.lower()
    content = re.sub("[^\w\-&\s]", " ", content)
    content = re.sub("[^\S\r\n]+", " ", content)
    content = re.sub("\s*-\s*", "-", content)
    lines = [x.strip(" ") for x in content.split("\n")]
    lines = set(lines)
    for line in lines:
        if line in black_list_sent:
            continue
        tokens = line.split()
        tokens = [x for x in tokens if x not in stopWords]
        final_lines.append(" ".join(tokens))
    return "\n".join(final_lines)


if __name__ == "__main__":
    for dir in BING_WORD_DIR:
        clean_dir = dir + "_clean"
        if not os.path.isdir(clean_dir):
            os.mkdir(clean_dir)
        file_names = os.listdir(dir)
        for file_name in file_names:
            file_path = os.path.join(dir, file_name)
            with open(file_path, 'r', encoding="utf8") as fin:
                content = fin.read()
                cleaned_content = clean_context_file(content)
                with open(os.path.join(clean_dir, file_name), 'w', encoding="utf8") as fout:
                    fout.write(cleaned_content)
    print("finished")
