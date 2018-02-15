"""
Merge the context documents and do clean.
"""
from common import *
from clean_vocab import WordCleaner
from document_clean import clean_context_file

if __name__ == "__main__":
    for dir in BING_WORD_DIR:
        merge_dir = dir + "_merge"
        if not os.path.isdir(merge_dir):
            os.mkdir(merge_dir)
        file_names = os.listdir(dir)
        for file_name in file_names:
            word = file_name[: file_name.index(".txt")]
            new_file_name = WordCleaner.to_file_name_format(word) + ".txt"
            new_path = os.path.join(merge_dir, new_file_name)
            old_path = os.path.join(dir, file_name)
            mode = "w"
            if os.path.isfile(new_path):
                mode = "a"
            with open(new_path, mode, encoding="utf8") as fout, open(old_path, "r", encoding="utf8") as fin:
                fout.write(fin.read() + "\n")

    for dir in BING_WORD_DIR:
        merge_dir = dir + "_merge"
        file_names = os.listdir(merge_dir)
        for file_name in file_names:
            file_path = os.path.join(merge_dir, file_name)
            with open(file_path, "r", encoding="utf8") as fin:
                dirty_content = fin.read()
            clean_content = clean_context_file(dirty_content)
            with open(file_path, "w", encoding="utf8") as fout:
                fout.write(clean_content)
print("finish")
