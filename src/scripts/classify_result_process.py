"""
Process the classification results, write the 'yes' result to different files which will be utilized by SENET CLI
"""
import os
from common import *

if __name__ == "__main__":
    file_in = ["extension_res_1.text", "extension_res_2.text"]
    ml_path = os.path.join(RESULT_DIR, "ml_pair.txt")
    hu_path = os.path.join(RESULT_DIR, "hu_pair.txt")
    with open(ml_path, 'w') as ml_in, open(hu_path, 'w') as hu_in:
        for file_name in file_in:
            fin_path = os.path.join(RESULT_DIR, file_name)
            with open(fin_path) as fin:
                for line in fin.readlines():
                    line = line.strip("\n")
                    if line.count(",") > 2:
                        continue
                    parts = line.split(",")
                    label = parts[2]
                    w1 = parts[0]
                    w2 = parts[1]
                    if label == "yes-m":
                        ml_in.write("{},{}\n".format(w1, w2))
                    elif label == "yes-h":
                        hu_in.write("{},{}\n".format(w1, w2))
    print("Finished")