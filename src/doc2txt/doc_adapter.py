from subprocess import Popen, PIPE
from common import *

"""
Install antiword,currently work on windows need to move antiword dir to C: or set ANTIWORDPATH
"""
class DocAdapter:
    def __init__(self):
        pass

    def parse(self, file_path):
        antiword_path = PACKAGES_DIR + os.sep + 'antiword'
        cmd = [antiword_path+ os.sep +'antiword', file_path]
        p = Popen(cmd, stdout=PIPE)
        stdout, stderr = p.communicate()
        return stdout

if __name__ == '__main__':
    # TODO Add command line support. Input the pdf path and output the file to a txt file
    text = DocAdapter().parse(PURE_REQ_DIR + os.sep + "2001 - npac.doc")
    print(text)