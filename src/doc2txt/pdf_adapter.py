#!/usr/bin/env python

from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import HTMLConverter, TextConverter
from pdfminer.layout import LAParams
from io import BytesIO
from common import *

'''
Convert a pdf document into plain txt. Require pdfminer.six on python3
'''


class PDFAdapter:
    def __init__(self):
        self.packageMapper = dict()
        self.packageMapper['pdfminer'] = self._pdfminer_parse

    def _pdfminer_parse(self, file_path):
        """
        Parse a single pdf document with pdfminer package.

        :param file_path: The path to the pdf file
        :return: string representation of the pdf
        """
        pfile = open(file_path, 'rb')
        output = BytesIO()
        manager = PDFResourceManager()
        # converter = TextConverter(manager, output, laparams=LAParams())
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)
        for page in PDFPage.get_pages(pfile, None):
            interpreter.process_page(page)
        text = output.getvalue()
        return text

    def parse(self, file_path, package_name='pdfminer'):
        """
        Do the parse with given package implementation

        :param file_path: The path to the pdf file
        :param package_name:  The package name used for implementation
        :return: String representation of the pdf
        """
        implementation = self.packageMapper.get(package_name)
        return implementation(file_path)


if __name__ == '__main__':
    # TODO Add command line support. Input the pdf path and output the file to a txt file
    text = PDFAdapter().parse(PURE_REQ_DIR + os.sep + "0000 - cctns.pdf")
    print(text)
