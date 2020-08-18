#!/usr/bin/env python
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Gerard Ion Gállego Olsina
# Machine Translation group
# Universitat Politècnica de Catalunya (UPC)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
Remove punctuation from a tokenized text file
"""

import sys
from string import punctuation

clean_file = []
for line in sys.stdin:
    my_punctuation = punctuation.replace("'", "")
    line_clean = line.translate(str.maketrans("", "", my_punctuation))
    print(" ".join(line_clean.split()).lower())





