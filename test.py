import re
from itertools import islice

text = ''
with open("glove/wordVectors.tsv", "r") as f:
    while True:
        line = list(islice(f, 1))
        if not line:
            break
        text += line[0].replace('\n', ' ').strip()

abc = text.replace(']', '\n').replace('[', '')
with open("ttt.txt", "w") as f:
    f.write(abc)



