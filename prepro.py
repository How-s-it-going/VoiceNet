import os
import re

rpath = './datasets/serif'
wpath = './datasets/sentences.dat'

serif_dict = {}

for fpath in os.listdir(rpath):
    with open(rpath + '/' + fpath, 'r', encoding='shift_jis') as f:
        texts = f.readlines()

    for idx, text in enumerate(texts):
        line = text.strip()
        if line == '!凛音':
            wavname = texts[idx - 1].split(' ')[1].strip()
            wavname = './datasets/sounds/' + wavname + '.wav'
            serif = texts[idx + 1]
            serif = re.sub(r'[　・「」@\[\]\\]', '', serif).strip()
            serif_dict[wavname] = serif

with open(wpath, 'w', encoding='utf-8') as f:
    for key, val in serif_dict.items():
        f.write(key + '|' + val + '\n')
