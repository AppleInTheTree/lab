import csv
from email.policy import default # cvs 파일 읽기 위해
import numpy as np
import pandas as pd
from itertools import zip_longest
import collections
import nltk # nlp 툴
from nltk.corpus import stopwords # stopword 제거 위해 
import re # 정규식
from nltk.tokenize import word_tokenize # dataset tokenizing 
from nltk.stem import PorterStemmer # libstemmer같은 stemming tool 

nltk.download('stopwords')
nltk.download('punkt')

#cvs파일로 non-dataset 불러오기 
f = open('test_non_negative.csv', encoding = 'UTF-8')
data_non = csv.reader(f)

#cvs파일 정제하기 위해 list로 받아오기
list_data_non =[]
for row in data_non:
  list_data_non += row
f.close()

#cvs파일로 negative-dataset 불러오기 
d = open('test_negative.csv', encoding = 'UTF-8')
data_negative = csv.reader(d)

#cvs파일 정제하기 위해 list로 받아오기
list_data_negative = []
for low in data_negative:
  list_data_negative += low

d.close()


def predictor(test_data):
    for row in test_data:
        

        pass