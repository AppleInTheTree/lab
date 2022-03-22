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
import train

nltk.download('stopwords')
nltk.download('punkt')

#test data 불러오기
f = open('/Users/ahn_euijin/lab/naive_bayes/test_non_negative.csv', encoding = 'UTF-8')
train_data_non = csv.reader(f)

f = open('/Users/ahn_euijin/lab/naive_bayes/test_negative.csv', encoding = 'UTF-8')
train_data_negative = csv.reader(f)

train_list_data_non = train.list_data(train_data_non)
train_list_data_nagative = train.list_data(train_data_negative)


def train_model(test_data):
      non_count = 1
      negative_count = 1
      non_sum = 0
      negative_sum = 0
      #count = 0
      for list in test_data:
            a = train.short(list)
            b = word_tokenize(a)
            c = train.tokenizer(b)

            for word in list:
                  if word in train.train_merge.keys():
                        non_sum += train.train_merge[word][0]
                        negative_sum += train.train_merge[word][1]
            if non_sum >=negative_sum:
                  non_count += 1
            else:
                  negative_count += 1
      
      return non_count, negative_count
  
# def probability(data1, data2):
#   return print("%.0f%%" % (100 * abs(data2/data1))) 

# non_count , negative_count = train_model(train_list_data_nagative)

# probability(non_count, negative_count)
print(train_model(train_list_data_non))
print(train_model(train_list_data_nagative))
