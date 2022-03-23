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

#test데이터 list화 시키기
train_list_data_non = train.list_data(train_data_non)
train_list_data_nagative = train.list_data(train_data_negative)

#train -> test data
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
            #list별로 단어를 각각 train모델에 있는 단어의 확률을 더해서 더 큰 쪽으로 return
            for word in list:
                  if word in train.train_merge.keys():
                        non_sum += train.train_merge[word][0]
                        negative_sum += train.train_merge[word][1]
                  else:
                        non_sum += 0.1
                        negative_sum += 0.1
            
            if non_sum >=negative_sum:
                  non_count += 1
            else:
                  negative_count += 1
      
      return non_count, negative_count
      
#acc
def acc(data1, data2, data3, data4):
      return round(((data1 + data2) / (data3 + data4)), 4)
#recall
def recall(data1, data2, data3, data4):
      return round(((data1 / (data1 + data2) + data4 / (data3 +data4))/2), 4)
#precision  
def precision(data1, data2, data3, data4):
      return round(((data1 / (data1 + data3) + data4 / (data2 + data4))/2), 4)

#dict unpacking 으로 데이터값 받기
non_non, non_neg = train_model(train_list_data_non)
neg_non, neg_neg = train_model(train_list_data_nagative)
print(train_model(train_list_data_non))
#acc, precision, recall 출력
print(acc(non_non, non_neg, neg_non, neg_neg))
print(precision(non_non, non_neg, neg_non, neg_neg))
print(recall(non_non, non_neg, neg_non, neg_neg))
