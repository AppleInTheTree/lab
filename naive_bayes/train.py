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
f = open('train_non_negative.csv', encoding = 'UTF-8')
data_non = csv.reader(f)

#cvs파일 정제하기 위해 list로 받아오기
list_data_non =[]
for row in data_non:
  list_data_non += row
f.close()

#cvs파일로 negative-dataset 불러오기 
d = open('train_negative.csv', encoding = 'UTF-8')
data_negative = csv.reader(d)

#cvs파일 정제하기 위해 list로 받아오기
list_data_negative = []
for low in data_negative:
  list_data_negative += low

d.close()

#정규식 이용하여 1,2자리수 영어 단어 삭제
shortword = re.compile(r'\W*\b\w{1,2}\b')
first_short_non = shortword.sub('',str(list_data_non))
first_short_negative = shortword.sub('',str(list_data_negative))

#특수문자 삭제
second_short_non = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', first_short_non)
second_short_negative = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', first_short_negative)
#stopword 삭제 
stop_words_list = stopwords.words('english')

#tokenize툴로 정규화된 문자 토큰화 
tokenizer_non = word_tokenize(second_short_non)
tokenizer_negative = word_tokenize(second_short_negative)

#정제된 데이터 results에 반환 
results_non =[]
results_negative =[]
for word in tokenizer_non:
  if word not in stop_words_list:
    results_non.append(word) 

for words in tokenizer_negative:
  if words not in stop_words_list:
    results_negative.append(words) 

#colletion.Counter 함수 이용하여 각 단어의 빈수도 확인 
counter_non = collections.Counter(results_non)
counter_negative = collections.Counter(results_negative)

dict_counter_non = dict(counter_non)
dict_counter_negative = dict(counter_negative)
#print(len(results_non))
#print(dict_counter_non)

#voca reduction 딕셔너리 언팩킹을 통해 빈수도 3 미만 삭제
precise_count_non = {}
precise_count_negative = {}

for key, value in dict_counter_non.items():
  if value >= 3:
    precise_count_non[key] = value

for key, value in dict_counter_negative.items():
  if value >= 3:
    precise_count_negative[key] = value

#마지막 정제된 데이터 리스트 생성
final_data_non = dict_counter_non.keys()
final_data_negative = dict_counter_negative.keys()

merge_dict =collections.defaultdict(list)

for data in (precise_count_non, precise_count_negative): 
    for key, value in data.items():
        merge_dict[key].append(value)

#print(merge_dict)

#making dataframe using pandas
col_names = ["non_negative", "negative"]
df = pd.DataFrame.from_dict(merge_dict, orient='index', columns=col_names)

# Nan value 제거 (smoothing)
df_filled  = df.fillna(1)
#print(df_filled)

# 연산 값 dataframe에 추가
df_sum = df_filled.sum(axis= 1)
df_filled.insert(2, "sum",df_sum,True)

df_non_stat = df_filled['non_negative'] / df_filled['sum']
df_negative_stat = df_filled['negative'] / df_filled['sum']

df_filled.insert(3, "non_stat",df_non_stat,True)
df_filled.insert(4, "negative_stat",df_negative_stat,True)


print(df_filled)










#결과값 확인
# print(type(final_data))
# print(precise_count)

#print(len(final_data_non))


# rows = list(
#     zip_longest(
#         precise_count_non.items(),
#         precise_count_negative.items()
#     ))
# print(rows)

# data 합치기 
# with open('merged_file.csv', 'w', newline='', encoding = 'UTF-8') as merge_f:
#     fieldnames = ['negative_word', 'count1', 'non_negative_word', 'count2']
#     writer = csv.DictWriter(merge_f, fieldnames=fieldnames)
    
#     writer.writeheader()
#     for key in precise_count:
#         writer.writerow({'negative_word': key, 'count1': precise_count[key]})
#     for key in temp_dict:
#         writer.writerow({'non_negative_word' : key, 'count2' : temp_dict})

# row_dict = {'negative_word': None, 'count1': None, 'non_negative_word': None, 'count2': None}

# with open('merged_file.csv', 'w', newline='', encoding = 'UTF-8') as merge_f:
#     writer = csv.DictWriter(merge_f, fieldnames=row_dict)
#     writer.writeheader()

#     for row in rows:
#         precise, temp = row

#         row_dict['negative_word']     = precise[0]
#         row_dict['count1']            = precise[1]
#         row_dict['non_negative_word'] = temp[0]
#         row_dict['count2']            = temp[1]

#         writer.writerow(row_dict)
# commit