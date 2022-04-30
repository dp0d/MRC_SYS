import  json
import numpy as np


len_list = []
with open("data/train.json",'r',encoding='utf-8') as f:
    samples = json.load(f)
    # 读取每个paragraphs
    for sample in samples['data'][0]['paragraphs']:

        len_list.append(len(sample['context'].strip()))
len_arr = np.array(len_list)
print(np.percentile(len_arr, [75,90,95])) # [400. 558. 711.]


# query
len_list = []
with open("data/train.json",'r',encoding='utf-8') as f:
    samples = json.load(f)
    # 读取每个paragraphs
    for sample in samples['data'][0]['paragraphs']:

        len_list.append(len(sample['qas'][0]['question'].strip()))
len_arr = np.array(len_list)
print(np.percentile(len_arr, [75,90,95])) # [400. 558. 711.]


