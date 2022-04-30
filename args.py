import torch
"""
项目 参数库
"""
# local
# project_dir = r"E:\临时转储\毕业设计\code\MRC_graduationProject_8003117062/"
# remote path
project_dir = '/mnt/MRC_testpoint/'

pretrained_file_dir = project_dir + "roberta_wwm_ext/"
dataset_dir = project_dir + "dataset/"
output_dir = project_dir + "model_dir/"
train_input_file = project_dir + "dataset/data/train.json"

dev_input_file = project_dir + "dataset/data/dev.json"
config_file = output_dir + 'bert_config.json'

predict_example_files = project_dir + 'predict/predict_dev.data'
predict_ground_truth_files = project_dir + 'predict/metric/ref_dev.json'
predict_result_files = project_dir + 'predict/metric/predicts_dev.json'\



seed = 42
device = torch.device("cuda", 0)
test_lines = 12839   # 1427！多少条训练数据，即：len(features), 记得修改 !

# 此处限制feature的数据文件
max_seq_length = 512  # 需要定义的文本最长长度（BERT常用512），包括'[CLS]'+query+'[SEP]'+answer+'[SEP]'，即query和answer的长度是max_seq_length-3
max_query_length = 60  # 需要定义的最长问题（query）长度，若query超过该长度，则会截断只取前半部分
max_para_num = 5  # 选择几篇文档进行预测
learning_rate = 5e-5
batch_size = 4
num_train_epochs = 2
gradient_accumulation_steps = 8  # 梯度累积,将梯度累积设置为8会放大8倍的batch_size，因为8次累计之后才会更新梯度
num_train_optimization_steps = int(test_lines / gradient_accumulation_steps / batch_size) * num_train_epochs
log_step = int(test_lines / batch_size / 4)  # 每个epoch验证几次，默认4次

#下层网络参数
down_net = 'attn_Bilstm_day2'   #'BiLstm','self-attention' #none代表base_line
attention_head_num = 8

#对抗指数
zhendang = 0.1

epsilon=0.25
