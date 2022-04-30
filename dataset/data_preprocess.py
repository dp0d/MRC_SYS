import sys
import json
from collections import Counter
import jieba
from tqdm import tqdm
sys.path.append('../')
import args
from tokenization import BertTokenizer


def read_squad_examples(input_file, is_training=True):
    examples = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        samples = json.load(infile)
        print(len(samples['data'][0]['paragraphs']))
        # 读取每个paragraphs
        for sample in samples['data'][0]['paragraphs']:
            doc_text = sample['context'].strip()
            qas_id = sample['qas'][0]['id'].strip()
            question_text = sample['qas'][0]['question'].strip()
            # 能否回答，根据过滤条件之后，均设置为能够回答
            can_answer = 1
            answer = sample['qas'][0]['answers'][0]['text'].strip()
            start_position = sample['qas'][0]['answers'][0]['answer_start']
            end_position = start_position + len(answer) - 1

            # 答案不存在，不建立example
            if len(answer) == 0 or len(question_text) == 0:
                continue
            if start_position > end_position or end_position >= len(doc_text) or start_position >= len(doc_text):
                continue
            if is_training:
                example = {
                    "qas_id": qas_id,
                    "question_text": question_text,
                    "doc_text": doc_text,
                    # 能否回答，根据过滤条件之后，均设置为能够回答
                    "can_answer": 1,
                    "start_position": start_position,
                    "end_position": end_position,
                    "answer": answer}
                examples.append(example)
    print("len(examples):", len(examples))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length, max_ans_length):
    features = []
    for example in tqdm(examples):
        # ['爬', '行', '垫', '什', '么', '材', '质', '的', '好']
        query_tokens = list(example['question_text'])
        doc_text = example['doc_text']
        doc_text = doc_text.replace(u"“", u"\"")
        doc_text = doc_text.replace(u"”", u"\"")
        start_position = example['start_position']
        end_position = example['end_position']
        can_answer = example['can_answer']
        answer = example['answer']
        # 中文反斜杠
        answer = answer.replace(u"“", u"\"")
        answer = answer.replace(u"”", u"\"")
        # ['X', 'P', 'E']
        answer_tokens = list(answer)
        # 字符个数超过max_embedding截断
        if len(answer_tokens) > max_ans_length:
            answer_tokens = answer_tokens[0:max_ans_length - 1]
        # [166, 158, 147],vocab.txt中的位置
        answer_ids = tokenizer.convert_tokens_to_ids(answer_tokens)
        answer_mask = [1] * len(answer_ids)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tokens = []
        tokens_q = []
        segment_ids = []
        # tokens:['[CLS]']
        tokens.append("[CLS]")
        # segment_ids:[0]
        segment_ids.append(0)

        if start_position != -1:
            start_position = start_position + 1
            end_position = end_position + 1
        for token in query_tokens:
            tokens_q.append(token)
            tokens.append(token)
            segment_ids.append(0)
            if start_position != -1:
                start_position = start_position + 1
                end_position = end_position + 1

        # tokens:['[CLS]', '爬', '行', '垫', '什', '么', '材', '质', '的', '好', '[SEP]']
        tokens.append("[SEP]")
        # segment_ids:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        segment_ids.append(0)
        if start_position != -1:
            start_position = start_position + 1
            end_position = end_position + 1

        for i in doc_text:
            tokens.append(i)
            segment_ids.append(1)
        # tokens:['[CLS]', '爬', '行', '垫', '什', '么', '材', '质', '的', '好', '[SEP]', '爬', '行', '垫', '根', '据', '中', '间', '材', '料',...
        #  ...'就', '的', '薄', '毯', '子', '让', '他', '爬', '。', '[SEP]']
        tokens.append("[SEP]")  # ！SEP
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1...1, 1, 1, 1, 1, 1, 1, 1, 1]
        segment_ids.append(1)
        if end_position >= max_seq_length:
            continue

        if len(tokens) > max_seq_length:
            tokens[max_seq_length - 1] = "[SEP]"
            input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])  # ！SEP
            segment_ids = segment_ids[:max_seq_length]
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_ids_q = tokenizer.convert_tokens_to_ids(tokens_q)

        input_mask = [1] * len(input_ids)
        input_mask_q = [1] * len(input_ids_q)
        assert len(input_ids) == len(segment_ids)
        assert len(input_ids_q) == len(input_mask_q)

        features.append(
            {"input_ids": input_ids,
             "input_ids_q": input_ids_q,
             "input_mask": input_mask,
             "input_mask_q": input_mask_q,
             "segment_ids": segment_ids,
             "can_answer": can_answer,
             "start_position": start_position,
             "end_position": end_position,
             "answer_ids": answer_ids,
             "answer_mask": answer_mask}
        )
    print("len(features):", len(features))
    return features
    


def make_feature_example_datafile(features, file_name='Undefined'):
    with open(args.project_dir+"dataset/{}.data".format(file_name), 'w', encoding="utf-8") as fout:
        for feature in features:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    examples = read_squad_examples(input_file=args.train_input_file)  # #
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_file_dir, do_lower_case=True)
    # 14266 条训练数据
    features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length, max_query_length=args.max_query_length,
                                            max_ans_length=args.max_seq_length)
    # 按照9:1划分训练集和测试集，其中测试集用于训练时验证
    train_data = features[:12839]  # 12839条训练数据集
    train_data_demo = features[:1427]
    test_data = features[12839:]  # 1427条验证集

    # 写入训练demo集
    # make_feature_example_datafile(train_data_demo, 'train_demo_feature')
    make_feature_example_datafile(test_data, 'test_feature')
    make_feature_example_datafile(train_data, 'train_feature')
