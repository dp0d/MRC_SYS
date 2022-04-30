import pickle

def creat_test_example(text,question,result):
    examples = [({
        'id': -1,
        'question_text': question.strip(),
        'doc_tokens': [{'doc_tokens': text}],
        'answers': -1})]
    with open(result, 'wb') as fw:
        pickle.dump(examples, fw)