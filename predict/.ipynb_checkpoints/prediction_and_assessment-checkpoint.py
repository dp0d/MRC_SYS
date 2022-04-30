from predicting_dev import eval_all
from metric.extracted_script import assessment
import sys
sys.path.append("../")
from args import num_train_epochs

if __name__ == '__main__':
    output_model_file = "../model_dir/best_model_adv"
    output_config_file = "../model_dir/bert_config.json"
    _ref_dev = './metric/ref_dev.json'
    _predicts_dev = './metric/predicts_dev.json'

    for _epoch in range(num_train_epochs):
        checkpoint_model = "../model_dir/checkpoint_epoch_{}".format(_epoch)
        eval_all(checkpoint_model, output_config_file)
        print(_epoch, ' model performance:')
        assessment(_ref_dev,_predicts_dev)