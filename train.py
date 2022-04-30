import os
import args
import torch
import random
import numpy as np
from tqdm import tqdm
from evaluate import evaluate
from optimizer import BertAdam
from dataset.dataloader import Dureader
from dataset.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from model_dir.modeling import BertForQuestionAnswering, BertConfig
from predicting_dev import eval_all
from predict.metric.extracted_script import assessment


# 设置随机数种子

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)

# 记录超参
import wandb
hyperparameter_defaults = dict(
    dropout = BertConfig.from_json_file('model_dir/bert_config.json').hidden_dropout_prob,
    batch_size = args.batch_size,
    learning_rate = args.learning_rate,
    epochs = args.num_train_epochs,
    max_len = args.max_seq_length,
    down_net = args.down_net,
    #attention_head_num = args.attention_head_num
    
    )

# # Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults,project="attention_test")

# os.system('wandb login e9d8966c81023a544b2aac0c670e7f026bd59141')
# wandb.init(project="test_project",)
# config = wandb.config  # Initialize config
# config.batch_size = args.batch_size  # input batch size for training (default:64)
# config.epochs = args.num_train_epochs  # number of epochs to train(default:10)
# config.lr = args.learning_rate  # learning rate(default:0.01)
# config.dropout = BertConfig.from_json_file('model_dir/bert_config.json').hidden_dropout_prob
print(hyperparameter_defaults)


def train():
    # 第一步加载预训练bert，用之前的数据把BERT在领域内先微调一下
    model = BertForQuestionAnswering.from_pretrained('./roberta_wwm_ext',
                                                     cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                            'distributed_{}'.format(-1)))
    model = model.cuda()

    # 准备 optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=0.1,
                         t_total=args.num_train_optimization_steps)
    # 准备数据
    data = Dureader()
    train_dataloader, dev_dataloader = data.train_iter, data.dev_iter

    best_loss = 1.0
    #定义测试集F1值
    f1 = 0
    model.train()
    for i in range(args.num_train_epochs):
        print("*************     epoch:({})     *************".format(i))
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Epoch")):
            input_ids, input_mask, input_ids_q, input_mask_q, \
            segment_ids, can_answer, start_positions, end_positions = \
                batch.input_ids, batch.input_mask, batch.input_ids_q, batch.input_mask_q, \
                batch.segment_ids, batch.can_answer, \
                batch.start_position, batch.end_position

            # 数据放在cuda上
            input_ids, input_mask, input_ids_q, input_mask_q, \
            segment_ids, can_answer, start_positions, end_positions = \
                input_ids.cuda(), input_mask.cuda(), input_ids_q.cuda(), input_mask_q.cuda(), \
                segment_ids.cuda(), can_answer.cuda(), start_positions.cuda(), \
                end_positions.cuda()
            # 计算loss
            total_loss, start_logits, end_logits = model(input_ids, input_ids_q, token_type_ids=segment_ids,
                                                         attention_mask=input_mask,
                                                         attention_mask_q=input_mask_q,
                                                         can_answer=can_answer,
                                                         start_positions=start_positions,
                                                         end_positions=end_positions)

            # 記錄
            wandb.log({"total_loss": total_loss})

            #             main_losses += main_loss.mean().item()
            #             start_losses += start_loss.mean().item()
            if step % 1000 == 0 and step:
                print('After step:{}, total_loss/step  is {}'.format(step, total_loss / step))
            elif step == 0:
                print('After step:{}, total_loss is {}'.format(step, total_loss))
            #             total_loss = total_loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps
            total_loss.backward()

            # 更新梯度
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()  # 清空梯度

            # 验证
            if step % args.log_step == 4:
                eval_loss = evaluate(model, dev_dataloader)
                wandb.log({"best_eval_loss": best_loss})  # wandb
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    # torch.save(model.state_dict(), './model_dir/' + "best_model")
                    print("##!Right now,the eval_loss is {},and the best_eval_loss has been updated.".format(eval_loss))
                else:
                    print("##!The eval_loss is {}".format(eval_loss))
                model.train()
        chekpoint_name = args.output_dir + "checkpoint_epoch_" + str(i)
        torch.save(model.state_dict(), chekpoint_name)
        print('checkpoint_epoch_{} has been saved'.format(i))
        eval_all(chekpoint_name, args.config_file)
        F1 = assessment(args.predict_ground_truth_files,args.predict_result_files)
    wandb.log({"F1": F1})



if __name__ == "__main__":
    train()
