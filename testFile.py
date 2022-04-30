import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_Layer(nn.Module):

    # 用来实现mask-attention layer
    def __init__(self, hidden_dim, is_bi_rnn):
        super(Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim
        self.is_bi_rnn = is_bi_rnn

        # 下面使用nn的Linear层来定义Q，K，V矩阵
        if is_bi_rnn:
            # 是双向的RNN
            self.Q_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.K_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.V_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        else:
            # 单向的RNN
            self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs, lens):

        size = inputs.size()
        # 计算生成QKV矩阵
        Q = self.Q_linear(inputs)
        K = self.K_linear(inputs).permute(0, 2, 1)  # 先进行一次转置
        V = self.V_linear(inputs)

        # 还要计算生成mask矩阵
        max_len = max(lens)  # 最大的句子长度，生成mask矩阵
        print(max_len)
        sentence_lengths = torch.Tensor(lens)  # 代表每个句子的长度
        mask = torch.arange(sentence_lengths.max().item())[None, :] < sentence_lengths[:, None]
        mask = mask.unsqueeze(dim=1)  # [batch_size, 1, max_len]
        mask = mask.expand(size[0], max_len, max_len)  # [batch_size, max_len, max_len]
        print(mask)

        # print('\nmask is :', mask.size())

        # 下面生成用来填充的矩阵
        padding_num = torch.ones_like(mask)
        padding_num = -2 ** 31 * padding_num.float()

        # print('\npadding num is :', padding_num.size())

        # 下面开始计算啦
        alpha = torch.matmul(Q, K)

        # 下面开始mask
        alpha = torch.where(mask, alpha, padding_num)
        # 下面开始softmax
        alpha = F.softmax(alpha, dim=2)
        # print('\nalpha is :', alpha)

        out = torch.matmul(alpha, V)

        return out


if __name__ == '__main__':
    out = torch.ones(3, 10, 128)  # 这里假设是RNN的输出，维度分别是[batch_size, max_len, hidden_size * 2]
    att_L = Attention_Layer(64, True)  # 参数分别是 hidden_size, 双向RNN：True
    lens = [7, 10, 4]  # 一个batch文本的真实长度

    att_out = att_L(out, lens)  # 开始计算
    print(att_out)
    print([att_out.shape[1]])


