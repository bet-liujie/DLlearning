import torch
import torchkeras
from TorchCRF import CRF
import torch.utils.data
from tqdm import tqdm
import datetime
import time
import copy
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, classification_report

import numpy as np
import pandas as pd

cwd_dir = '/home/jie/deeplearning'
data_base_dir = cwd_dir + '/data/rmrb/'
save_dir = cwd_dir + 'save/'
imgs_dir = cwd_dir + 'imgs/'

pad_token = '<pad>'
pad_id = 0
unk_token = '<unk>'
unk_id = 1

tag_to_id = {'<pad>': 0, 'O': 1, 'B-LOC': 2, 'I-LOC': 3, 'B-PER': 4, 'I-PER': 5, 'B-ORG': 6, 'I-ORG': 7}
id_to_tag = {id: tag for tag, id in tag_to_id.items()}
word_to_id = {'<pad>': 0, '<unk>': 1}
tags_num = len(tag_to_id)

LR = 1e-3
EPOCHS = 30

maxlen = 60
# total_words = 4000

embedding_dim = 100
hidden_size = 128
batch_size = 512


# 读取数据  数据格式：字 tag
def read_data(filepath):
    sentences = []
    tags = []
    with open(filepath, 'r', encoding='utf-8') as f:
        tmp_sentence = []
        tmp_tags = []
        for line in f:
            if line == '\n' and len(tmp_sentence) != 0:
                assert len(tmp_sentence) == len(tmp_tags)
                sentences.append(tmp_sentence)
                tags.append(tmp_tags)
                tmp_sentence = []
                tmp_tags = []
            else:
                line = line.strip().split(' ')
                tmp_sentence.append(line[0])
                tmp_tags.append(line[1])
        if len(tmp_sentence) != 0:
            assert len(tmp_sentence) == len(tmp_tags)
            sentences.append(tmp_sentence)
            tags.append(tmp_tags)
    return sentences, tags


sentences, tags = read_data(data_base_dir + 'train.txt')
print(sentences[0], tags[0])

s_lengths = [len(s) for s in sentences]
print('最大句子长度：{}, 最小句子长度：{}, 平均句子长度：{:.2f}, 句子长度中位数：{:.2f}'.format(
    max(s_lengths), min(s_lengths), np.mean(s_lengths), np.median(s_lengths)))
df_len = pd.DataFrame({'s_len': s_lengths})
print(df_len.describe())


def build_vocab(sentences):
    global word_to_id
    for sentence in sentences:  # 建立word到索引的映射
        for word in sentence:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
    return word_to_id


word_to_id = build_vocab(sentences)
print('vocab size:', len(word_to_id))


def convert_to_ids_and_padding(seqs, to_ids):
    ids = []
    for seq in seqs:
        if len(seq) >= maxlen:  # 截断
            ids.append([to_ids[w] if w in to_ids else unk_id for w in seq[:maxlen]])
        else:  # padding
            ids.append([to_ids[w] if w in to_ids else unk_id for w in seq] + [0] * (maxlen - len(seq)))

    return torch.tensor(ids, dtype=torch.long)


def load_data(filepath, word_to_id, shuffle=False):
    sentences, tags = read_data(filepath)

    inps = convert_to_ids_and_padding(sentences, word_to_id)
    trgs = convert_to_ids_and_padding(tags, tag_to_id)

    inp_dset = torch.utils.data.TensorDataset(inps, trgs)
    inp_dloader = torch.utils.data.DataLoader(inp_dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=4)
    return inp_dloader


# 查看data pipeline是否生效
inp_dloader = load_data(data_base_dir + 'train.txt', word_to_id)
sample_batch = next(iter(inp_dloader))
print('sample_batch:', len(sample_batch), sample_batch[0].size(), sample_batch[0].dtype,
      sample_batch[1].size(), sample_batch[1].dtype)  # [b,60] int64

ngpu = 1
device = 'cpu'


class BiLSTM_CRF(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BiLSTM_CRF, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_id)
        self.bi_lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size // 2, batch_first=True,
                                     bidirectional=True)  # , dropout=0.2)
        self.hidden2tag = torch.nn.Linear(hidden_size, tags_num)

        self.crf = CRF(num_tags=tags_num, batch_first=True)

    def init_hidden(self, batch_size):
        # device = 'cpu'
        _batch_size = batch_size // ngpu
        return (torch.randn(2, _batch_size, self.hidden_size // 2, device=device),
                torch.randn(2, _batch_size, self.hidden_size // 2,
                            device=device))  # ([b=1,2,hidden_size//2], [b=1,2,hidden_size//2])

    def forward(self, inp):  # inp [b, seq_len=60]
        self.bi_lstm.flatten_parameters()

        embeds = self.embedding(inp)  # [b,seq_len]=>[b, seq_len, embedding_dim]
        lstm_out, _ = self.bi_lstm(embeds,
                                   None)  # lstm_out: =>[b, seq_len, hidden_size], #####################################################
        # lstm_out, self.hidden = self.bi_lstm(embeds, self.hidden)  # lstm_out: =>[b, seq_len, hidden_size], #####################################################
        # h_n: ([b,2,hidden_size//2], c_n: [b,2,hidden_size//2])

        logits = self.hidden2tag(lstm_out)  # [b, seq_len, hidden_size]=>[b, seq_len, tags_num]
        return logits  # [b, seq_len=60, tags_num=10]

    # 计算CRF 条件对数似然，并返回其负值作为loss
    def crf_neg_log_likelihood(self, inp, tags, mask=None, inp_logits=False):  # [b, seq_len, tags_num], [b, seq_len]
        if inp_logits:
            logits = inp
        else:
            logits = self.forward(inp)

        if mask is None:
            mask = torch.logical_not(
                torch.eq(tags, torch.tensor(0)))  # =>[b, seq_len],每个元素为bool值，如果序列中有pad，则mask相应位置就为False
            mask = mask.type(torch.uint8)

        crf_llh = self.crf(logits, tags, mask,
                           reduction='mean')  # Compute the conditional log likelihood of a sequence of tags given emission scores
        # crf_llh = self.crf(logits, tags, mask) # Compute the conditional log likelihood of a sequence of tags given emission scores
        return -crf_llh

    def crf_decode(self, inp, mask=None, inp_logits=False):
        if inp_logits:
            logits = inp
        else:
            logits = self.forward(inp)

        if mask is None and inp_logits is False:
            mask = torch.logical_not(
                torch.eq(inp, torch.tensor(0)))  # =>[b, seq_len],每个元素为bool值，如果序列中有pad，则mask相应位置就为False
            mask = mask.type(torch.uint8)

        return self.crf.decode(emissions=logits, mask=mask)


# 查看模型
model = BiLSTM_CRF(len(word_to_id), hidden_size)
torchkeras.summary(model, input_shape=(60,), input_dtype=torch.int64)
