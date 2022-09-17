
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random



import scipy
from sklearn.metrics.pairwise import cosine_similarity


K = 5
EMBEDDING_SIZE = 100
batch_size = 32
lr = 0.2



class NodeEmbeddingDataset(tud.Dataset):
    def __init__(self, User, Item, usernum, item_count):
        ''' text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            word_freqs: the frequency of each word
        '''
        super(NodeEmbeddingDataset, self).__init__()
        self.User = User
        self.Item = Item
        self.usernum = usernum
        self.maxlen = 20
        self.item_count = item_count
        self.max_item_freq = usernum//10

    def __len__(self):
        return self.usernum

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''
        center_nodes = idx + 1
        idx = self.maxlen - 1
        neigh_seq = []
        for i in reversed(self.User[center_nodes][:-1]):
            item_freq = self.item_count[i]
            if (item_freq > self.max_item_freq):
                continue
            neigh_seq.append(i)
            idx -= 1
            if idx == -1: break
        if(len(neigh_seq) == 0):
            neigh_seq.append(self.User[center_nodes][-1])
        pos_indices = self.computeNeigh(neigh_seq, self.Item)
        pos_nodes = pos_indices
        neg_nodes = np.zeros([K * pos_nodes.shape[0]], dtype=np.int32)
        count=0
        for _ in range(K * len(pos_nodes)-1):
            t = np.random.randint(1, self.usernum)
            while t in pos_nodes:
                t = np.random.randint(1, self.usernum)
            neg_nodes[count] = t
            count+=1
        return center_nodes, pos_nodes, neg_nodes
    def computeNeigh(self, item_seq,  Item):
        # pos_indices = []
        maxlen = 50
        pos_indices = np.zeros([maxlen], dtype=np.int32)
        firstDimen = -1
        count = 0
        for i in item_seq:
            if (i == 0):
                continue
            for k in Item[i]:
                pos_indices[count] = k
                count+=1
                if(count>=50):
                    break
            if (count >=50):
                break
        while(count<50):
            last_one = pos_indices[-1]
            pos_indices.append(last_one)
            count+=1
        return pos_indices



class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size, device):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        # self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.dev = device

    def forward(self, input_labels, pos_labels, neg_labels):
        ''' input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]

            return: loss, [batch_size]
        '''

        input_embedding = self.in_embed(torch.LongTensor(input_labels).to(self.dev))  # [batch_size, embed_size]
        pos_embedding = self.in_embed(torch.LongTensor(pos_labels).to(self.dev))  # [batch_size, (window * 2), embed_size]
        neg_embedding = self.in_embed(torch.LongTensor(neg_labels).to(self.dev) ) # [batch_size, (window * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1]

        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]

        neg_dot = torch.bmm(neg_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg

        return -loss

    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()





