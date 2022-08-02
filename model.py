import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


# Neighbor user short-term intention generator
class ShortAttention(torch.nn.Module):
    def __init__(self, user_num, item_num, args, dropout_rate):
        super(ShortAttention, self).__init__()


        self.dev = args.device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training

        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)


    def forward(self, log_seqs, user_seqs, neighAt_matrices, neigh_mask, args):

        # shape = [neighAt_matrices.shape[0], args.maxlen, 10, args.hidden_units]
        positions = np.tile(np.array(range(args.short_length)), [neighAt_matrices.shape[0] * args.maxlen * 10, 1])
        pos = self.pos_emb(torch.LongTensor(positions).to(self.dev))

        y = torch.cat(torch.split(neighAt_matrices, 1, dim=2), dim=1)
        y = torch.cat(torch.split(y, 1, dim=1), dim=0)
        seqs = torch.squeeze(y)
        seqs += pos
        seqs = self.emb_dropout(seqs)
        log_zeroseqs = np.zeros([neighAt_matrices.shape[0] * args.maxlen * 10, args.short_length], dtype=np.int32)
        timeline_mask = torch.BoolTensor(log_zeroseqs).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        for j in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[j](seqs)
            mha_outputs, _ = self.attention_layers[j](Q, seqs, seqs, attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs *= ~timeline_mask.unsqueeze(-1)
        seqs = torch.sum(seqs, dim=1)
        seqs = torch.squeeze(seqs)
        seqs = seqs.unsqueeze(1).unsqueeze(1)
        output = torch.cat(torch.split(seqs, neighAt_matrices.shape[0], dim=0), dim=1)
        output = torch.cat(torch.split(output, args.maxlen, dim=1), dim=2)

        newkeys = output
        newkeys = newkeys.to(self.dev)
        neigh_matrix = newkeys

        attn_weights = neigh_matrix.matmul(user_seqs.unsqueeze(-1)).squeeze(-1)

        attn_weights = attn_weights / (neigh_matrix.shape[-1] ** 0.5)

        neigh_mask = neigh_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])

        paddings = torch.ones(attn_weights.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(neigh_mask, paddings, attn_weights)  # True:pick padding


        attn_weights = self.softmax(attn_weights)  # code as below invalids pytorch backward rules

        attn_weights = self.dropout(attn_weights)


        outputs = attn_weights.unsqueeze(2).matmul(neigh_matrix).squeeze(2)

        return outputs


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(MultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, neighbor_matrix, neighAt_matrix, abs_pos_K, abs_pos_V, attn_mask):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)
        neighbor_matrix = torch.cat(torch.split(neighbor_matrix, self.head_size, dim=2), dim=0)
        neighAt_matrix = torch.cat(torch.split(neighAt_matrix, self.head_size, dim=2), dim=0)

        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(neighbor_matrix, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(neighAt_matrix, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(attn_mask, paddings, attn_weights)

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(neighbor_matrix)
        outputs += attn_weights.matmul(neighAt_matrix)

        outputs += attn_weights.matmul(abs_pos_V_)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)  # div batch_size

        return outputs

# Neighbor user long-term intention generator
class NeighAttention(torch.nn.Module):

    def __init__(self, hidden_size, head_num, dropout_rate, dev, args):
        super(NeighAttention, self).__init__()

        #         self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        # //表示整数除法
        self.head_size = hidden_size // head_num
        self.dev = dev
        self.gru_layer = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

    def forward(self, queries, keys, neigh_mask, args):
        Q_ = torch.cat(torch.split(queries, self.head_size, dim=2), dim=0)
        shape = [keys.shape[0], args.maxlen, 10, args.maxlen]
        # newkeys = torch.zeros(size=shape)


        y = torch.cat(torch.split(keys, 1, dim=2), dim=1)
        y = torch.cat(torch.split(y, 1, dim=1), dim=0)
        input = torch.squeeze(y)
        out, hidden = self.gru_layer(input)
        gruout = out

        positions = np.tile(np.array(range(args.short_length)), [keys.shape[0] * args.maxlen * 10, 1])
        pos = self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = gruout[:, -10:, :] + pos
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        for j in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[j](seqs)
            mha_outputs, _ = self.attention_layers[j](Q, seqs, seqs,
                                                      attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
        output = seqs

        output = torch.sum(output, dim=1)
        output = output + hidden
        output = torch.squeeze(output)
        output = output.unsqueeze(1).unsqueeze(1)
        output = torch.cat(torch.split(output, keys.shape[0], dim=0), dim=1)
        output = torch.cat(torch.split(output, args.maxlen, dim=1), dim=2)
        newkeys = output
        newkeys = newkeys.to(self.dev)
        neigh_matrix = torch.cat(torch.split(newkeys, self.head_size, dim=3), dim=0)
        # ....................................................GRU的部分
        attn_weights = neigh_matrix.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (neigh_matrix.shape[-1] ** 0.5)



        neigh_mask = neigh_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        paddings = torch.ones(attn_weights.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(neigh_mask, paddings, attn_weights)  # True:pick padding

        attn_weights = self.softmax(attn_weights)  # code as below invalids pytorch backward rules
        attn_weights = self.dropout(attn_weights)
        outputs = attn_weights.unsqueeze(2).matmul(neigh_matrix).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, queries.shape[0], dim=0), dim=2)  # div batch_size

        return outputs


class NISRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(NISRec, self).__init__()

        self.args = args
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.abs_pos_K_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.neigh_attention_layers = torch.nn.ModuleList()
        self.neigh_attention_layernorms = torch.nn.ModuleList()
        self.short_attention_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

            self.neigh_attention_layernorms.append(new_layernorm)

            new_SA_layer = ShortAttention(self.args, item_num, args, args.dropout_rate)
            self.short_attention_layers.append(new_SA_layer)

            new_layer = NeighAttention(args.hidden_units,
                                       args.num_heads,
                                       args.dropout_rate,
                                       args.device,
                                       args)
            self.neigh_attention_layers.append(new_layer)
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = MultiHeadAttention(args.hidden_units,
                                                args.num_heads,
                                                args.dropout_rate,
                                                args.device)
            # 这部分还得是用自己设计的Attention
            # new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
            #                                                 args.num_heads,
            #                                                 args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs, neighbor_matrices, neighAt_matrices):

        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])

        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        neighbor_matrices = self.item_emb(torch.LongTensor(neighbor_matrices).to(self.dev))
        neighbor_matrices = self.emb_dropout(neighbor_matrices)

        neighAt_matrices = self.item_emb(torch.LongTensor(neighAt_matrices).to(self.dev))
        neighAt_matrices = self.emb_dropout(neighAt_matrices)

        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        neighbor_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            sa_outputs = self.short_attention_layers[i](log_seqs, seqs, neighAt_matrices,
                                                        neighbor_mask, self.args)
        for i in range(len(self.attention_layers)):
            Q = self.neigh_attention_layernorms[i](seqs)  # PyTorch mha requires time first fmt
            nei_outputs = self.neigh_attention_layers[i](Q, neighbor_matrices,
                                                         neighbor_mask, self.args)
            result = Q + nei_outputs
        neighAt_matrix = sa_outputs
        neighbor_matrix = result
        # 在这边可以尝试直接把neighbor矩阵加到item_matrix上去
        for i in range(len(self.attention_layers)):
            #             seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            # def forward(self, queries, keys, neighbor_matrix, attn_mask, abs_pos_K, abs_pos_V):
            #  mha_outputs = self.attention_layers[i](Q, seqs, neighbor_matrix,
            #                                                    timeline_mask, attention_mask,
            #                                                    neighbor_mask,
            #                                                    time_matrix_K, time_matrix_V,
            #                                                    abs_pos_K, abs_pos_V)
            mha_outputs = self.attention_layers[i](Q, seqs, neighbor_matrix, neighAt_matrix, abs_pos_K, abs_pos_V,
                                                   attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            #             seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, neighbor_matrices, neighAt_matrices, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs, neighbor_matrices, neighAt_matrices)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, neigh_matrices, neighAt_matrices, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs, neigh_matrices, neighAt_matrices)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)
