import os
import pickle
import time
import torch
import argparse
import torch.utils.data as tud
from model import NISRec
from tqdm import tqdm

from node2vec import NodeEmbeddingDataset, EmbeddingModel
from utils import *


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
# parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--short_length', default=10, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=151, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

# SAVE_MODEL_DIR = f"./saved_models/{args.dataset}"
# if not os.path.isdir(SAVE_MODEL_DIR):
#     os.mkdir(SAVE_MODEL_DIR)
# SAVE_MODEL_PATH = SAVE_MODEL_DIR + f"/checkpoint.{args.dataset}.pth.tar"

dataset = data_partition(args.dataset, args.maxlen)
[user_train, user_train_valid, user_train_test, user_valid, user_test, usernum, itemnum, all_User, all_Item, User, Item, train_User, train_Item, n_hots, user_map, item_count] = dataset

##############
# A special node2vec algorithm for Learning user representations
MAX_USER_SIZE = usernum
EMBEDDING_SIZE = 100
batch_size = 32
dataset = NodeEmbeddingDataset(user_train, train_Item, usernum, item_count)
dataloader = tud.DataLoader(dataset, batch_size, shuffle=True)


pre_model = EmbeddingModel(usernum+1, EMBEDDING_SIZE, args.device)
optimizer = torch.optim.Adam(pre_model.parameters(), lr=1e-3)
pre_model.to(args.device)
pre_model.train()
for e in range(100):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = np.array(input_labels)
        pos_labels = np.array(pos_labels)
        neg_labels = np.array(neg_labels)
        input_labels = input_labels
        pos_labels = pos_labels
        neg_labels = neg_labels

        optimizer.zero_grad()
        loss = pre_model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()

        optimizer.step()

        if i % 10 == 0:
            print('epoch', e, 'iteration', i, loss.item())

############
pre_model.eval()


num_batch = len(user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')


# Data Preparation & Preprocessing for  user-item interaction bipartite graph.
try:
    train_neighbor_matrix = pickle.load(
        open('data/ml-1m/train_neighbor_matrix_%s_%d_%d.pickle' % (args.dataset, 10, 50), 'rb'))
except:
    train_neighbor_matrix = Neighbor_Op(user_train, usernum, args.maxlen, train_Item, n_hots, "train", pre_model, args.device)
    pickle.dump(train_neighbor_matrix,
                open('data/ml-1m/train_neighbor_matrix_%s_%d_%d.pickle' % (args.dataset, 10, 50), 'wb'))

try:
    train_neighborAt_matrix = pickle.load(
        open('data/ml-1m/train_neighborAt_matrix_%s_%d_%d.pickle' % (args.dataset, 10, 10), 'rb'))
except:
    train_neighborAt_matrix = NeighborAt_Op(user_train, usernum, args.maxlen, train_Item, User, n_hots, "train",
                                            args.short_length, pre_model, args.device)
    pickle.dump(train_neighborAt_matrix,
                open('data/ml-1m/train_neighborAt_matrix_%s_%d_%d.pickle' % (args.dataset, 10, 10), 'wb'))

try:
    valid_neighbor_matrix = pickle.load(
        open('data/ml-1m/valid_neighbor_matrix_%s_%d_%d.pickle' % (args.dataset, 10, 50), 'rb'))
except:
    valid_neighbor_matrix = Neighbor_Op(user_train_valid, usernum, args.maxlen, Item, n_hots, "valid", pre_model, args.device)
    pickle.dump(valid_neighbor_matrix,
                open('data/ml-1m/valid_neighbor_matrix_%s_%d_%d.pickle' % (args.dataset, 10, 50), 'wb'))

try:
    valid_neighborAt_matrix = pickle.load(
        open('data/ml-1m/valid_neighborAt_matrix_%s_%d_%d.pickle' % (args.dataset, 10, 10), 'rb'))
except:
    valid_neighborAt_matrix = NeighborAt_Op(user_train_valid, usernum, args.maxlen, Item, User, n_hots, "valid",
                                           args.short_length, pre_model, args.device)
    pickle.dump(valid_neighborAt_matrix,
                open('data/ml-1m/valid_neighborAt_matrix_%s_%d_%d.pickle' % (args.dataset, 10, 10), 'wb'))

try:
    test_neighbor_matrix = pickle.load(
        open('data/ml-1m/test_neighbor_matrix_%s_%d_%d.pickle' % (args.dataset, 10, 50), 'rb'))
except:
    test_neighbor_matrix = Neighbor_Op(user_train_test, usernum, args.maxlen, Item, n_hots, "test", pre_model, args.device)
    pickle.dump(test_neighbor_matrix,
                open('data/ml-1m/test_neighbor_matrix_%s_%d_%d.pickle' % (args.dataset, 10, 50), 'wb'))

try:
    test_neighborAt_matrix = pickle.load(
        open('data/ml-1m/test_neighborAt_matrix_%s_%d_%d.pickle' % (args.dataset, 10, 10), 'rb'))
except:
    test_neighborAt_matrix = NeighborAt_Op(user_train_test, usernum, args.maxlen, Item, User, n_hots, "test",
                                           args.short_length, pre_model, args.device)
    pickle.dump(test_neighborAt_matrix,
                open('data/ml-1m/test_neighborAt_matrix_%s_%d_%d.pickle' % (args.dataset, 10, 10), 'wb'))

# .................................................
sampler = WarpSampler(user_train, train_neighbor_matrix, train_neighborAt_matrix, usernum, itemnum,
                      batch_size=args.batch_size, maxlen=args.maxlen,
                      n_workers=3)
model = NISRec(usernum, itemnum, args).to(args.device)

for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_uniform_(param.data)
    except:
        pass  # just ignore those failed init layers


model.train()  # enable model training

epoch_start_idx = 1
if args.state_dict_path is not None:
    try:
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1
    except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
        print('failed loading state_dicts, pls check file path: ', end="")
        print(args.state_dict_path)
        print('pdb enabled for your quick check, pls type exit() if you do not need it')
        import pdb;

        pdb.set_trace()

if args.inference_only:
    model.eval()
    t_test = evaluate(model, dataset, args)
    print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

T = 0.0
t0 = time.time()



# model.load_state_dict(
#     torch.load('saved_models/Toys_and_Games/checkpoint.256_20_100_2_0.1_time_2_20_32_32_0.003.pth.tar')[
#         'model_state_dict'], strict=True)

# model.eval()
# t_test = evaluate_valid(model, dataset, test_neighbor_matrix, test_neighborAt_matrix, args)
# print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))


for epoch in range(epoch_start_idx, args.num_epochs + 1):
    if args.inference_only: break
    for step in range(num_batch):
        u, train_neigh_matrix, train_neighAt_matrix, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        train_neigh_matrix = np.array(train_neigh_matrix)
        train_neighAt_matrix = np.array(train_neighAt_matrix)
        pos_logits, neg_logits = model(u, seq, train_neigh_matrix, train_neighAt_matrix, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                               device=args.device)
        adam_optimizer.zero_grad()
        indices = np.where(pos != 0)
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        loss.backward()
        adam_optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        t1 = time.time() - t0
        T += t1
        print('Evaluating', end='')
        t_test = evaluate(model, dataset, test_neighbor_matrix, test_neighborAt_matrix, args)
        t_valid = evaluate_valid(model, dataset, valid_neighbor_matrix, valid_neighborAt_matrix, args)
        print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
              % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

        f.write(str(t_valid) + ' ' + str(t_test) + '\n')
        f.flush()
        t0 = time.time()
        model.train()

    # if epoch == args.num_epochs:
    #     torch.save(
    #         {
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': adam_optimizer.state_dict(),
    #             #                     'loss': np.mean(loss),
    #         }, SAVE_MODEL_PATH
    #     )
    #     print("model saved")
    #     folder = args.dataset + '_' + args.train_dir
    #     fname = 'NISRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
    #     fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
    #     torch.save(model.state_dict(), os.path.join(folder, fname))

f.close()
sampler.close()
print("Done")
