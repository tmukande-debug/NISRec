import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# sampler for batch generation
from tqdm import tqdm


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def NeighborAt_Op(user_train, usernum, maxlen, Item, User, n_hots, label, short_length, pre_model, device):
    data_train = dict()
    #      for user in tqdm(range(1, usernum + 1), desc='Preparing neighborAt matrix'):
    for user in tqdm(range(1, usernum + 1), desc='Preparing' + label + 'neighborAt matrix'):
        neighbor_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            neighbor_seq[idx] = i
            idx -= 1
            if idx == -1: break
        data_train[user] = computeNeighAt(neighbor_seq, user_train, Item, maxlen, user, n_hots, short_length, pre_model,
                                          device)
    return data_train


def computeNeighAt(item_seq, User, Item, maxlen, user, n_hots, short_length, pre_model, device):
    n_hots = torch.tensor(n_hots, dtype=torch.float)
    size = item_seq.shape[0]
    neigh_matrix = np.zeros([maxlen, 10, short_length], dtype=np.int32)
    firstDimen = -1

    for i in item_seq:
        count = -1
        firstDimen += 1
        if (i == 0):
            continue
        userList = []
        rank = dict()
        for k in Item[i]:
            with torch.no_grad():
                #                 ulist = [ 348,1147,1879,1041,169]
                ufeat = pre_model.in_embed(torch.LongTensor([user]).to(device))
                kfeat = pre_model.in_embed(torch.LongTensor([k]).long().to(device))
                cos = torch.cosine_similarity(ufeat, kfeat)
                rank[k] = cos
        rank_sorted = sorted(rank.items(), key=lambda x: x[1], reverse=True)
        if len(rank_sorted) <= 10:
            length = len(rank_sorted)
        else:
            length = 10
        for m in range(length):
            userList.append(rank_sorted[m][0])

        count = -1
        for k in userList:
            idx = short_length - 1
            count += 1
            counter = 0
            for m in User[user][:-1]:
                if m == i:
                    break
                counter += 1
            for j in reversed(User[k][:counter]):
                neigh_matrix[firstDimen][count][idx] = j
                idx -= 1
                if idx == -1: break

    return neigh_matrix


def Neighbor_Op(User, usernum, maxlen, Item, n_hots, label, pre_model, device):
    data_train = dict()
    for user in tqdm(range(1, usernum + 1), desc='Preparing' + label + 'neighbor matrix'):
        neighbor_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(User[user][:-1]):
            neighbor_seq[idx] = i
            idx -= 1
            if idx == -1: break
        data_train[user] = computeNeigh(neighbor_seq, User, Item, maxlen, user, n_hots, pre_model, device)
    return data_train


def computeNeigh(item_seq, User, Item, maxlen, user, n_hots, pre_model, device):
    n_hots = torch.tensor(n_hots, dtype=torch.float)
    neigh_matrix = np.zeros([maxlen, 10, maxlen], dtype=np.int32)
    firstDimen = -1

    for i in item_seq:
        count = -1
        firstDimen += 1
        if (i == 0):
            continue
        userList = []
        rank = dict()
        for k in Item[i]:
            with torch.no_grad():
                #                 ulist = [ 348,1147,1879,1041,169]
                ufeat = pre_model.in_embed(torch.LongTensor([user]).to(device))
                kfeat = pre_model.in_embed(torch.LongTensor([k]).long().to(device))
                cos = torch.cosine_similarity(ufeat, kfeat)
                rank[k] = cos
        rank_sorted = sorted(rank.items(), key=lambda x: x[1], reverse=True)
        if len(rank_sorted) <= 10:
            length = len(rank_sorted)
        else:
            length = 10
        for m in range(length):
            userList.append(rank_sorted[m][0])

        count = -1
        for k in userList:
            idx = maxlen - 1
            count += 1
            for j in reversed(User[k][:-1]):
                neigh_matrix[firstDimen][count][idx] = j
                idx -= 1
                if idx == -1: break

    return neigh_matrix


def sample_function(user_train, neighbor_matrix, neighborAt_matrix, usernum, itemnum, batch_size, maxlen, result_queue,
                    SEED):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1
        ts = set(user_train[user])
        neigh_matrix = neighbor_matrix[user]
        neighAt_matrix = neighborAt_matrix[user]
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt

            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, neigh_matrix, neighAt_matrix, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, neighbor_matrix, neighborAt_matrix, usernum, itemnum, batch_size=64, maxlen=10,
                 n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      neighbor_matrix,
                                                      neighborAt_matrix,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def reduceNeighUser(Item, User, usersize, itemsize, maxlen):
    user_matrix = np.zeros([usersize + 1, maxlen], dtype=np.int64)
    for i in range(usersize + 1):
        if (i == 0):
            continue
        if (len(User[i]) > maxlen):
            start = len(User[i]) - maxlen
        else:
            start = 0
        count = 0
        for j in range(start, len(User[i])):
            user_matrix[i][count] = User[i][j]
            count += 1
    n = itemsize
    user_matrix = torch.from_numpy(user_matrix)

    one_hots = torch.nn.functional.one_hot(user_matrix, n)  # size=(15, 15, n)
    n_hots = torch.sum(one_hots, dim=1)
    n_hots[:, 0] = 0
    return n_hots


def cleanAndsort(Item, User, maxlen):
    User_all = defaultdict(list)
    Item_all = defaultdict(list)
    User_filted = defaultdict(list)
    Item_filted = defaultdict(list)
    user_set = set()
    item_set = set()

    train_User = defaultdict(list)
    train_Item = defaultdict(list)
    for user, items in User.items():

        user_set.add(user)
        User_all[user] = items[:-1]
        User_filted[user] = items[:-2]
        train_User[user] = items[:-3]
        for item in items:
            item_set.add(item)
        for item in items[:-3]:
            #             print(item)
            Item_filted[item].append(user)
            train_Item[item].append(user)
            Item_all[item].append(user)

        try:
            Item_all[items[-3]].append(user)
            Item_filted[items[-3]].append(user)
        except:
            continue
        try:
            Item_all[items[-2]].append(user)
        except:
            continue

    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    User_all_res = dict()
    User_res = dict()
    train_User_res = dict()

    Item_all_res = dict()
    Item_res = dict()
    train_Item_res = dict()

    for user, items in User_all.items():
        User_all_res[user_map[user]] = list(map(lambda x: item_map[x], items))
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: item_map[x], items))
    for user, items in train_User.items():
        train_User_res[user_map[user]] = list(map(lambda x: item_map[x], items))

    for item, users in Item_all.items():
        Item_all_res[item_map[item]] = list(map(lambda x: user_map[x], users))
    for item, users in Item_filted.items():
        Item_res[item_map[item]] = list(map(lambda x: user_map[x], users))
    for item, users in train_Item.items():
        train_Item_res[item_map[item]] = list(map(lambda x: user_map[x], users))
    n_hots = reduceNeighUser(Item_res, User_res, len(user_set), len(item_set) + 1, maxlen)

    return User_all_res, Item_all_res, User_res, Item_res, train_User_res, train_Item_res, len(user_set), len(
        item_set), n_hots, user_map


# train/val/test data generation
def data_partition(fname, maxlen):
    User = defaultdict(list)
    Item = defaultdict(list)

    user_train = {}
    user_train_valid = {}
    user_train_test = {}
    user_valid = {}
    user_test = {}

    print('Preparing data...')
    f = open('data/%s.txt' % fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        #         try:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open('data/%s.txt' % fname, 'r')
    for line in f:

        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        if user_count[u] < 5 or item_count[i] < 5:
            continue
        User[u].append(i)
        Item[i].append(u)
    f.close()
    all_User, all_Item, User, Item, train_User, train_Item, usernum, itemnum, n_hots, user_map = cleanAndsort(Item,
                                                                                                              User,
                                                                                                              maxlen)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_train_valid[user] = User[user]
            user_train_test[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_train_valid[user] = User[user][:-1]
            user_train_test[user] = User[user]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print('Preparing done...')
    return [user_train, user_train_valid, user_train_test, user_valid, user_test, usernum, itemnum, all_User, all_Item,
            User, Item, train_User, train_Item, n_hots, user_map, item_count]


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, neigh_matrix, test_neighAt_matrix, args):
    [train, train_valid, train_test, valid, test, usernum, itemnum, all_User, all_Item, User, Item, train_User,
     train_Item, n_hots, user_map, item_count] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        neighbor_matrix = neigh_matrix[u]
        test_neighborAt_matrix = test_neighAt_matrix[u]
        u, seq = np.array(u).reshape(1, -1), np.array(seq).reshape(1, -1)
        neighbor_matrix = np.array(neighbor_matrix).reshape(1, args.maxlen, 10, args.maxlen)
        test_neighborAt_matrix = np.array(test_neighborAt_matrix).reshape(1, args.maxlen, 10, args.short_length)
        predictions = -model.predict(u, seq, neighbor_matrix, test_neighborAt_matrix, item_idx)
        predictions = predictions[0]  # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, neigh_matrix, test_neighAt_matrix, args):
    [train, train_valid, train_test, valid, test, usernum, itemnum, all_User, all_Item, User, Item, train_User,
     train_Item, n_hots, user_map, item_count] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        neighbor_matrix = neigh_matrix[u]
        test_neighborAt_matrix = test_neighAt_matrix[u]
        u, seq = np.array(u).reshape(1, -1), np.array(seq).reshape(1, -1)
        neighbor_matrix = np.array(neighbor_matrix).reshape(1, args.maxlen, 10, args.maxlen)
        test_neighborAt_matrix = np.array(test_neighborAt_matrix).reshape(1, args.maxlen, 10, args.short_length)
        predictions = -model.predict(u, seq, neighbor_matrix, test_neighborAt_matrix, item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
