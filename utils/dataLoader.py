# FlameMind 🔥, 1.0.0 license
"""DataLoder utils."""
import random

import torch

import os
from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np
# from skimage.data import data_dir
from torch.utils.data import Dataset, DataLoader

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # backfire root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.scaler_data import scaler_data, get_features_labels
#===========================================随机分区=================================================#
"""
def seq_data_iter_random(features,labels, batch_size, num_steps):
    #使用随机抽样生成一个小批量序列
    features = features[random.randint(0, num_steps-1):]
    num_subseqs = (len(features) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs*num_steps, num_steps))
    random.shuffle(initial_indices)
    dims = features.shape[-1]
    def get_feature(pos):
        return features.reshape(dims,-1)[:, pos: pos + num_steps]
    def get_label(pos):
        return labels[pos: pos + num_steps]
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size*num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i:i+batch_size]
        X = np.array([get_feature(j) for j in initial_indices_per_batch])
        Y = np.array([get_label(j+1) for j in initial_indices_per_batch])
        yield torch.tensor(X).permute(0,2,1), torch.tensor(Y)
"""
def seq_data_iter_random(features, labels, batch_size, num_steps):
    """使用随机采样生成小批量序列"""
    offset = random.randint(0, num_steps - 1)
    features = features[offset:]
    labels = labels[offset:]

    num_subseqs = (len(features) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    num_batches = len(initial_indices) // batch_size
    for i in range(0, num_batches * batch_size, batch_size):
        batch_indices = initial_indices[i: i + batch_size]
        X = [features[j: j + num_steps] for j in batch_indices]
        Y = [labels[j + 1: j + 1 + num_steps] for j in batch_indices]
        yield torch.stack(X), torch.stack(Y)


#===========================================顺序分区=================================================#
"""
def seq_data_iter_sequential(features,labels, batch_size, num_steps):
    # 使用顺序分区生成一个小批量子序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(features) - offset - 1) // batch_size) * batch_size
    Xs =features[offset: offset + num_tokens].clone().detach()
    Ys =labels[offset + 1: offset + 1 + num_tokens].clone().detach()
    dims = Xs.shape[1]
    Xs = Xs.permute(1,0)
    Xs, Ys = Xs.reshape(dims, batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[-1] // num_steps # 93
    for i in range(0, num_steps * num_batches, num_steps):
        print()
        X = Xs[:, :, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X.permute(1,2,0), Y
"""
def seq_data_iter_sequential(features, labels, batch_size, num_steps):
    """使用顺序分区生成小批量序列"""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(features) - offset - 1) // batch_size) * batch_size

    Xs = features[offset: offset + num_tokens]
    Ys = labels[offset + 1: offset + 1 + num_tokens]

    Xs = Xs.reshape(batch_size, -1, features.shape[-1])  # (B, L, D)
    Ys = Ys.reshape(batch_size, -1)                      # (B, L)

    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i:i + num_steps, :]
        Y = Ys[:, i:i + num_steps]
        yield X, Y
class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, features, labels, batch_size, num_steps, use_random_iter):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.features, self.labels = features, labels
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.features, self.labels, self.batch_size, self.num_steps)

    # def __len__(self):
    #     return (len(self.features) - self.num_steps) // self.batch_size

    def __len__(self):
        num_subseqs = (len(self.features) - 1) // self.num_steps
        return num_subseqs // self.batch_size

def concat_data(data_folder, use_random=True):
    file_list = os.listdir(data_folder)
    if use_random:
        random.shuffle(file_list)
    scalered_features, scalered_labels,scaler = [], [], None
    for filename in file_list:
        file_path = os.path.join(data_folder, filename)
        df = pd.read_csv(file_path)
        features, labels = get_features_labels(df)
        scalered_fea,scaler = scaler_data(features)
        scalered_features.append(torch.tensor(scalered_fea))
        scalered_labels.append(torch.tensor(labels))
    return torch.concat(scalered_features,dim=0),torch.concat(scalered_labels,dim=0),scaler

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-folder", type=str, default=ROOT / "datasets/train", help="train dataset path")
    parser.add_argument("--valid-folder", type=str, default=ROOT / "datasets/valid", help="valid dataset path")
    parser.add_argument("--test-folder", type=str, default=ROOT / "datasets/test", help="test dataset path")
    parser.add_argument("--use-random", type=bool, default=False, help="use random concat")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-steps", type=int, default=10, help="number of steps")
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    train_folder, valid_folder = opt.train_folder, opt.valid_folder
    use_random = opt.use_random
    batch_size, num_steps = opt.batch_size, opt.num_steps
    # =======================================训练集==========================================#
    train_features, train_labels,_= concat_data(data_folder=train_folder,
                                               use_random=use_random)
    print(train_features.shape, train_labels.shape)
    print(train_features[:10,:])
    train_iter = SeqDataLoader(
        train_features, train_labels, batch_size, num_steps, use_random_iter=False)
    print(f"#=======train iter========#")
    for X, Y in train_iter:
        print('X: ', X.shape, '\nY: ', Y.shape)
        break
    print(f"num_batches: {len(train_iter)}")
    # =======================================验证集==========================================#
    valid_features, valid_labels,_ = concat_data(data_folder=valid_folder,
                                               use_random=use_random)
    valid_iter = SeqDataLoader(
        valid_features, valid_labels, batch_size, num_steps, use_random_iter=False)
    print(f"#=======valid iter========#")
    for X, Y in valid_iter:
        print('X: ', X.shape, '\nY: ', Y.shape)
        break
    print(f"num_batches: {len(valid_iter)}")

    for X, Y in seq_data_iter_random(train_features, train_labels, 32, 10):
        print('X: ', X.shape, '\nY: ', Y.shape)
        break
    for X, Y in seq_data_iter_sequential(train_features, train_labels, 32, 10):
        print('X: ', X.shape, '\nY: ', Y.shape)
        break