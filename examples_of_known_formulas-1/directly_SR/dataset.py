import torch
import utils
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset


class BandGapDataset(Dataset):
    def __init__(self, para, data_type, mask=None, isNormalize=True):
        if mask is None:
            mask = [i for i in range(4)]
        mask = mask + [4]
        self.data_type = data_type

        # -------------------- 数据读取 ----------------------------------
        data_file_path = para.data_root + 'train_val.txt'
        train_val_data = np.loadtxt(data_file_path)[:, mask]

        # -------------------- 数据划分 ----------------------------------
        self.train_num = 8000
        self.valid_num = 1000
        self.train_df = deepcopy(train_val_data[:self.train_num])
        self.valid_df = deepcopy(train_val_data[self.train_num:])

        # -------------------- 数据归一化 ---------------------------------
        if isNormalize:
            mean, sigma = np.mean(self.train_df, axis=0), np.std(self.train_df, axis=0)
            sigma[sigma < 1e-2] = 1
            self.train_df = (self.train_df - mean) / sigma
            self.valid_df = (self.valid_df - mean) / sigma
            # min_, max_ = np.min(self.train_df, axis=0), np.max(self.train_df, axis=0)
            # self.train_df = (self.train_df - min_) / (max_ - min_ + 1e-6)
            # self.valid_df = (self.valid_df - min_) / (max_ - min_ + 1e-6)

        # ------------------- 数据格式转换 ---------------------------------
        self.train_df = torch.from_numpy(self.train_df).float()
        self.valid_df = torch.from_numpy(self.valid_df).float()

    def __getitem__(self, idx):
        if self.data_type == 'train':
            x, y = self.train_df[idx, :-1], self.train_df[idx, -1].reshape(-1)  # (,4) (,1)
        else:
            x, y = self.valid_df[idx, :-1], self.valid_df[idx, -1].reshape(-1)  # (,4) (,1)
        return x, y

    def __len__(self):
        if self.data_type == 'train':
            return self.train_num
        else:
            return self.valid_num
