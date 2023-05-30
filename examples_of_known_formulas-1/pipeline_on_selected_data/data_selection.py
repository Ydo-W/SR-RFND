import os
import time
import utils
import sympy
import torch
import models
import random
import dataset
import pickle
import datetime
import SR_learner
import numpy as np
import torch.nn as nn
from copy import deepcopy
from para import Parameter
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


if __name__ == '__main__':

    para = Parameter().args
    device = torch.device('cpu')

    # -------------------- 数据读取 ----------------------------------
    data_file_path = para.data_root + 'train_val.txt'
    train_val_data = np.loadtxt(data_file_path)

    # -------------------- 数据划分 ----------------------------------
    train_num = 8000
    valid_num = 1000
    ori_train_df = deepcopy(train_val_data[:train_num])
    ori_valid_df = deepcopy(train_val_data[train_num:])

    # -------------------- 数据归一化 ---------------------------------
    # mean, sigma = np.mean(ori_train_df, axis=0), np.std(ori_train_df, axis=0)
    # sigma[sigma < 1e-2] = 1
    # train_df = (ori_train_df[:] - mean) / sigma
    # valid_df = (ori_valid_df[:] - mean) / sigma
    min_, max_ = np.min(ori_train_df, axis=0), np.max(ori_train_df, axis=0)
    train_df = (ori_train_df - min_) / (max_ - min_ + 1e-9)
    valid_df = (ori_valid_df - min_) / (max_ - min_ + 1e-9)

    # ------------------- 数据格式转换 ---------------------------------
    train_df = torch.from_numpy(train_df).float()
    valid_df = torch.from_numpy(valid_df).float()

    # ------------------- 加载模型 ---------------------------------
    baseline_model = models.MultiBranchModel([4], para.layer_num).to(device)
    checkpoint0 = torch.load(para.save_dir + 'baseline_ncr_0.2/latest.pth')
    baseline_model.load_state_dict(checkpoint0['model'])
    baseline_model.eval()

    # ------------------- 模型处理数据 ---------------------------------
    train_x, train_y = train_df[:, :-1], train_df[:, -1].reshape(-1)
    valid_x, valid_y = valid_df[:, :-1], valid_df[:, -1].reshape(-1)
    with torch.no_grad():
        train_y_pred = baseline_model(train_x)
        valid_y_pred = baseline_model(valid_x)

    # ------------------- 进行样本筛选 ---------------------------------
    threshold = 0.03
    train_selected, valid_selected = [], []
    train_y, valid_y = train_y.squeeze().numpy(), valid_y.squeeze().numpy()
    train_y_pred, valid_y_pred = train_y_pred.squeeze().numpy(), valid_y_pred.squeeze().numpy()
    for i in range(train_y.shape[0]):
        if abs(train_y[i] - train_y_pred[i]) < threshold:
            train_selected.append(ori_train_df[i])
    for i in range(valid_y.shape[0]):
        if abs(valid_y[i] - valid_y_pred[i]) < threshold:
            valid_selected.append(ori_valid_df[i])

    # ------------------- 保存筛选结果 ---------------------------------
    print('Length of train_selected:', len(train_selected))
    print('Length of valid_y_selected:', len(valid_selected))
    train_selected, valid_selected = np.array(train_selected), np.array(valid_selected)

    np.savetxt('../datasets/new-feynman-i.12.4/train_selected.txt', train_selected, fmt='%.8e')
    np.savetxt('../datasets/new-feynman-i.12.4/valid_selected.txt', valid_selected, fmt='%.8e')


