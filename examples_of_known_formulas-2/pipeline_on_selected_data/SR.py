import pysr
pysr.install()
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

    # Setting the random seed
    torch.manual_seed(para.seed)
    torch.cuda.manual_seed(para.seed)
    random.seed(para.seed)
    np.random.seed(para.seed)

    # 加载综合模型以及特征情况
    load_dir = 'checkpoints/feature-num=4/'
    mask_read = np.loadtxt(load_dir + 'mask.txt')
    mask = [int(i) for i in mask_read]

    integrative_model = models.MultiBranchModel([len(mask)], para.layer_num).to(device)
    checkpoint = torch.load(load_dir + '/latest.pth')
    integrative_model.load_state_dict(checkpoint['model'])

    train_dataset = dataset.BandGapDataset(para, 'train', mask, isNormalize=False)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        pin_memory=True
    )

    val_dataset = dataset.BandGapDataset(para, 'val', mask, isNormalize=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        pin_memory=True
    )

    # 符号回归模型拟合
    logger = utils.get_logger(filename='Formula regression.log', name='Formula regression')
    learner = SR_learner.BaseLearner()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
    sympy_expression = learner.fit(x, y)
    logger.info('Sub_expression: {}'.format(sympy_expression))

    # 在验证集上验证模型结果
    for val_x, val_y in val_loader:
        val_x, val_y = val_x.to(device), val_y.to(device)
    val_y_pred = learner.get_prediction(val_x)
    val_y_pred, val_y_gt = val_y_pred.squeeze(), val_y.squeeze().numpy()
    val_R2 = utils.getR2(val_y_gt, val_y_pred)
    print('The valid R2 is {:.4f}'.format(val_R2))
    logger.info('The valid R2 is {:.4f}'.format(val_R2))

    # 测试模型
    data_file_path = para.data_root + 'test.txt'
    test_data = np.loadtxt(data_file_path)[:, mask + [7]]
    test_x, test_y = test_data[:, :-1], test_data[:, -1]

    test_y_pred = learner.get_prediction(test_x)
    test_y_pred, test_y_gt = test_y_pred.squeeze(), test_y.squeeze()
    test_R2 = utils.getR2(test_y_gt, test_y_pred)
    print('The test R2 is {:.4f}'.format(test_R2))
    logger.info('The test R2 is {:.4f}'.format(test_R2))


    # 测试集绘图
    plt.figure(figsize=(8, 6), dpi=200)
    prediction, label = test_y_pred, test_y_gt
    # 绘制散点图
    font1 = {'size': 14}
    x = range(len(prediction))
    j = np.argsort(label)
    label, prediction = label[j], prediction[j]
    # prediction.sort(), label.sort()
    plt.plot(x, label, c='orangered', linewidth=4, linestyle='-', label='label')
    plt.scatter(x, prediction, c='navy', s=30, marker='x', label='fitted', alpha=1.0)
    plt.xlabel("Samples", fontdict={'size': 14})  # X轴标题及字号
    plt.ylabel("Salinities", fontdict={'size': 14})  # Y轴标题及字号
    plt.legend(loc='best', prop=font1), plt.savefig('SR.jpg')


