import os
import time
import utils
import torch
import models
import random
import dataset
import datetime
import numpy as np
import torch.nn as nn
from para import Parameter
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


if __name__ == '__main__':

    para = Parameter().args
    device = torch.device('cpu')

    valid_dataset = dataset.BandGapDataset(para, 'valid')
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=2000,
        shuffle=False,
        pin_memory=True
    )
    print('length_of_valid: ', len(valid_dataset))

    # Networks
    baseline_model = models.MultiBranchModel([4], para.layer_num).to(device)
    checkpoint0 = torch.load(para.save_dir + 'baseline_ncr_0.0/latest.pth')
    baseline_model.load_state_dict(checkpoint0['model'])
    baseline_model.eval()

    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            y_pred = baseline_model(x)
    # evaluate
    pred, gt = y_pred.squeeze().cpu().numpy(), y.squeeze().cpu().numpy()
    MSE = np.power(pred - gt, 2).mean()
    gt_mean = np.mean(gt)
    molecule, denominator = 0, 0
    for i in range(gt.shape[0]):
        molecule += (gt[i] - pred[i]) ** 2
        denominator += (gt[i] - gt_mean) ** 2
    R2 = 1 - molecule / denominator
    print("Baseline MSE:", MSE)
    print("Baseline R2:", R2)

    # plot
    # 绘图前的一些操作
    th = 0.4
    plt.figure(figsize=(8, 6), dpi=200)
    prediction, label = pred, gt
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
    plt.legend(loc='best', prop=font1), plt.savefig(para.save_dir + 'baseline.jpg')

