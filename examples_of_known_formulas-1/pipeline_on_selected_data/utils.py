import torch
import dataset
import random
import logging
import numpy as np
from para import Parameter
from captum.attr import Saliency
from torch.utils.data import DataLoader
from torch.autograd.functional import hessian


def shuffle_data(data):
    num, _ = data.shape
    lst = list(range(num))
    random.shuffle(lst)
    return data[lst]


def train(device, train_loader, baseline_model, opt, criterion, epoch):
    baseline_model.train()
    loss_recoder = AverageMeter()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        baseline_model.zero_grad()
        y_pred = baseline_model(x)  # (64, 1)
        loss = criterion(y, y_pred)
        loss_recoder.update(loss.detach().item(), y.shape[0])
        loss.backward()
        opt.step()
    # ---------------- info printing ------------------
    print('Epoch [{:03d}], Train loss: {:.5f}'.format(epoch, loss_recoder.avg), end='; ')
    return loss_recoder.avg


def valid(device, valid_loader, baseline_model, criterion):
    baseline_model.eval()
    loss_recoder = AverageMeter()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            y_pred = baseline_model(x)
            loss = criterion(y, y_pred)
            loss_recoder.update(loss.detach().item(), y.shape[0])
    # ---------------- info printing ------------------
    print('Valid loss: {:.5f}'.format(loss_recoder.avg), end=', ')
    return loss_recoder.avg


# Computing some values
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def getR2(gt, pred):
    gt_mean = np.mean(gt)
    molecule, denominator = 0, 0
    for i in range(gt.shape[0]):
        molecule += (gt[i] - pred[i]) ** 2
        denominator += (gt[i] - gt_mean) ** 2
    R2 = 1 - molecule / denominator
    return R2


def get_mean_saliency_map(device, data_type, column_ids, model):
    para = Parameter().args
    test_dataset = dataset.BandGapDataset(para, data_type, column_ids)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    saliency = Saliency(model)
    for i, (x, gt_y) in enumerate(test_data_loader):
        x, y = x.to(device), gt_y.to(device)
        attribution = torch.mean(saliency.attribute(x), dim=0, keepdim=False)
        return attribution


def get_logger(filename="importance_output.log", name="importance_logger"):
    """
    Input Args:
        filename: output file
        name: logger需要通过name唯一标识，否则输出会重叠

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    sh1 = logging.StreamHandler()
    sh1.setLevel(logging.WARNING)
    # print(logger)
    # print(type(logger))
    fh1 = logging.FileHandler(filename=filename, mode='w')
    fmt1 = logging.Formatter(fmt=
                             "%(asctime)s - %(levelname)-9s - "
                             "%(filename)-8s : %(lineno)s line - "
                             "%(message)s")
    sh1.setFormatter(fmt1)
    fh1.setFormatter(fmt1)
    logger.addHandler(sh1)
    logger.addHandler(fh1)
    return logger


def get_hessian(device, data_type, column_ids, model):
    para = Parameter().args
    test_dataset = dataset.BandGapDataset(para, data_type, column_ids)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    hessian_mat = []
    for x, gt_y in test_data_loader:
        x, y = x.to(device), gt_y.to(device)
        hessian_mat.append(torch.abs(hessian(func=model, inputs=x).reshape(1, len(column_ids), len(column_ids))))
    hessian_mat = torch.cat(hessian_mat, dim=0)
    return torch.mean(hessian_mat, dim=0, keepdim=False).cpu()


def decoupling(hessian_mat, mask, gamma=2.0):
    seperate_lst = [0]
    column_group_lst = mask
    for group in column_group_lst:
        seperate_lst.append(seperate_lst[-1] + len(group))  # seperate_lst存放每组对应的长度
    copy_hessian_mat = hessian_mat.clone()
    copy_hessian_mat += torch.diag(torch.Tensor([np.inf for _ in range(copy_hessian_mat.shape[0])]))  # 将对角线置为无穷大
    min_tensor = [torch.min(copy_hessian_mat[seperate_lst[i]:seperate_lst[i + 1],
                            seperate_lst[i]:seperate_lst[i + 1]]).item() for i in range(len(column_group_lst))]
    partition_index = np.argmin(min_tensor)  # 最小index所在的group_idx

    linshi_group = column_group_lst[partition_index]
    sub_hessian_mat = copy_hessian_mat[seperate_lst[partition_index]:seperate_lst[partition_index + 1],
                      seperate_lst[partition_index]:seperate_lst[partition_index + 1]]
    index = torch.argmin(sub_hessian_mat)  # 子hessian矩阵中的最小位置

    # 找到该组别下关联性最弱的两个特征进行打断
    row_index = torch.div(index, sub_hessian_mat.shape[0], rounding_mode='floor')
    col_index = index % sub_hessian_mat.shape[0]
    partition_result = [[linshi_group[row_index]], [linshi_group[col_index]]]
    # print("init partition_result = ", partition_result)
    # print("row_index = ", row_index)
    # print("col_index = ", col_index)

    # for i in range(sub_hessian_mat.shape[0]):  # 按当前group对应的子hessian阵的长宽进行遍历
    #     if i == row_index or i == col_index:
    #         continue
    #     else:
    #         row_coupled_coeff = sub_hessian_mat[i, row_index]
    #         col_coupled_coeff = sub_hessian_mat[i, col_index]
    #         if row_coupled_coeff > col_coupled_coeff:
    #             # 越大说明越耦合
    #             if row_coupled_coeff * min_ratio > col_coupled_coeff:
    #                 # 假如耦合度远大于另一边，则只一边添加
    #                 partition_result[0].append(linshi_group[i])
    #             else:
    #                 partition_result[0].append(linshi_group[i])
    #                 partition_result[1].append(linshi_group[i])
    #         else:
    #             if col_coupled_coeff * min_ratio > row_coupled_coeff:
    #                 partition_result[1].append(linshi_group[i])
    #             else:
    #                 partition_result[0].append(linshi_group[i])
    #                 partition_result[1].append(linshi_group[i])
    # partition_result = [list(set(group)) for group in partition_result]  # 去除可能重复的变量

    for i in range(sub_hessian_mat.shape[0]):  # 按当前group对应的子hessian阵的长宽进行遍历
        if i == row_index or i == col_index:
            continue
        else:
            row_coupled_coeff = sub_hessian_mat[i, row_index]
            col_coupled_coeff = sub_hessian_mat[i, col_index]
            superior_proportion = 1 - (len(partition_result[0]) - len(partition_result[1])) / \
                                  ((len(partition_result[0]) + len(partition_result[1])) * gamma)
            if row_coupled_coeff * superior_proportion > col_coupled_coeff:  # 更有可能产生均衡的结果
                partition_result[0].append(linshi_group[i])
            else:
                partition_result[1].append(linshi_group[i])
    partition_result = [list(set(group)) for group in partition_result]  # 去除可能重复的变量

    return partition_index, partition_result


