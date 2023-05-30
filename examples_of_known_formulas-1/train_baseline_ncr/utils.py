import torch
import random


def shuffle_data(data):
    num, _ = data.shape
    lst = list(range(num))
    random.shuffle(lst)
    return data[lst]


def train(device, train_loader, baseline_model, opt, criterion, epoch, alpha=0.2):
    baseline_model.train()
    loss_recoder = AverageMeter()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        baseline_model.zero_grad()
        sim_matrix, arg_order, y_pred = baseline_model(x)  # (64, 1)
        pre_loss = criterion(y, y_pred)
        con_loss = ncr(sim_matrix, arg_order, y_pred, 10)
        loss = (1 - alpha) * pre_loss + alpha * con_loss
        # y_pred = baseline_model(x)
        # pre_loss = criterion(y, y_pred)
        # loss = pre_loss
        loss_recoder.update(pre_loss.detach().item(), y.shape[0])
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
            y_pred = baseline_model(x, False)
            loss = criterion(y, y_pred)
            loss_recoder.update(loss.detach().item(), y.shape[0])
    # ---------------- info printing ------------------
    print('Valid loss: {:.5f}'.format(loss_recoder.avg), end=', ')
    return loss_recoder.avg


def ncr(sim_matrix, arg_order, y_pred, k):
    '''
    计算约束正则项
    :param sim_matrix: bsxbs
    :param y_pred: bsx1
    :return: bsx1
    '''
    # 先根据邻域数据计算y_consistency
    bs, d = y_pred.shape
    y_cons = torch.ones_like(y_pred)
    for i in range(bs):
        order = arg_order[i][:k]
        sim_sum = 0.0
        for j in range(k):
            sim_sum += sim_matrix[i][order[j]]
        for j in range(k):
            y_cons[i] = sim_matrix[i][order[j]] / sim_sum * y_pred[order[j]]
    return torch.mean((torch.abs(y_cons - y_pred)))


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