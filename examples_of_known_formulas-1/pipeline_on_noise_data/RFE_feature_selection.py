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

    # Setting the random seed
    torch.manual_seed(para.seed)
    torch.cuda.manual_seed(para.seed)
    random.seed(para.seed)
    np.random.seed(para.seed)

    # Dataset
    valid_dataset = dataset.BandGapDataset(para, 'val')
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=2000,
        shuffle=False,
        pin_memory=True
    )
    print('length_of_valid: ', len(valid_dataset))

    # Networks
    baseline_model = models.MultiBranchModel([para.init_feature_num], para.layer_num).to(device)
    checkpoint0 = torch.load('checkpoints/baseline/latest.pth')
    baseline_model.load_state_dict(checkpoint0['model'])
    baseline_model.eval()

    # 获取基础模型在全部数据和测试数据上面的准确度
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            y_pred = baseline_model(x)
    # evaluate
    pred, gt = y_pred.squeeze().cpu().numpy(), y.squeeze().cpu().numpy()

    print('baseline_R2: {:.4f}'.format(utils.getR2(gt, pred)))

    # 计算特征重要性
    column_ids = [i for i in range(para.init_feature_num)]
    total_saliency_map = utils.get_mean_saliency_map(device, 'train', column_ids, baseline_model)

    # 开始特征筛选
    total_feature_num = para.init_feature_num
    filter_num = 1
    logger = utils.get_logger(filename='Feature importance.log', name='Feature importance')
    while True:
        idx_tensor = torch.topk(total_saliency_map, filter_num, largest=True)[1]
        masks = [int(idx) for idx in idx_tensor]  # 需要保留的特征序号

        # 仅使用部分特征的数据集
        train_dataset = dataset.BandGapDataset(para, 'train', masks)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=para.batch_size,
            shuffle=True,
            pin_memory=True
        )

        # Networks
        model = models.MultiBranchModel([len(masks)], para.layer_num).to(device)

        # Setting the optimizers
        lr = 1e-2
        opt = optim.Adam(model.parameters(), lr)
        criterion = nn.L1Loss().to(device)

        # Record relevant indicators
        min_loss = 1000
        train_loss, valid_loss = [], []
        date = datetime.datetime.now()
        model_path = para.save_dir + 'feature-num={:d}/'.format(len(masks))
        os.makedirs(model_path, exist_ok=True)
        np.savetxt(model_path + '/mask.txt', masks)

        # train
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        for epoch in range(1, para.end_epoch + 1):

            if epoch % 40 == 0:
                lr /= 5
                for param_group in opt.param_groups:
                    param_group['lr'] = lr

            tra_loss = utils.train(device, train_loader, model, opt, criterion, epoch)
            print()
            checkpoint = {'model': model.state_dict(), 'epoch': epoch}
            torch.save(checkpoint, model_path + '/latest.pth')

        # 对当前模型进行评价
        valid_dataset = dataset.BandGapDataset(para, 'valid', masks)
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=2000,
            shuffle=False,
            pin_memory=True
        )

        # 获取基础模型在全部数据和测试数据上面的准确度
        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)

        pred, gt = y_pred.squeeze().numpy(), y.squeeze().numpy()

        logger.info('Feature_mask: {:}, R2: {:.4f}'.format(masks, utils.getR2(gt, pred)))
        if filter_num >= para.init_feature_num:
            print('Unable to continue feature filtering.')
            break
        else:
            filter_num += 1
