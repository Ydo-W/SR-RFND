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
    train_dataset = dataset.BandGapDataset(para, 'train')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=para.batch_size,
        shuffle=True,
        pin_memory=True
    )
    print('length_of_train: ', len(train_dataset))

    valid_dataset = dataset.BandGapDataset(para, 'val')
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=para.batch_size,
        shuffle=False,
        pin_memory=True
    )
    print('length_of_valid: ', len(valid_dataset))

    # Networks
    baseline_model = models.MultiBranchModel([7], para.layer_num).to(device)
    parameters = sum([np.prod(list(p.size())) for p in baseline_model.parameters()])
    print('Baseline_model_params: {:4f}M'.format(parameters / 1000 / 1000))

    # Setting the optimizers
    lr = 1e-2
    opt = optim.Adam(baseline_model.parameters(), lr)
    criterion = nn.L1Loss().to(device)

    # Record relevant indicators
    min_loss = 100
    train_loss, valid_loss = [], []
    date = datetime.datetime.now()
    model_path = para.save_dir + 'baseline'
    os.makedirs(model_path, exist_ok=True)

    # train
    for epoch in range(1, para.end_epoch + 1):

        if epoch % 40 == 0:
            lr /= 5
            for param_group in opt.param_groups:
                param_group['lr'] = lr

        start = time.time()
        tra_loss = utils.train(device, train_loader, baseline_model, opt, criterion, epoch)
        val_loss = utils.valid(device, valid_loader, baseline_model, criterion)
        train_loss.append(tra_loss), valid_loss.append(val_loss)
        end = time.time()
        print('time:{:.2f}s'.format(end - start))
        checkpoint = {'model': baseline_model.state_dict(), 'epoch': epoch}
        torch.save(checkpoint, model_path + '/latest.pth')
        if valid_loss[-1] <= min_loss:
            torch.save(checkpoint, model_path + '/best.pth')
            min_loss = valid_loss[-1]

        # Plotting
        plt.switch_backend('agg')
        np.savetxt(model_path + '/train_loss.txt', train_loss)
        np.savetxt(model_path + '/valid_loss.txt', valid_loss)
        plt.figure(), plt.plot(train_loss), plt.plot(valid_loss, alpha=0.5)
        plt.savefig(model_path + '/loss.jpg'), plt.close()
