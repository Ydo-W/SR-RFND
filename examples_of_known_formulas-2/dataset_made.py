import os
import math
import numpy as np


def target_eq1(m, u, v, w):
    c = 2.998e8
    return 0.5 * m * (u**2 + v**2 + w**2)


def default_sampling(sample_size, min_value=1.0e-1, max_value=1.0e1):
    # x ~ either U(0.1, 10.0) or U(-10.0, -0.1) with 50% chance
    num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
    num_negatives = sample_size - num_positives
    log10_min = np.log10(min_value)
    log10_max = np.log10(max_value)
    pos_samples = 10.0 ** np.random.uniform(log10_min, log10_max, size=num_positives)
    neg_samples = -10.0 ** np.random.uniform(log10_min, log10_max, size=num_negatives)
    all_samples = np.concatenate([pos_samples, neg_samples])
    np.random.shuffle(all_samples)
    return all_samples


def positive_sampling(sample_size, min_value=1.0e-1, max_value=1.0e1):
    log10_min = np.log10(min_value)
    log10_max = np.log10(max_value)
    pos_samples = 10.0 ** np.random.uniform(log10_min, log10_max, size=sample_size)
    return pos_samples


def add_gaussian_noise(data, sigma=0.3):
    y = data[:, -1]

    # 归一化
    mean_data, sigma_data = np.mean(y), np.std(y)
    y = (y - mean_data) / sigma_data

    # 加入高斯噪声
    noise = np.random.normal(0., sigma, len(y))
    y = y + noise

    # 反归一化
    y = y * sigma_data + mean_data

    data[:, -1] = y
    return data


def add_burst_noise(data, burst_ratio=0.3):
    y = data[:, -1]

    # 归一化
    mean_data, sigma_data = np.mean(y), np.std(y)
    y = (y - mean_data) / sigma_data

    # 随机选取若干个数据点，将其扰动
    num_spikes = int(len(y) * burst_ratio)
    spike_idx = np.random.choice(np.arange(len(y)), num_spikes, replace=False)
    spike_height = np.random.uniform(-2, 2, num_spikes)
    y[spike_idx] += spike_height

    # 反归一化
    y = y * sigma_data + mean_data

    data[:, -1] = y
    return data


if __name__ == '__main__':
    # 默认输入变量
    x_all = np.zeros((10000, 4))
    x_all[:, 0] = default_sampling(10000, 1.0e-2, 1.0)
    x_all[:, 1] = positive_sampling(10000, 1.0e-1, 1.0e1)
    x_all[:, 2] = positive_sampling(10000, 1.0e-1, 1.0e1)
    x_all[:, 3] = positive_sampling(10000, 1.0e-1, 1.0e1)

    # 添加四个额外特征变量
    extra = np.zeros((10000, 3))
    extra[:, 0] = default_sampling(10000, 1.0e-2, 1.0e1)
    extra[:, 1] = default_sampling(10000, 1.0e-2, 1.0e1)
    extra[:, 2] = default_sampling(10000, 1.0e-2, 1.0e1)

    # 计算目标值
    y_all = np.zeros((10000, 1))
    for i in range(10000):
        y_all[i, 0] = target_eq1(x_all[i, 0], x_all[i, 1], x_all[i, 2], x_all[i, 3])

    all_data = np.concatenate([extra, x_all, y_all], axis=1)
    all_data = all_data[:, [0, 6, 3, 2, 4, 1, 5, 7]]  # 正确变量：1, 2, 4, 6
    train_and_val = all_data[:9000]
    test = all_data[9000:]

    # 增加噪声
    train_and_val = add_gaussian_noise(train_and_val)
    train_and_val = add_burst_noise(train_and_val)

    # 保存数据
    out_dir = 'datasets/new-feynman-i.13.4/'
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(out_dir + 'train_val.txt', train_and_val, fmt='%.8e')
    np.savetxt(out_dir + 'test.txt', test, fmt='%.8e')

