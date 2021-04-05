import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torch
from scipy.optimize import linear_sum_assignment

import config


def get_features(data_loader, net, mode=None, source_centers=None):
    features_list = []
    res_features_list = []
    labels_list = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.cuda()
            net.eval()
            features, _, res_features = net(inputs, source_centers=source_centers)
            features_list.append(features)
            res_features_list.append(torch.tensor(res_features))
            for label in labels:
                labels_list.append(label.numpy())
            print("extract the {} feature".format(i * config.target_batch_size))
            torch.cuda.empty_cache()

    features = torch.cat(features_list, 0).cpu().numpy()
    res_features = torch.cat(res_features_list, 0).numpy()
    labels = np.array(labels_list)
    if mode is None:
        pass
    elif mode == 'train':
        np.save('train_features.npy', features, allow_pickle=True)
        np.save('train_labels.npy', labels, allow_pickle=True)
    elif mode == 'test':
        np.save('test_features.npy', features, allow_pickle=True)
        np.save('test_labels.npy', labels, allow_pickle=True)
    return features, labels, res_features


def save_txt_files(path, the_list):
    if os.path.exists(path) is not True:
        f = open(path,'w')
        f.close()
    f = open(path, 'a')
    for i in the_list:
        f.write(str(i) + '\n')
    f.close()


def save_txt_files2(path, the_list):
    f = open(path, 'w')
    for i in the_list:
        f.write(str(i) + '\n')
    f.close()


def cluster_acc(y_true, y_pred, class_number):  # 聚类精度  真正标签与预测标签
    cnt_mtx = np.zeros([class_number, class_number])

    # fill in matrix
    for i in range(len(y_true)):
        cnt_mtx[int(y_pred[i]), int(y_true[i])] += 1

    # find optimal permutation
    row_ind, col_ind = linear_sum_assignment(-cnt_mtx)

    # compute error
    acc = cnt_mtx[row_ind, col_ind].sum() / cnt_mtx.sum()

    labels_pred = []
    for index, label in enumerate(y_pred):
        target_label = col_ind[label]
        # print('label', label)
        # print('target', target_label)
        # print('true', y_true[index])
        labels_pred.append(target_label)
    # print(labels_pred[:10])
    # print(list(y_true)[:10])

    return acc, list(y_true), labels_pred









