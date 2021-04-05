import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import numpy as np
import os
import time
import utils_fun
from ResNet import Net
import config
import loss_fun
from sklearn.cluster import KMeans
from torchsampler.imbalanced import ImbalancedDatasetSampler
#from mmd_loss import mmd_loss
from lmmd import lmmd
from sklearn import preprocessing
import math
import torch.nn.functional as F


def generate_batch(source_data, target_data):
    source_iter = iter(source_data)
    target_iter = iter(target_data)
    source_x1 = []
    source_labels1 = []
    source_x2 = []
    source_labels2 = []
    target_x = []
    target_labels = []
    inputs, labels = next(source_iter)
    source_x1.append(inputs)
    for label in labels:
        source_labels1.append(label.item())
    inputs, labels = next(source_iter)
    source_x2.append(inputs)
    for label in labels:
        source_labels2.append(label.item())
    inputs, labels = next(target_iter)
    target_x.append(inputs)
    for label in labels:
        target_labels.append(label.item())
    source_x1 = torch.cat(source_x1, 0)
    source_x2 = torch.cat(source_x2, 0)
    target_x = torch.cat(target_x, 0)
    source_labels1 = torch.tensor(source_labels1)
    source_labels2 = torch.tensor(source_labels2)
    target_labels = torch.tensor(target_labels)
    source_labels = torch.tensor((source_labels1.numpy() == source_labels2.numpy()).astype('float32'))
    return source_x1, source_x2, target_x, source_labels1, source_labels2, target_labels, source_labels


def train():
    source_dataset = dataset.ImageFolder(config.source_path, transform=config.train_transform)
    source_data_loader = DataLoader(source_dataset, config.source_batch_size, sampler=ImbalancedDatasetSampler(source_dataset))
    target_dataset = dataset.ImageFolder(config.target_path, transform=config.test_transform)
    target_data_loader = DataLoader(target_dataset, config.target_batch_size, shuffle=True)

    test_dataset = dataset.ImageFolder(config.target_path, transform=config.test_transform)
    test_data_loader = DataLoader(test_dataset, config.target_batch_size, shuffle=False)

    # define GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define model
    if os.path.exists(config.model_path) is not True:
        net = Net().to(device)
    else:
        # net = torch.load(config.model_path)
        net = Net().to(device)

    con_loss = loss_fun.ContrastiveLoss()
    domain_loss = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    best_acc = 0

    for epoch in range(config.epoches):
        since = time.time()
        sum_loss1 = 0.
        sum_loss2 = 0.
        sum_loss3 = 0.
        net.train()
        length = config.source_batch_size + config.target_batch_size
        dis_list = []
        source_centers_sum = np.zeros([config.class_num, config.features_dim])
        source_res_centers_sum = np.zeros([config.class_num, config.features_dim])

        for i in range(300):
            source_x1, source_x2, target_x, source_labels1, source_labels2, target_labels, source_labels = \
                generate_batch(source_data_loader, target_data_loader)
            source_x1, source_x2, target_x, source_labels = \
                source_x1.to(device), source_x2.to(device), target_x.to(device), source_labels.to(device)
            optimizer.zero_grad()

            labels = np.concatenate((source_labels1.numpy(), source_labels2.numpy()))
            outputs, A, res_features = net(torch.cat([source_x1, source_x2, target_x], 0), len(source_x1) + len(source_x2), source_labels=labels)
            source_outputs1, source_outputs2, target_outputs = outputs[:len(source_labels1)], outputs[len(source_labels1):len(source_labels1) + len(source_labels2)], outputs[len(source_labels1) + len(source_labels2):]
            source_res_features = res_features[:len(source_labels1) + len(source_labels2)]
            source_features = torch.cat([source_outputs1, source_outputs2], 0).data.cpu().numpy()
            source_labels_numpy = np.concatenate([source_labels1.numpy(), source_labels2.numpy()], 0)
            target_features = target_outputs.data.cpu().numpy()
            target_labels_numpy = target_labels.numpy()
            k_means_source = KMeans(config.class_num)
            k_means_source.fit(source_features)
            acc_source, _, _ = utils_fun.cluster_acc(source_labels_numpy, k_means_source.labels_, config.class_num)
            k_means_target = KMeans(config.class_num)
            k_means_target.fit(target_features)
            target_label_centers = torch.tensor(k_means_target.cluster_centers_, requires_grad=True).to(device)
            acc_target, target_labels_true, target_labels_pred = utils_fun.cluster_acc(target_labels_numpy, k_means_target.labels_, config.class_num)
            one_hot = preprocessing.OneHotEncoder(sparse=False, categories='auto')
            target_one_hot = one_hot.fit_transform(np.array(target_labels_pred).reshape(-1, 1))
            target_labels_pred_torch = torch.tensor(target_one_hot).to(device)
            target_soft_labels = F.softmax(torch.exp(torch.sqrt(torch.sum(torch.pow((target_outputs.unsqueeze(1) - target_label_centers), 2), 2))*-1), 1)

            loss1 = con_loss(source_outputs1, source_outputs2, source_labels)
            # loss2 = mmd_loss(source_outputs1[:len(target_outputs)], target_outputs)
            source_outputs = outputs[:len(source_labels1) + len(source_labels2)]
            source_centers = torch.mean(source_outputs, dim=0)
            source_res_centers = np.mean(source_res_features, axis=0)
            source_res_centers_sum += source_res_centers
            target_centers = torch.mean(target_outputs, dim=0)
            loss2 = domain_loss(source_centers, target_centers)
            loss3 = lmmd(source_outputs[:len(target_outputs)], target_outputs, source_labels1[:len(target_labels)].long(), target_soft_labels)
            lambda1 = 2/(1 + math.exp(-10/config.epoches)) - 1
            # loss4 = half_feature_matching_loss(A, outputs)
            loss = loss1 + loss2

            sum_loss1 += loss1
            sum_loss2 += loss2
            sum_loss3 += loss3

            loss.backward()
            optimizer.step()

            iter_num = i + 1 + epoch * length
            print('[epoch:%d, iter:%d] Loss: %f | Acc_source: %f | Acc_target: %f | Loss_con: %f | Loss_domain: %f | Loss_lmmd: %f | time: %f'
                  % (epoch + 1, iter_num, loss, acc_source, acc_target, sum_loss1/iter_num, sum_loss2/iter_num, sum_loss3/iter_num, time.time() -  since))

        source_res_centers = source_res_centers_sum / 300
        test_features, test_labels, test_res_features = utils_fun.get_features(test_data_loader, net, source_centers=source_res_centers)
        k_means_test = KMeans(config.class_num)
        k_means_test.fit(test_features)
        acc_test, labels_true, labels_pred = utils_fun.cluster_acc(test_labels, k_means_test.labels_, config.class_num)
        print('Test_acc:', acc_test, 'Time:', time.time() - since)
        f = open('test_acc1.txt', 'a')
        f.write(str(acc_test) + '\n')
        f.close()
        if acc_test > best_acc:
            best_acc = acc_test
            np.save('features.npy', test_features)
            np.save('labels_true.npy', labels_true)
            np.save('labels_pred.npy', labels_pred)
            torch.save(net, 'net.pkl')
        scheduler.step(epoch)


if __name__ == '__main__':
    train()
