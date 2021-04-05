import numpy as np
import distance_fun
from sklearn.neighbors import kneighbors_graph
import torch


def process_graph_torch(adj):
    A = adj
    I = torch.eye(len(A))
    A = A + I
    D_hat = torch.sum(A, dim=0)
    D_hat = D_hat ** 0.5
    D_hat = torch.diag(D_hat ** -1)
    return A, D_hat


def process_graph(adj):
    A = np.asmatrix(adj)
    I = np.eye(len(A))
    A_hat = A + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = D_hat**0.5
    D_hat = np.matrix(np.diag(D_hat))
    D_hat = D_hat**-1
    return A_hat, D_hat


def distance_graph(features, distance='Euclidean', thre=0.):
    adj = np.zeros((len(features), len(features)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            if distance == 'Euclidean':
                dis = np.exp(distance_fun.euclidean_distance(features[i], features[j])*-1)
                if dis >= thre:
                    adj[i, j] = dis
            elif distance == 'cos':
                dis = distance_fun.cos_distance(features[i], features[j])
                if dis >= thre:
                    adj[i, j] = dis
            elif distance == 'pearson_correlation':
                dis = distance_fun.pearson_correlation(features[i], features[j])
                if dis >= thre:
                    adj[i, j] = dis
    return adj


def supervised_graph(labels):
    adj_labels = np.zeros((len(labels), len(labels)))
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] == labels[j]:
                adj_labels[i, j] = 1
    return adj_labels


def supervised_distance_graph(features, labels, distance='Euclidean', thre=0.):
    adj = distance_graph(features, distance=distance, thre=thre)
    adj_labels = supervised_graph(features[:len(labels)], labels)
    adj[:len(adj_labels), :len(adj_labels)] = adj_labels
    return adj


# 限制ϵ半径建图算法中的ϵ
def from_labels_get_radius(features, labels, class_number):
    distance_list = np.zeros(class_number)
    for index1, feature1 in enumerate(features):
        for index2, feature2 in enumerate(features):
            if labels[index1] == labels[index2]:
                distance_list[labels[index1]] += distance_fun.euclidean_distance(features[index1], features[index2])
    count = (len(features) * (len(features) - 1)) / 2
    radius_limit = np.mean(distance_list / count)
    return radius_limit


# ϵ半径建图
def radius_graph(features, dis=None, labels=None, class_number=None, include_self=True, limit_radius=False):
    if limit_radius:
        dis = from_labels_get_radius(features, labels, class_number)
    adjacency_matrix = np.zeros((len(features), len(features)))
    for index1, feature1 in enumerate(features):
        for index2, feature2 in enumerate(features):
            if dis >= distance_fun.euclidean_distance(feature1, feature2) and index1 != index2:
                adjacency_matrix[index1, index2] = 1
            elif dis >= distance_fun.euclidean_distance(feature1, feature2) and index1 == index2 and include_self:
                adjacency_matrix[index1, index2] = 1
    return adjacency_matrix


# knn与ϵ半径组合建图
def knn_and_radius_graph(features, k, dis=None, labels=None, class_number=None, include_self=True):
    adjacency_matrix_knn = kneighbors_graph(features, k, include_self=include_self).toarray()
    if labels is not None:
        features_label = features[:len(labels)]
        dis = from_labels_get_radius(features_label, labels, class_number)
    adjacency_matrix_radius = radius_graph(features, dis, include_self)
    adjacency_matrix = np.zeros((len(features), len(features)))
    for index in range(len(features)):
        radius_node_number = np.sum(adjacency_matrix_radius[index])
        if radius_node_number > k:
            adjacency_matrix[index] = adjacency_matrix_radius[index]
        else:
            adjacency_matrix[index] = adjacency_matrix_knn[index]
    if labels is not None:
        return adjacency_matrix, dis
    else:
        return adjacency_matrix


def domain_cluster_graph(source_labels, target_labels, init_graph=None):
    length = len(source_labels) + len(target_labels)
    if init_graph is None:
        adj = np.zeros([length, length])
    else:
        adj = init_graph
    for i, target in enumerate(target_labels):
        for j, source in enumerate(source_labels):
            if target == source:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj


if __name__ == '__main__':
    print(np.zeros((15, 15)))












