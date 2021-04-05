import numpy as np


def euclidean_distance(features_1, features_2):
    dis = np.sqrt(np.sum(np.square(features_1 - features_2)))
    return dis


def cos_distance(features_1, features_2):
    dis = np.dot(features_1, features_2) / (np.linalg.norm(features_1) * (np.linalg.norm(features_2)))
    return dis


def pearson_correlation(features_1, features_2):
    features_1_ = features_1 - np.mean(features_1)
    features_2_ = features_2 - np.mean(features_2)
    dis = cos_distance(features_1_, features_2_)
    return dis


if __name__ == '__main__':
    a = np.array([1, 2, 3])
    b = np.array([5, 6, 7])
    print(np.dot(a, b))
