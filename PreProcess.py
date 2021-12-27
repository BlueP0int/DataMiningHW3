import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def knn(target_idx, data, k):
    ret = []
    target = data[target_idx]
    for idx, elem in enumerate(data):
        if idx != target_idx:
            dist = np.linalg.norm(elem - target)
            ret.append((dist, idx))
            if len(ret) > k:
                ret = sorted(ret)
                del ret[0]
    ret.reverse()
    return [r[1] for r in ret]


def complete_by_knn(X, col, reduced, k):
    for idx, record in enumerate(X):
        if np.isnan(record[col]):
            neighbors = knn(idx, reduced, k)
            record[col] = np.mean([X[row][col] for row in neighbors])


def data_preprocess(X):
    X[:, 12][np.isnan(X[:, 12])] = np.nanmean(X[:, 12])
    subX = X[:, [0, 1, 4, 5, 10, 12, 13, 15]]
    stdSubX = StandardScaler().fit_transform(subX)
    pca = PCA(n_components=6)
    reduced = pca.fit_transform(stdSubX)

    complete_by_knn(X, 14, reduced, 5)
    X = StandardScaler().fit_transform(X)
    return X
