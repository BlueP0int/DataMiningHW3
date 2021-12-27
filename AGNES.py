import numpy as np


def dist(a, b):
    return np.linalg.norm(a - b)


def build_dist_matrix(data):
    ret = []
    for i in range(len(data)):
        row = []
        for j in range(i):
            row.append(dist(data[i], data[j]))
        ret.append(row)
    return ret


def merge_by_min(clusters_before_merge, dist_mat, i, j):
    if i > j:
        big = i
        small = j
    else:
        big = j
        small = i
    for col in range(small):
        dist_mat[small][col] = min(dist_mat[small][col], dist_mat[big][col])
    for row in range(small + 1, big):
        dist_mat[row][small] = min(dist_mat[row][small], dist_mat[big][row])
    for row in range(big + 1, len(dist_mat)):
        dist_mat[row][small] = min(dist_mat[row][small], dist_mat[row][big])
        dist_mat[row].pop(big)
    dist_mat.pop(big)


def merge_by_max(clusters_before_merge, dist_mat, i, j):
    if i > j:
        big = i
        small = j
    else:
        big = j
        small = i
    for col in range(small):
        dist_mat[small][col] = max(dist_mat[small][col], dist_mat[big][col])
    for row in range(small + 1, big):
        dist_mat[row][small] = max(dist_mat[row][small], dist_mat[big][row])
    for row in range(big + 1, len(dist_mat)):
        dist_mat[row][small] = max(dist_mat[row][small], dist_mat[row][big])
        dist_mat[row].pop(big)
    dist_mat.pop(big)


def merge_by_avg(clusters_before_merge, dist_mat, i, j):
    if i > j:
        big = i
        small = j
    else:
        big = j
        small = i
    new_size = len(clusters_before_merge[big]) + len(clusters_before_merge[small])
    s_size = len(clusters_before_merge[small])
    b_size = len(clusters_before_merge[big])
    for col in range(small):
        dist_mat[small][col] = (dist_mat[small][col] * s_size + dist_mat[big][col] * b_size) / new_size
    for row in range(small + 1, big):
        dist_mat[row][small] = (dist_mat[row][small] * s_size + dist_mat[big][row] * b_size) / new_size
    for row in range(big + 1, len(dist_mat)):
        dist_mat[row][small] = (dist_mat[row][small] * s_size + dist_mat[row][big] * b_size) / new_size
        dist_mat[row].pop(big)
    dist_mat.pop(big)


def find_nearest(dist_mat):
    cur_min = None
    ret = None
    for i in range(len(dist_mat)):
        for j in range(len(dist_mat[i])):
            if cur_min is None or dist_mat[i][j] < cur_min:
                cur_min = dist_mat[i][j]
                ret = (i, j)
    return ret


class AGNES:
    def __init__(self, k, mode="avg"):
        self.__k = k
        self.__mode = mode

    def fit(self, data):
        if self.__mode == "min":
            merge_func = merge_by_min
        elif self.__mode == "max":
            merge_func = merge_by_max
        elif self.__mode == "avg":
            merge_func = merge_by_avg
        else:
            return

        clusters = [[i] for i in range(len(data))]
        c = len(data)
        dist_mat = build_dist_matrix(data)

        while c > self.__k:
            i, j = find_nearest(dist_mat)
            merge_func(clusters, dist_mat, i, j)
            clusters[j].extend(clusters[i])
            clusters.pop(i)
            c -= 1
        ret = []
        for idx in range(len(data)):
            for i, cluster in enumerate(clusters):
                if idx in cluster:
                    ret.append(i + 1)
        return ret
