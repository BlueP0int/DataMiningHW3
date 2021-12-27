import numpy as np
from sklearn.neighbors import NearestNeighbors


class MyDBSCAN:
    def __init__(self, radius=0.1, min_samples=4):
        self.__radius = radius
        self.__min_samples = min_samples

    def fit(self, data):
        nn = NearestNeighbors(radius=self.__radius)
        nn.fit(data)
        neighbors = nn.radius_neighbors(data, return_distance=False)

        labels = [-1 for i in range(len(data))]
        cores = [len(n) >= self.__min_samples for n in neighbors]

        cur_label = 1
        stack = []

        for i in range(len(labels)):
            cur = i
            if labels[cur] != -1 or (not cores[cur]):
                continue

            while True:
                if labels[cur] == -1:
                    labels[cur] = cur_label
                    if cores[cur]:
                        for neighbor in neighbors[cur]:
                            if labels[neighbor] == -1:
                                stack.append(neighbor)

                if len(stack) == 0:
                    break

                cur = stack.pop()

            cur_label += 1

        return labels
