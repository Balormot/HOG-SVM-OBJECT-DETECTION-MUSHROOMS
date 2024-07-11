import numpy as np
class DBSCAN:
    def __init__(self, epsilon, min_pts):
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.clusters = []
        self.noise = []
        self.center = []

    @staticmethod
    def _euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def _get_neighbors(self, dataset, point):
        neighbors = []
        for index, candidate in enumerate(dataset):
            if self._euclidean_distance(point, candidate) < self.epsilon:
                neighbors.append(index)
        return neighbors

    def fit(self, dataset):
        visited = [False] * len(dataset)
        for index in range(len(dataset)):
            if not visited[index]:
                visited[index] = True
                neighbors = self._get_neighbors(dataset, dataset[index])
                if len(neighbors) < self.min_pts:
                    self.noise.append(index)
                else:
                    self._expand_cluster(dataset, visited, index, neighbors)
                    # self.center.append(index)
        return self.clusters, self.noise

    def _expand_cluster(self, dataset, visited, index, neighbors):
        self.clusters.append([index])
        i = 0
        while i < len(neighbors):
            next_index = neighbors[i]
            if not visited[next_index]:
                visited[next_index] = True
                next_neighbors = self._get_neighbors(dataset, dataset[next_index])
                if len(next_neighbors) >= self.min_pts:
                    neighbors += next_neighbors
            cluster_indices = [i for cluster in self.clusters for i in cluster]
            if next_index not in cluster_indices:
                self.clusters[-1].append(next_index)
            i += 1