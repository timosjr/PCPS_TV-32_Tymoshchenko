import numpy as np
import random

class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.distances = np.zeros((num_nodes, num_nodes), dtype=float)
        np.fill_diagonal(self.distances, 0)
        self.node_positions = []

    def add_edge(self, node1, node2, distance):
        if 0 <= node1 < self.num_nodes and 0 <= node2 < self.num_nodes:
            self.distances[node1, node2] = float(distance)
            self.distances[node2, node1] = float(distance)
        else:
            print(f"Помилка: Вузли {node1} або {node2} виходять за межі діапазону.")

    def set_node_positions(self, positions):
        if len(positions) == self.num_nodes:
            self.node_positions = positions
        else:
            raise ValueError(f"Кількість позицій ({len(positions)}) не відповідає кількості вузлів ({self.num_nodes}).")

    def get_distance(self, node1, node2):
        return self.distances[node1, node2]

    def get_neighbors(self, node):
        return [i for i in range(self.num_nodes) if i != node]

    @classmethod
    def generate_random_graph(cls, num_nodes, max_distance=100, x_range=(0, 10), y_range=(0, 10)):
        graph = cls(num_nodes)
        positions = []
        for _ in range(num_nodes):
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            positions.append((x, y))
        graph.set_node_positions(positions)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                graph.add_edge(i, j, dist)

        return graph