"""
GridWorld: a small 2D path-planning simulator.

Implements two classic search algorithms on a grid with obstacles:
- Breadth-First Search (BFS): guaranteed shortest path, explores ring by ring.
- A* Search: guaranteed shortest path, explores using a cost + heuristic estimate,
  typically visiting fewer cells than BFS on larger grids.

Built as a Python-fundamentals capstone project, alongside training ACT
(Action Chunking with Transformers) policies for robotic manipulation.
"""

import heapq
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


class GridWorld:
    def __init__(self, grid, start, goal):
        """
        grid: 2D NumPy array. 0 = free cell, 1 = obstacle.
        start: (row, col) tuple, the starting cell.
        goal: (row, col) tuple, the target cell.
        """
        self.grid = grid
        self.start = start
        self.goal = goal

    def get_neighbors(self, cell):
        """Return valid (in-bounds, non-obstacle) neighboring cells: up, down, left, right."""
        row, col = cell
        valid_neighs = []
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        rows, cols = self.grid.shape

        for dr, dc in directions:
            v_row = row + dr
            v_col = col + dc
            v_cell = (v_row, v_col)
            if 0 <= v_row < rows and 0 <= v_col < cols:
                if self.grid[v_cell] != 1:
                    valid_neighs.append((v_row, v_col))
        return valid_neighs

    def bfs(self):
        """Breadth-First Search. Returns a came_from dict mapping cell -> parent cell."""
        queue = deque([self.start])
        visited = {self.start}
        came_from = {}

        while queue:
            removed_cell = queue.popleft()
            valid_neighs = self.get_neighbors(removed_cell)
            for neighbor in valid_neighs:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    came_from[neighbor] = removed_cell
                    if neighbor == self.goal:
                        return came_from

    def heuristic(self, cell):
        """Manhattan distance from cell to self.goal (admissible for 4-directional movement)."""
        row1, col1 = cell
        row2, col2 = self.goal
        return abs(row1 - row2) + abs(col1 - col2)

    def a_star(self):
        """A* Search. Returns a came_from dict mapping cell -> parent cell."""
        open_set = []
        g_score = {self.start: 0}
        f_score = g_score[self.start] + self.heuristic(self.start)
        heapq.heappush(open_set, (f_score, self.start))
        came_from = {}

        while open_set:
            priority, cell = heapq.heappop(open_set)
            if cell == self.goal:
                return came_from

            neighbors = self.get_neighbors(cell)
            for neighbor in neighbors:
                tentative_g = g_score[cell] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = g_score[neighbor] + self.heuristic(neighbor)
                    came_from[neighbor] = cell
                    heapq.heappush(open_set, (f_score, neighbor))

    def reconstruct_path(self, came_from):
        """Walk backward from self.goal to self.start using came_from, then reverse."""
        current = self.goal
        path = []

        while current != self.start:
            path.append(current)
            current = came_from[current]

        path.append(self.start)
        path.reverse()
        return path

    def plot(self, algorithm="a_star"):
        """Plot the grid with start/goal markers and the found path drawn as a line.

        algorithm: "a_star" or "bfs" - which search method to use for the path.
        """
        if algorithm == "bfs":
            came_from = self.bfs()
        else:
            came_from = self.a_star()

        path = np.array(self.reconstruct_path(came_from))
        y_rows = path[:, 0]
        x_cols = path[:, 1]

        plt.imshow(self.grid)
        plt.plot(x_cols, y_rows)
        plt.plot(x_cols[0], y_rows[0], "go", ms=10)   # start, green
        plt.plot(x_cols[-1], y_rows[-1], "ro", ms=10)  # goal, red
        plt.show()


if __name__ == "__main__":
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[3, 3] = 1
    grid[2, 2] = 1
    grid[4, 1] = 1
    start = (0, 0)
    goal = (4, 2)

    world = GridWorld(grid, start, goal)
    world.plot(algorithm="a_star")
