from __future__ import annotations

import heapq
import math

import cv2
import numpy as np


class Navigation:
    def __init__(
        self, blocked: np.ndarray, projectiles: np.ndarray, cell: int = 30, clearance: int = 20
    ):
        self.cell = cell
        self.height, self.width = blocked.shape
        if projectiles.shape != blocked.shape:
            raise ValueError("Movement and projectile maps must have the same dimensions")
        self.projectiles = projectiles
        expanded = cv2.dilate(
            blocked.astype("uint8"), np.ones((clearance * 2 + 1,) * 2, dtype="uint8")
        )
        gh, gw = math.ceil(self.height / cell), math.ceil(self.width / cell)
        padded = np.pad(
            expanded, ((0, gh * cell - self.height), (0, gw * cell - self.width)), constant_values=1
        )
        self.grid = padded.reshape(gh, cell, gw, cell).max(axis=(1, 3)).astype(bool)

    def _cell(self, point):
        return (
            min(self.grid.shape[1] - 1, max(0, int(point[0] // self.cell))),
            min(self.grid.shape[0] - 1, max(0, int(point[1] // self.cell))),
        )

    def next_step(self, start, goal):
        origin, target = self._cell(start), self._cell(goal)
        if origin == target:
            return None
        frontier = [(0.0, origin)]
        costs, previous = {origin: 0.0}, {}
        closest = origin

        def distance(p):
            return math.hypot(p[0] - target[0], p[1] - target[1])

        visited = set()
        while frontier:
            _, current = heapq.heappop(frontier)
            if current in visited:
                continue
            visited.add(current)
            if distance(current) < distance(closest):
                closest = current
            if current == target:
                closest = current
                break
            x, y = current
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0), (-1, -1), (1, -1), (1, 1), (-1, 1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.grid.shape[1] and 0 <= ny < self.grid.shape[0]):
                    continue
                if self.grid[ny, nx]:
                    continue
                if dx and dy and (self.grid[y, nx] or self.grid[ny, x]):
                    continue
                cost = costs[current] + math.hypot(dx, dy)
                neighbor = (nx, ny)
                if cost < costs.get(neighbor, float("inf")):
                    costs[neighbor] = cost
                    previous[neighbor] = current
                    heapq.heappush(frontier, (cost + distance(neighbor), neighbor))
        if closest == origin:
            return None
        while previous[closest] != origin:
            closest = previous[closest]
        return ((closest[0] + 0.5) * self.cell, (closest[1] + 0.5) * self.cell)

    def clear_line(self, start, end):
        count = max(2, math.ceil(math.dist(start, end)) + 1)
        x = np.linspace(start[0], end[0], count).astype(int).clip(0, self.width - 1)
        y = np.linspace(start[1], end[1], count).astype(int).clip(0, self.height - 1)
        return not self.projectiles[y, x].any()
