from typing import Iterable, Iterator

import numpy as np

from .particle import Particle


class Grid:
    def __init__(self, rows: int, columns: int, height: int, width: int) -> None:
        self.grid = np.empty((rows, columns), dtype="object")
        self.rows = rows
        self.columns = columns
        self.row_height: float = height / self.rows
        self.column_width: float = width / self.columns
        self.reset_grid()

    def reset_grid(self) -> None:
        for i in range(self.rows):
            for j in range(self.columns):
                self.grid[i, j] = []

    def extend(self, particles: Iterable[Particle]) -> None:
        for particle in particles:
            row = self._return_row(particle.y)
            column = self._return_column(particle.x)
            if 0 <= row < self.rows and 0 <= column < self.columns:
                self.grid[row, column].append(particle)

    def _return_row(self, y: float) -> int:
        return min(int(y / self.row_height), self.rows - 1)

    def _return_column(self, x: float) -> int:
        return min(int(x / self.column_width), self.rows - 1)

    def return_particles(self, particle: Particle) -> Iterator[Particle]:
        p_range = particle.range_
        min_row = self._return_row(particle.y - p_range)
        max_row = self._return_row(particle.y + p_range)
        min_col = self._return_column(particle.x - p_range)
        max_col = self._return_column(particle.x + p_range)
        for i in range(min_row, max_row + 1):
            for j in range(min_col, max_col + 1):
                if 0 <= i < self.rows and 0 <= j < self.columns:
                    for particle in self.grid[i][j]:
                        yield particle
