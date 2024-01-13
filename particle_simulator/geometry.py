import math
from typing import NamedTuple

from typing_extensions import Self


class Rectangle(NamedTuple):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def isdisjoint(self, item: Self) -> bool:
        return (
            self.x_min >= item.x_max
            or self.y_min >= item.y_max
            or self.x_max <= item.x_min
            or self.y_max <= item.y_min
        )


class Circle(NamedTuple):
    x: float
    y: float
    radius: float

    def is_in_range(self, item: Self) -> bool:
        """Symmetric operation to check if at least one of
        the two circles contains the center of the other
        """
        return math.dist((self.x, self.y), (item.x, item.y)) <= max(
            self.radius, item.radius
        )
