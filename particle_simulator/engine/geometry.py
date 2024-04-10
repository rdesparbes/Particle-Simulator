import math
from typing import NamedTuple, Tuple

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


def rotate_2d(
    x: float, y: float, cx: float, cy: float, angle: float
) -> Tuple[float, float]:
    angle_rad = -math.radians(angle)
    dist_x = x - cx
    dist_y = y - cy
    current_angle = math.atan2(dist_y, dist_x)
    angle_rad += current_angle
    radius = math.hypot(dist_x, dist_y)
    x = cx + radius * math.cos(angle_rad)
    y = cy + radius * math.sin(angle_rad)

    return x, y
