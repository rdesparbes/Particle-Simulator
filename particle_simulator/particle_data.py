import math
from dataclasses import dataclass, field
from typing import (
    Self,
    Tuple,
    Dict,
    Collection,
    Optional,
    Sequence,
)

import numpy as np
import numpy.typing as npt


@dataclass
class ParticleData:
    x: float
    y: float
    velocity: npt.NDArray[np.float_] = field(default_factory=lambda: np.zeros(2))
    radius: float = 4.0
    color: Tuple[int, int, int] = (0, 0, 0)
    mass: float = 1.0
    bounciness: float = 0.7
    locked: bool = False
    collisions: bool = False
    attract_r: float = -1.0
    repel_r: float = 10.0
    attraction_strength: float = 0.5
    repulsion_strength: float = 1.0
    linked_group_particles: bool = True
    link_attr_breaking_force: float = -1.0
    link_repel_breaking_force: float = -1.0
    group: str = "group1"
    separate_group: bool = False
    gravity_mode: bool = False
    mouse = False
    link_lengths: Dict[Self, Optional[float]] = field(default_factory=dict)
    link_indices_lengths: Dict[int, Optional[float]] = field(default_factory=dict)

    def distance(self, x: float, y: float) -> float:
        return math.dist((x, y), (self.x, self.y))

    @property
    def x_min(self) -> float:
        return self.x - self.radius

    @property
    def y_min(self) -> float:
        return self.y - self.radius

    @property
    def x_max(self) -> float:
        return self.x + self.radius

    @property
    def y_max(self) -> float:
        return self.y + self.radius

    @property
    def interacts_with_all(self) -> bool:
        return self.attraction_strength != 0.0 and self.attract_r < 0

    @property
    def interacts(self) -> bool:
        return (
            self.attraction_strength != 0.0
            or self.repulsion_strength != 0.0
            or self.collisions
        )

    @property
    def range_(self) -> float:
        if self.attraction_strength != 0.0 and not (
            self.collisions and self.radius > self.attract_r
        ):
            return self.attract_r
        if (
            self.attraction_strength == 0.0
            and self.repulsion_strength != 0.0
            and not (self.collisions and self.radius > self.repel_r)
        ):
            return self.repel_r
        return self.radius

    def _apply_force(self, force: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return force / abs(self.mass)

    def calculate_magnitude(
        self,
        part: Self,
        attr: float,
        distance: float,
        gravity: bool,
        repel: float,
        repel_r: float,
    ) -> float:
        rest_distance = abs(distance - repel_r)
        if distance < repel_r:
            return -repel * rest_distance / 10.0
        if self._is_in_same_group(part) or self._is_linked_to(part):
            if gravity:
                return 10.0 * attr * self.mass * part.mass / distance**2
            return attr * rest_distance / 3000.0
        return 0.0

    def __hash__(self) -> int:
        return id(self)

    def _is_linked_to(self, p: Self) -> bool:
        return p in self.link_lengths

    def _is_in_same_group(self, p: Self) -> bool:
        return not self.separate_group and p.group == self.group

    def _link(
        self,
        particles: Sequence[Self],
        fit_link: bool = False,
        distance: Optional[float] = None,
    ) -> None:
        def _compute_link_length(p: Self) -> Optional[float]:
            if not fit_link:
                return None
            if distance is None:
                return self.distance(p.x, p.y)
            return distance

        for particle in particles:
            if particle is not self:
                self.link_lengths[particle] = _compute_link_length(particle)

    def _unlink(self, particles: Collection[Self]) -> None:
        self.link_lengths = {
            other: length
            for other, length in self.link_lengths.items()
            if other not in particles
        }

    def reaches(self, distance: float) -> bool:
        return (self.attraction_strength != 0.0 or self.repulsion_strength != 0.0) and (
            self.attract_r < 0 or distance < self.attract_r
        )

    def _compute_collision_speed(self, other: Self) -> npt.NDArray[np.float_]:
        return (self.mass - other.mass) / (
            self.mass + other.mass
        ) * self.velocity + 2 * other.mass / (self.mass + other.mass) * other.velocity

    @staticmethod
    def link(
        particles: Sequence["ParticleData"],
        fit_link: bool = False,
        distance: Optional[float] = None,
    ) -> None:
        for p in particles:
            p._link(particles, fit_link=fit_link, distance=distance)

    @staticmethod
    def unlink(particles: Collection["ParticleData"]) -> None:
        for p in particles:
            p._unlink(particles)
