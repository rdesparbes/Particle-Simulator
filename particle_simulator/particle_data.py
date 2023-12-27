from dataclasses import dataclass, field
from typing import Self, Tuple, Union, Literal, Dict

import numpy as np
import numpy.typing as npt


@dataclass
class ParticleData:
    x: float
    y: float
    velocity: npt.NDArray[np.float_] = field(default_factory=lambda: np.zeros(2))
    acceleration: npt.NDArray[np.float_] = field(default_factory=lambda: np.zeros(2))
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
    link_lengths: Dict[Self, Union[Literal["repel"], float]] = field(
        default_factory=dict
    )
    link_indices_lengths: Dict[int, Union[Literal["repel"], float]] = field(
        default_factory=dict
    )

    @property
    def interacts_will_all(self) -> bool:
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

    def _apply_force(self, force: npt.NDArray[np.float_]) -> None:
        self.acceleration += force / abs(self.mass)

    def _compute_magnitude(
        self,
        part: Self,
        attr: float,
        distance: float,
        gravity: bool,
        is_in_group: bool,
        is_linked: bool,
        repel: float,
        repel_r: float,
    ) -> float:
        rest_distance = abs(distance - repel_r)
        if distance < repel_r:
            return -repel * rest_distance / 10
        elif is_in_group or is_linked:
            if gravity:
                return attr * self.mass * part.mass / distance**2 * 10
            else:
                return attr * rest_distance / 3000
        return 0.0

    def __hash__(self) -> int:
        return id(self)

    def _interacts(self, distance: float) -> bool:
        return (self.attraction_strength != 0.0 or self.repulsion_strength != 0.0) and (
            self.attract_r < 0 or distance < self.attract_r
        )

    def _compute_collision_speed(self, other: Self) -> npt.NDArray[np.float_]:
        return (self.mass - other.mass) / (
            self.mass + other.mass
        ) * self.velocity + 2 * other.mass / (self.mass + other.mass) * other.velocity

    def _are_interaction_attributes_equal(self, other: Self) -> bool:
        return (
            self.attraction_strength == other.attraction_strength
            and self.repulsion_strength == other.repulsion_strength
            and self.link_attr_breaking_force == other.link_attr_breaking_force
            and self.link_repel_breaking_force == other.link_repel_breaking_force
            and self.gravity_mode == other.gravity_mode
        )
