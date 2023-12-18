from dataclasses import dataclass
from typing import Tuple, Literal, Union, Optional


@dataclass
class ParticleFactory:
    color: Union[Tuple[int, int, int], Literal["random"]]
    mass: float
    velocity: Tuple[float, float]
    bounciness: float
    locked: bool
    collisions: bool
    attract_r: float
    repel_r: float
    attraction_strength: float
    repulsion_strength: float
    linked_group_particles: bool
    link_attr_breaking_force: float
    link_repel_breaking_force: float
    group: str
    separate_group: bool
    gravity_mode: bool
    radius: Optional[float] = None
