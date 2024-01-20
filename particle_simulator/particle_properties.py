from dataclasses import dataclass


@dataclass
class ParticleProperties:
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
