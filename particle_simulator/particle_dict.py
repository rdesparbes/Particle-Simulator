from typing import TypedDict, Tuple, Literal, Union, NotRequired


class ParticleDict(TypedDict):
    radius: NotRequired[float]
    color: NotRequired[Union[Tuple[int, int, int], Literal["random"]]]
    mass: NotRequired[float]
    velocity: NotRequired[Tuple[float, float]]
    bounciness: NotRequired[float]
    locked: NotRequired[bool]
    collisions: NotRequired[bool]
    attract_r: NotRequired[int]
    repel_r: NotRequired[int]
    attraction_strength: NotRequired[float]
    repulsion_strength: NotRequired[float]
    linked_group_particles: NotRequired[bool]
    link_attr_breaking_force: NotRequired[float]
    link_repel_breaking_force: NotRequired[float]
    group: NotRequired[str]
    separate_group: NotRequired[bool]
    gravity_mode: NotRequired[bool]
