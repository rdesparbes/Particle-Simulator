from typing import Self, List, Union, Literal, Dict, Any, Optional, Sequence

import numpy as np
import numpy.typing as npt


class ParticleData:
    def __init__(
        self,
        x: float,
        y: float,
        radius: int = 4,
        color: Union[List[int], Literal["random"]] = "random",
        mass: float = 1,
        velocity: npt.NDArray[np.float_] = np.zeros(2),
        bounciness: float = 0.7,
        locked: bool = False,
        collisions: bool = False,
        attract_r: int = -1,
        repel_r: int = 10,
        attraction_strength: float = 0.5,
        repulsion_strength: float = 1,
        linked_group_particles: bool = True,
        link_attr_breaking_force: float = -1,
        link_repel_breaking_force: float = -1,
        group: str = "group1",
        separate_group: bool = False,
        gravity_mode: bool = False,
    ) -> None:
        self.x = x
        self.y = y
        self.r = radius
        if color == "random":
            self.color: List[int] = np.random.randint(0, 255, 3).tolist()
        else:
            self.color = color

        self.m = mass
        self.v = np.array(velocity).astype(np.float32)
        self.a = np.zeros(2)
        self.bounciness = bounciness
        self.collision_bool = collisions
        self.locked = locked

        self.attr_r = attract_r
        self.repel_r = repel_r
        self.attr = attraction_strength
        self.repel = repulsion_strength
        self.gravity_mode = gravity_mode

        self.return_all: Optional[bool] = None
        self.return_none: Optional[bool] = None
        self.range_: Optional[int] = None
        self.init_constants()

        self.linked: List[ParticleData] = []
        self.link_lengths: Dict[ParticleData, Union[Literal["repel"], float]] = {}
        self.linked_group_particles = linked_group_particles
        self.link_attr_breaking_force = link_attr_breaking_force
        self.link_repel_breaking_force = link_repel_breaking_force
        self.separate_group = separate_group

        self.group = group

        self.collisions: List[ParticleData] = []
        self.forces: List[npt.NDArray[np.float_]] = []

        self.mouse = False

    def init_constants(self) -> None:
        self.return_all = self.attr_r < 0 and self.attr != 0
        self.return_none = (
            self.attr == 0 and self.repel == 0 and not self.collision_bool
        )
        if self.attr != 0 and not (self.collision_bool and self.r > self.attr_r):
            self.range_ = self.attr_r
        elif (
            self.attr == 0
            and self.repel != 0
            and not (self.collision_bool and self.r > self.repel_r)
        ):
            self.range_ = self.repel_r
        else:
            self.range_ = self.r

    def return_dict(self, index_source: Sequence[Self]) -> Dict[str, Any]:
        dictionary = self.__dict__.copy()
        dictionary.pop("sim", None)
        del dictionary["collisions"]
        del dictionary["forces"]

        dictionary["linked"] = [
            index_source.index(particle)
            for particle in dictionary["linked"]
            if particle in index_source
        ]
        dictionary["link_lengths"] = {
            index_source.index(particle): value
            for particle, value in dictionary["link_lengths"].items()
            if particle in index_source
        }

        return dictionary

    def _apply_force(self, force: npt.NDArray[np.float_]) -> None:
        self.a = self.a + force / abs(self.m)

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
                return attr * self.m * part.m / distance**2 * 10
            else:
                return attr * rest_distance / 3000
        return 0.0

    def _interacts(self, distance: float) -> bool:
        return (self.attr != 0 or self.repel != 0) and (
            self.attr_r < 0 or self.attr_r < 0 or distance < self.attr_r
        )

    def _compute_collision_speed(self, other: Self) -> npt.NDArray[np.float_]:
        return (self.m - other.m) / (self.m + other.m) * self.v + 2 * other.m / (
            self.m + other.m
        ) * other.v

    def _are_interaction_attributes_equal(self, other: Self) -> bool:
        return (
            self.attr == other.attr
            and self.repel == other.repel
            and self.link_attr_breaking_force == other.link_attr_breaking_force
            and self.link_repel_breaking_force == other.link_repel_breaking_force
            and self.gravity_mode == other.gravity_mode
        )
