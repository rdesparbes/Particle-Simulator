import random
from typing import (
    Self,
    Any,
    Optional,
    Sequence,
    Dict,
    Iterable,
    Callable,
    Iterator,
    Tuple,
)

import numpy as np
import numpy.typing as npt

from .particle_data import ParticleData
from .particle_interaction import ParticleInteraction
from .simulation_data import SimulationData

ComputeMagnitudeStrategy = Callable[
    ["Particle", "Particle", float, Optional[float]], float
]


def default_compute_magnitude_strategy(
    part_a: "Particle", part_b: "Particle", distance: float, repel_r: Optional[float]
) -> float:
    if repel_r is None:
        repel_r = max(part_a.repel_r, part_b.repel_r)
    magnitude = part_a.calculate_magnitude(
        part=part_b,
        distance=distance,
        repel_r=repel_r,
        attr=part_b.attraction_strength + part_a.attraction_strength,
        repel=part_b.repulsion_strength + part_a.repulsion_strength,
        is_in_group=part_a._is_in_same_group(part_b),
        gravity=part_a.gravity_mode or part_b.gravity_mode,
    )
    return magnitude


def radii_compute_magnitude_strategy(
    part_a: "Particle",
    part_b: "Particle",
    distance: float,
    repel_r: Optional[float],
) -> float:
    magnitude = 0.0
    is_in_group = part_a._is_in_same_group(part_b)
    if part_b.reaches(distance):
        magnitude += part_a.calculate_magnitude(
            part=part_b,
            distance=distance,
            repel_r=part_b.repel_r if repel_r is None else repel_r,
            attr=part_b.attraction_strength,
            repel=part_b.repulsion_strength,
            is_in_group=is_in_group,
            gravity=part_b.gravity_mode,
        )
    if part_a.reaches(distance):
        magnitude += part_b.calculate_magnitude(
            part=part_a,
            distance=distance,
            repel_r=part_a.repel_r if repel_r is None else repel_r,
            attr=part_a.attraction_strength,
            repel=part_a.repulsion_strength,
            is_in_group=is_in_group,
            gravity=part_a.gravity_mode,
        )
    return magnitude


class Particle(ParticleData):
    def __init__(
        self,
        sim: SimulationData,
        x: float,
        y: float,
        radius: float = 4.0,
        color: Optional[Sequence[int]] = None,
        mass: float = 1.0,
        velocity: Optional[npt.NDArray[np.float_]] = None,
        bounciness: float = 0.7,
        locked: bool = False,
        collisions: bool = False,
        attract_r: float = -1.0,
        repel_r: float = 10.0,
        attraction_strength: float = 0.5,
        repulsion_strength: float = 1.0,
        linked_group_particles: bool = True,
        link_attr_breaking_force: float = -1.0,
        link_repel_breaking_force: float = -1.0,
        group: str = "group1",
        separate_group: bool = False,
        gravity_mode: bool = False,
        link_lengths: Optional[Dict[Self, Optional[float]]] = None,
        link_indices_lengths: Optional[Dict[int, Optional[float]]] = None,
    ) -> None:
        if color is None:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        elif len(color) != 3:
            raise ValueError(f"Expected 3 color channels, found {len(color)}")
        else:
            color = (color[0], color[1], color[2])
        if velocity is None:
            velocity = np.zeros(2)
        if link_lengths is None:
            link_lengths = {}
        if link_indices_lengths is None:
            link_indices_lengths = {}
        super().__init__(
            x=x,
            y=y,
            radius=radius,
            color=color,
            mass=mass,
            velocity=velocity,
            bounciness=bounciness,
            locked=locked,
            collisions=collisions,
            attract_r=attract_r,
            repel_r=repel_r,
            attraction_strength=attraction_strength,
            repulsion_strength=repulsion_strength,
            linked_group_particles=linked_group_particles,
            link_attr_breaking_force=link_attr_breaking_force,
            link_repel_breaking_force=link_repel_breaking_force,
            group=group,
            separate_group=separate_group,
            gravity_mode=gravity_mode,
            link_lengths=link_lengths,
            link_indices_lengths=link_indices_lengths,
        )

        self._sim = sim
        self._collisions: Dict[Particle, npt.NDArray[np.float_]] = {}

    def return_dict(self, index_source: Sequence[Self]) -> Dict[str, Any]:
        dictionary: Dict[str, Any] = self.__dict__.copy()
        del dictionary["_sim"]
        del dictionary["_collisions"]
        dictionary["link_lengths"] = {
            index_source.index(particle): value
            for particle, value in self.link_lengths.items()
            if particle in index_source
        }
        return dictionary

    @staticmethod
    def _compute_link_percentage(magnitude: float, max_force: float) -> float:
        if max_force > 0.0:
            return abs(magnitude) / max_force
        return 1.0 if max_force == 0.0 else 0.0

    def _compute_max_force(self, distance: float, repel_r: float) -> float:
        attract = distance >= repel_r
        max_force = (
            self.link_attr_breaking_force if attract else self.link_repel_breaking_force
        )
        return max_force

    def iter_interactions(
        self,
        near_particles: Iterable[Self],
        compute_magnitude_strategy: ComputeMagnitudeStrategy = default_compute_magnitude_strategy,
    ) -> Iterator[Tuple[Self, ParticleInteraction]]:
        if self.locked:
            return
        for near_particle in near_particles:
            interaction = self._compute_interaction(
                near_particle, compute_magnitude_strategy
            )
            if interaction is not None:
                yield near_particle, interaction

    def _compute_delta_velocity(
        self, force: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        forces = [force, self._sim.wind_force * self.radius]
        acceleration = np.sum(forces, axis=0) / self.mass + self._sim.g_vector

        return (
            np.clip(acceleration, -2, 2)
            + np.random.uniform(-1, 1, 2) * self._sim.temperature
        )

    def update(self, force: Optional[npt.NDArray[np.float_]] = None) -> None:
        if self.mouse:
            delta_mouse_pos = self._sim.delta_mouse_pos
            if not self._sim.paused:
                self.velocity = delta_mouse_pos * self._sim.speed
            delta_mx, delta_my = delta_mouse_pos
            self.x += delta_mx
            self.y += delta_my
        elif force is not None and not self.locked:
            self.velocity += self._compute_delta_velocity(force)
            self.velocity *= self._sim.air_res_calc
            dx, dy = self.velocity * self._sim.speed
            self.x += dx
            self.y += dy

        if self._sim.right and self.x_max >= self._sim.width:
            self.velocity *= [-self.bounciness, 1 - self._sim.ground_friction]
            self.x = self._sim.width - self.radius
        if self._sim.left and self.x_min <= 0:
            self.velocity *= [-self.bounciness, 1 - self._sim.ground_friction]
            self.x = self.radius
        if self._sim.bottom and self.y_max >= self._sim.height:
            self.velocity *= [1 - self._sim.ground_friction, -self.bounciness]
            self.y = self._sim.height - self.radius
        if self._sim.top and self.y_min <= 0:
            self.velocity *= [1 - self._sim.ground_friction, -self.bounciness]
            self.y = self.radius

        self._collisions = {}

    def fix_overlap(self, p: Self) -> None:
        direction: npt.NDArray[np.float_] = np.subtract([p.x, p.y], [self.x, self.y])
        distance: float = float(np.linalg.norm(direction))
        overlap = self.radius + p.radius - distance
        if not self.collisions or overlap <= 0.0:
            return
        new_speed = self._compute_collision_speed(p)
        p.velocity = p._compute_collision_speed(self)
        self.velocity = new_speed
        translate_vector = overlap * direction
        dx, dy = self._compute_collision_delta_pos(-translate_vector, p.mass)
        self.x += dx
        self.y += dy
        if not p.locked:
            p_dx, p_dy = p._compute_collision_delta_pos(translate_vector, self.mass)
            p.x += p_dx
            p.y += p_dy

    def _are_compatible(self, p: Self) -> bool:
        return (
            p != self
            and (
                self.linked_group_particles
                or self._is_linked_to(p)
                or not self._is_in_same_group(p)
            )
            and p not in self._collisions
        )

    def _compute_interaction(
        self, p: Self, compute_magnitude_strategy: ComputeMagnitudeStrategy
    ) -> Optional[ParticleInteraction]:
        if not self._are_compatible(p):
            return None
        direction = np.array([p.x, p.y]) - np.array([self.x, self.y])
        distance: float = float(np.linalg.norm(direction))
        if not (p.reaches(distance) or self.reaches(distance)):
            return None
        link_percentage: Optional[float] = None
        if distance == 0.0:
            force = self._compute_default_force(p)
        else:
            direction = direction / distance
            repel_r: Optional[float] = self.link_lengths.get(p)
            magnitude = compute_magnitude_strategy(self, p, distance, repel_r)
            if repel_r is None:
                repel_r = max(self.repel_r, p.repel_r)
            if self._is_linked_to(p):
                max_force = p._compute_max_force(distance, repel_r)
                link_percentage = self._compute_link_percentage(magnitude, max_force)
            force = direction * magnitude
        return ParticleInteraction(force, link_percentage)

    def _compute_collision_delta_pos(
        self, translate_vector: npt.NDArray[np.float_], mass: float
    ) -> npt.NDArray[np.float_]:
        if self.mouse:
            return np.zeros(2)
        return translate_vector * (self.mass / (self.mass + mass))

    def _compute_default_force(
        self,
        p: Self,
    ) -> npt.NDArray[np.float_]:
        if self.gravity_mode or p.gravity_mode:
            return np.zeros(2)
        force = np.random.uniform(-10, 10, 2)
        return force / np.linalg.norm(force) * -self.repulsion_strength
