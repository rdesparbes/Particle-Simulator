import math
from dataclasses import dataclass, field
from typing import (
    Self,
    Tuple,
    Dict,
    Collection,
    Optional,
    Sequence,
    Callable,
    Iterable,
    Iterator,
)

import numpy as np
import numpy.typing as npt

from particle_simulator.geometry import Circle, Rectangle, rotate_2d
from particle_simulator.particle_interaction import ParticleInteraction
from particle_simulator.particle_properties import ParticleProperties

ComputeMagnitudeStrategy = Callable[
    ["Particle", "Particle", float, Optional[float]], float
]


def default_compute_magnitude_strategy(
    part_a: "Particle",
    part_b: "Particle",
    distance: float,
    link_length: Optional[float],
) -> float:
    if link_length is None:
        link_length = max(part_a.props.repel_r, part_b.props.repel_r)
    magnitude = part_a.calculate_magnitude(
        part=part_b,
        distance=distance,
        repel_r=link_length,
        attr=part_b.props.attraction_strength + part_a.props.attraction_strength,
        repel=part_b.props.repulsion_strength + part_a.props.repulsion_strength,
        is_in_group=part_a._is_in_same_group(part_b),
        gravity=part_a.props.gravity_mode or part_b.props.gravity_mode,
    )
    return magnitude


def radii_compute_magnitude_strategy(
    part_a: "Particle",
    part_b: "Particle",
    distance: float,
    link_length: Optional[float],
) -> float:
    magnitude = 0.0
    is_in_group = part_a._is_in_same_group(part_b)
    if part_b._reaches(distance):
        magnitude += part_a.calculate_magnitude(
            part=part_b,
            distance=distance,
            repel_r=part_b.props.repel_r if link_length is None else link_length,
            attr=part_b.props.attraction_strength,
            repel=part_b.props.repulsion_strength,
            is_in_group=is_in_group,
            gravity=part_b.props.gravity_mode,
        )
    if part_a._reaches(distance):
        magnitude += part_b.calculate_magnitude(
            part=part_a,
            distance=distance,
            repel_r=part_a.props.repel_r if link_length is None else link_length,
            attr=part_a.props.attraction_strength,
            repel=part_a.props.repulsion_strength,
            is_in_group=is_in_group,
            gravity=part_a.props.gravity_mode,
        )
    return magnitude


def link_particles(
    particles: Sequence["Particle"],
    fit_link: bool = False,
    distance: Optional[float] = None,
) -> None:
    for p in particles:
        p._link(particles, fit_link=fit_link, distance=distance)


def unlink_particles(particles: Collection["Particle"]) -> None:
    for p in particles:
        p._unlink(particles)


@dataclass(slots=True)
class Particle:
    x: float
    y: float
    velocity: npt.NDArray[np.float_] = field(default_factory=lambda: np.zeros(2))
    radius: float = 4.0
    color: Tuple[int, int, int] = (0, 0, 0)
    props: ParticleProperties = field(default_factory=ParticleProperties)
    # Non-serializable fields:
    link_lengths: Dict[Self, Optional[float]] = field(default_factory=dict)
    mouse: bool = False
    _collisions: Dict[Self, npt.NDArray[np.float_]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _compute_default_force(
        self,
        p: Self,
    ) -> npt.NDArray[np.float_]:
        if self.props.gravity_mode or p.props.gravity_mode:
            return np.zeros(2)
        force = np.random.normal(size=2)
        return force / np.linalg.norm(force) * -self.props.repulsion_strength

    def _compute_collision_delta_pos(
        self, translate_vector: npt.NDArray[np.float_], mass: float
    ) -> npt.NDArray[np.float_]:
        if self.mouse:
            return np.zeros(2)
        return translate_vector * (self.props.mass / (self.props.mass + mass))

    @staticmethod
    def _compute_link_percentage(magnitude: float, max_force: float) -> float:
        if max_force > 0.0:
            return abs(magnitude) / max_force
        return 1.0 if max_force == 0.0 else 0.0

    def _compute_max_force(self, distance: float, repel_r: float) -> float:
        attract = distance >= repel_r
        max_force = (
            self.props.link_attr_breaking_force
            if attract
            else self.props.link_repel_breaking_force
        )
        return max_force

    def _are_compatible(self, p: Self) -> bool:
        return (
            p != self
            and (
                self.props.linked_group_particles
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
        direction = np.subtract([p.x, p.y], [self.x, self.y])
        distance: float = float(np.linalg.norm(direction))
        if not (p._reaches(distance) or self._reaches(distance)):
            return None
        link_percentage: Optional[float] = None
        if distance == 0.0:
            force = self._compute_default_force(p)
        else:
            direction = direction / distance
            link_length: Optional[float] = self.link_lengths.get(p)
            magnitude = compute_magnitude_strategy(self, p, distance, link_length)
            if link_length is None:
                link_length = max(self.props.repel_r, p.props.repel_r)
            if self._is_linked_to(p):
                max_force = p._compute_max_force(distance, link_length)
                link_percentage = self._compute_link_percentage(magnitude, max_force)
            force = direction * magnitude
        return ParticleInteraction(force, link_percentage)

    def iter_interactions(
        self,
        near_particles: Iterable[Self],
        compute_magnitude_strategy: ComputeMagnitudeStrategy = default_compute_magnitude_strategy,
    ) -> Iterator[Tuple[Self, ParticleInteraction]]:
        if self.props.locked:
            return
        for near_particle in near_particles:
            interaction = self._compute_interaction(
                near_particle, compute_magnitude_strategy
            )
            if interaction is not None:
                yield near_particle, interaction

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
    def rectangle(self) -> Rectangle:
        return Rectangle(self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def circle(self) -> Circle:
        return Circle(self.x, self.y, self.radius)

    def rotate(self, x: float, y: float, angle: float) -> None:
        self.x, self.y = rotate_2d(self.x, self.y, x, y, angle)

    @property
    def interacts_with_all(self) -> bool:
        return self.props.attraction_strength != 0.0 and self.props.attract_r < 0

    @property
    def interacts(self) -> bool:
        return (
            self.props.attraction_strength != 0.0
            or self.props.repulsion_strength != 0.0
            or self.props.collisions
        )

    @property
    def range_(self) -> float:
        if self.props.attraction_strength != 0.0 and not (
            self.props.collisions and self.radius > self.props.attract_r
        ):
            return self.props.attract_r
        if (
            self.props.attraction_strength == 0.0
            and self.props.repulsion_strength != 0.0
            and not (self.props.collisions and self.radius > self.props.repel_r)
        ):
            return self.props.repel_r
        return self.radius

    def calculate_magnitude(
        self,
        part: Self,
        attr: float,
        distance: float,
        gravity: bool,
        repel: float,
        is_in_group: bool,
        repel_r: float,
    ) -> float:
        rest_distance = abs(distance - repel_r)
        if distance < repel_r:
            return -repel * rest_distance / 10.0
        if is_in_group or self._is_linked_to(part):
            if gravity:
                return 10.0 * attr * self.props.mass * part.props.mass / distance**2
            return attr * rest_distance / 3000.0
        return 0.0

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Particle):
            return self is other
        raise NotImplementedError

    def _is_linked_to(self, p: Self) -> bool:
        return p in self.link_lengths

    def _is_in_same_group(self, p: Self) -> bool:
        return not self.props.separate_group and p.props.group == self.props.group

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

    def _reaches(self, distance: float) -> bool:
        return (
            self.props.attraction_strength != 0.0
            or self.props.repulsion_strength != 0.0
        ) and (self.props.attract_r < 0 or distance < self.props.attract_r)

    def _compute_collision_speed(self, other: Self) -> npt.NDArray[np.float_]:
        total_mass = self.props.mass + other.props.mass
        return (
            (self.props.mass - other.props.mass) / total_mass * self.velocity
            + 2.0 * other.props.mass / total_mass * other.velocity
        )

    def compute_collision(self, p: Self) -> None:
        if not self.props.collisions:
            return
        direction: npt.NDArray[np.float_] = np.subtract([p.x, p.y], [self.x, self.y])
        distance: float = float(np.linalg.norm(direction))
        overlap = self.radius + p.radius - distance
        if overlap <= 0.0:
            return
        new_speed = self._compute_collision_speed(p)
        p.velocity = p._compute_collision_speed(self)
        self.velocity = new_speed
        translate_vector = overlap * direction
        dx, dy = self._compute_collision_delta_pos(-translate_vector, p.props.mass)
        self.x += dx
        self.y += dy
        if not p.props.locked:
            p_dx, p_dy = p._compute_collision_delta_pos(
                translate_vector, self.props.mass
            )
            p.x += p_dx
            p.y += p_dy
