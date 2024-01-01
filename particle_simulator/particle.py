from typing import (
    Self,
    Any,
    Optional,
    Tuple,
    NamedTuple,
    List,
    Sequence,
    Dict,
    Union,
    Literal,
    Iterable,
    Collection,
    Callable,
)

import numpy as np
import numpy.typing as npt

from .particle_data import ParticleData

_Simulation = Any


class Link(NamedTuple):
    particle_a: "Particle"
    particle_b: "Particle"
    percentage: float


ComputeMagnitudeStrategy = Callable[
    ["Particle", "Particle", float, Optional[float]], float
]


class Particle(ParticleData):
    def __init__(
        self,
        sim: _Simulation,
        x: float,
        y: float,
        radius: float = 4.0,
        color: Union[List[int], Literal["random"]] = "random",
        mass: float = 1.0,
        acceleration: Optional[npt.NDArray[np.float_]] = None,
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
        link_lengths: Optional[Dict[Self, Union[Literal["repel"], float]]] = None,
        link_indices_lengths: Optional[
            Dict[int, Union[Literal["repel"], float]]
        ] = None,
    ) -> None:
        if color == "random":
            color = tuple(np.random.randint(0, 255, 3))
        if acceleration is None:
            acceleration = np.zeros(2)
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
            acceleration=acceleration,
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
        dictionary: Dict[str, Any] = super().__dict__.copy()
        del dictionary["_sim"]
        del dictionary["_collisions"]
        dictionary["link_lengths"] = {
            index_source.index(particle): value
            for particle, value in self.link_lengths.items()
            if particle in index_source
        }
        return dictionary

    def _compute_link(self, part: Self, magnitude: float, max_force: float) -> Link:
        if max_force > 0.0:
            percentage: float = min(abs(magnitude) / max_force, 1.0)
        else:
            percentage = 1.0 if max_force == 0.0 else 0.0
        return Link(
            particle_a=self,
            particle_b=part,
            percentage=percentage,
        )

    def _remove_broken_link(
        self, part: Self, magnitude: float, max_force: float
    ) -> None:
        if 0.0 <= max_force <= abs(magnitude):
            Particle.unlink([self, part])

    def _compute_max_force(self, distance: float, repel_r: float) -> float:
        attract = repel_r >= distance
        max_force = (
            self.link_attr_breaking_force if attract else self.link_repel_breaking_force
        )
        return max_force

    def _calc_magnitude(
        self,
        part: Self,
        distance: float,
        repel_r: Optional[float],
        attr: float,
        repel: float,
        gravity: bool,
    ) -> float:
        repel_r = part.repel_r if repel_r is None else repel_r
        magnitude = self._compute_magnitude(
            part,
            attr,
            distance,
            gravity,
            repel,
            repel_r,
        )
        if self._is_linked_to(part):
            max_force = part._compute_max_force(distance, repel_r)
            if self._sim.stress_visualization:
                self._sim.link_colors.append(
                    self._compute_link(part, magnitude, max_force)
                )
            self._remove_broken_link(part, magnitude, max_force)

        return magnitude

    @staticmethod
    def link(
        particles: List["Particle"],
        fit_link: bool = False,
        distance: Union[None, float, Literal["repel"]] = None,
    ) -> None:
        for p in particles:
            p._link(particles, fit_link=fit_link, distance=distance)

    @staticmethod
    def unlink(particles: Collection["Particle"]) -> None:
        for p in particles:
            p._unlink(particles)

    def update(self, near_particles: Iterable[Self]) -> None:
        if not self._sim.paused:
            self.acceleration = np.zeros(2)
            self._apply_force(self._sim.g_vector * self.mass)  # Gravity
            self._apply_force(self._sim.wind_force * self.radius)

            for force in self._collisions.values():
                self._apply_force(force)

            if not self.locked:
                if self._sim.calculate_radii_diff:
                    compute_magnitude_strategy: ComputeMagnitudeStrategy = (
                        radii_compute_magnitude_strategy
                    )
                else:
                    compute_magnitude_strategy = default_compute_magnitude_strategy
                for near_particle in near_particles:
                    self._compute_interactions(
                        near_particle, compute_magnitude_strategy
                    )

                if not self.mouse:
                    self.velocity += np.clip(self.acceleration, -2, 2) * self._sim.speed
                    self.velocity += (
                        np.random.uniform(-1, 1, 2)
                        * self._sim.temperature
                        * self._sim.speed
                    )
                    self.velocity *= self._sim.air_res_calc
                    self.x += self.velocity[0] * self._sim.speed
                    self.y += self.velocity[1] * self._sim.speed

        if self.mouse:
            delta_mx = self._sim.mx - self._sim.prev_mx
            delta_my = self._sim.my - self._sim.prev_my
            self.x += delta_mx
            self.y += delta_my
            if not self._sim.paused:
                self.velocity = np.divide([delta_mx, delta_my], self._sim.speed)

        if self._sim.right and self.x + self.radius >= self._sim.width:
            self.velocity *= [-self.bounciness, 1 - self._sim.ground_friction]
            self.x = self._sim.width - self.radius
        if self._sim.left and self.x - self.radius <= 0:
            self.velocity *= [-self.bounciness, 1 - self._sim.ground_friction]
            self.x = self.radius
        if self._sim.bottom and self.y + self.radius >= self._sim.height:
            self.velocity *= [1 - self._sim.ground_friction, -self.bounciness]
            self.y = self._sim.height - self.radius
        if self._sim.top and self.y - self.radius <= 0:
            self.velocity *= [1 - self._sim.ground_friction, -self.bounciness]
            self.y = self.radius

        if self._sim.void_edges and (
            self.x - self.radius >= self._sim.width
            or self.x + self.radius <= 0
            or self.y - self.radius >= self._sim.height
            or self.y + self.radius <= 0
        ):
            self._sim.remove_particle(self)
            return

        self._collisions = {}

    def _compute_interactions(
        self, p: Self, compute_magnitude_strategy: ComputeMagnitudeStrategy
    ) -> None:
        if p == self:
            return

        if (
            not self.linked_group_particles
            and not self._is_linked_to(p)
            and self._is_in_same_group(p)
        ) or p in self._collisions:
            return

        # Attract / repel

        direction = np.array([p.x, p.y]) - np.array([self.x, self.y])
        distance: float = np.linalg.norm(direction)
        if distance != 0:
            direction = direction / distance
        are_reaching: Tuple[bool, bool] = (
            p._reaches(distance),
            self._reaches(distance),
        )
        if any(are_reaching):
            if distance == 0.0:
                force = self._compute_default_force(p)
            else:
                repel_r: Optional[float] = self._compute_repel_r(p)
                magnitude = compute_magnitude_strategy(self, p, distance, repel_r)

                force = direction * magnitude
            self._apply_force(force)
            p._collisions[self] = -force

        if self.collisions and distance < self.radius + p.radius:
            new_speed = self._compute_collision_speed(p)
            p.v = p._compute_collision_speed(self)
            self.velocity = new_speed

            # Visual overlap fix
            translate_vector = direction * (distance - (self.radius + p.radius))
            if not self.mouse:
                delta_pos = translate_vector * (self.mass / (self.mass + p.mass))
                self.x += delta_pos[0]
                self.y += delta_pos[1]
            if not p.mouse and not p.locked:
                delta_pos = translate_vector * (p.mass / (self.mass + p.mass))
                p.x -= delta_pos[0]
                p.y -= delta_pos[1]

    def _compute_default_force(
        self,
        p: Self,
    ) -> npt.NDArray[np.float_]:
        if self.gravity_mode or p.gravity_mode:
            return np.zeros(2)
        force = np.random.uniform(-10, 10, 2)
        return force / np.linalg.norm(force) * -self.repulsion_strength

    def _calculate_magnitude(
        self,
        p: Self,
        distance: float,
        repel_r: Optional[float],
    ) -> float:
        return self._calc_magnitude(
            part=p,
            distance=distance,
            repel_r=p.repel_r if repel_r is None else repel_r,
            attr=p.attraction_strength,
            repel=p.repulsion_strength,
            gravity=p.gravity_mode,
        )

    def _compute_repel_r(self, p: Self) -> Optional[float]:
        if self._is_linked_to(p):
            repel_radius = self.link_lengths[p]
            if repel_radius != "repel":
                return repel_radius
        return None


def default_compute_magnitude_strategy(
    part_a: Particle, part_b: Particle, distance: float, repel_r: Optional[float]
) -> float:
    repel_r_ = max(part_a.repel_r, part_b.repel_r) if repel_r is None else repel_r
    magnitude = part_a._calc_magnitude(
        part=part_b,
        distance=distance,
        repel_r=repel_r_,
        attr=part_b.attraction_strength + part_a.attraction_strength,
        repel=part_b.repulsion_strength + part_a.repulsion_strength,
        gravity=part_a.gravity_mode or part_b.gravity_mode,
    )
    return magnitude


def radii_compute_magnitude_strategy(
    part_a: Particle,
    part_b: Particle,
    distance: float,
    repel_r: Optional[float],
) -> float:
    are_reaching: Tuple[bool, bool] = (
        part_b._reaches(distance),
        part_a._reaches(distance),
    )
    if all(are_reaching) and part_a._are_interaction_attributes_equal(part_b):
        # Optimization to avoid having to compute the magnitude twice
        return 2.0 * part_a._calculate_magnitude(
            p=part_b,
            distance=distance,
            repel_r=repel_r,
        )
    magnitude = 0.0
    particles: List[Tuple[Particle, Particle]] = [(part_a, part_b), (part_b, part_a)]
    for reaches, (particle_a, particle_b) in zip(are_reaching, particles):
        if reaches:
            magnitude += particle_a._calculate_magnitude(
                p=particle_b,
                distance=distance,
                repel_r=repel_r,
            )
    return magnitude
