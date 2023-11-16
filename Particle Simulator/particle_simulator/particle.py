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
)

import numpy as np
import numpy.typing as npt

from .particle_data import ParticleData

_Simulation = Any


class Particle(ParticleData):
    def __init__(self, sim: _Simulation, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.linked: List[Particle] = []
        self.link_lengths: Dict[Particle, Union[Literal["repel"], float]] = {}

        self._sim = sim
        self._collisions: Dict[Particle, npt.NDArray[np.float_]] = {}

    def return_dict(self, index_source: Sequence[Self]) -> Dict[str, Any]:
        dictionary: Dict[str, Any] = super().__dict__.copy()
        del dictionary["_sim"]
        del dictionary["_collisions"]

        dictionary["linked"] = [
            index_source.index(particle)
            for particle in self.linked
            if particle in index_source
        ]
        dictionary["link_lengths"] = {
            index_source.index(particle): value
            for particle, value in self.link_lengths.items()
            if particle in index_source
        }
        return dictionary

    def _calc_magnitude(
        self,
        part: Self,
        distance: float,
        repel_r: Optional[float],
        attr: float,
        repel: float,
        is_in_group: bool,
        is_linked: bool,
        gravity: bool,
    ) -> float:
        repel_r = part.repel_r if repel_r is None else repel_r
        magnitude = self._compute_magnitude(
            part,
            attr,
            distance,
            gravity,
            is_in_group,
            is_linked,
            repel,
            repel_r,
        )

        if is_linked:
            attract = repel_r >= distance
            max_force = (
                part.link_attr_breaking_force
                if attract
                else part.link_repel_breaking_force
            )
            if self._sim.stress_visualization:
                if max_force > 0.0:
                    percentage: float = round(abs(magnitude) / max_force, 2)
                else:
                    percentage = 1.0 if max_force == 0.0 else 0.0

                self._sim.link_colors.append(
                    Link(
                        particle_a=self, particle_b=part, percentage=min(percentage, 1)
                    )
                )

            if 0 <= max_force <= abs(magnitude):
                self._sim.unlink([self, part])

        return magnitude

    def update(self, near_particles: Iterable[Self]) -> None:
        if not self._sim.paused:
            self.a = np.zeros(2)
            self._apply_force(self._sim.g_vector * self.m)  # Gravity
            self._apply_force(self._sim.wind_force * self.r)

            for force in self._collisions.values():
                self._apply_force(force)

            if not self.locked:
                for near_particle in near_particles:
                    self._compute_interactions(near_particle)

                if not self.mouse:
                    self.v += np.clip(self.a, -2, 2) * self._sim.speed
                    self.v += (
                        np.random.uniform(-1, 1, 2)
                        * self._sim.temperature
                        * self._sim.speed
                    )
                    self.v *= self._sim.air_res_calc
                    self.x += self.v[0] * self._sim.speed
                    self.y += self.v[1] * self._sim.speed

        if self.mouse:
            delta_mx = self._sim.mx - self._sim.prev_mx
            delta_my = self._sim.my - self._sim.prev_my
            self.x += delta_mx
            self.y += delta_my
            if not self._sim.paused:
                self.v = np.divide([delta_mx, delta_my], self._sim.speed)

        if self._sim.right and self.x + self.r >= self._sim.width:
            self.v *= [-self.bounciness, 1 - self._sim.ground_friction]
            self.x = self._sim.width - self.r
        if self._sim.left and self.x - self.r <= 0:
            self.v *= [-self.bounciness, 1 - self._sim.ground_friction]
            self.x = self.r
        if self._sim.bottom and self.y + self.r >= self._sim.height:
            self.v *= [1 - self._sim.ground_friction, -self.bounciness]
            self.y = self._sim.height - self.r
        if self._sim.top and self.y - self.r <= 0:
            self.v *= [1 - self._sim.ground_friction, -self.bounciness]
            self.y = self.r

        if self._sim.void_edges and (
            self.x - self.r >= self._sim.width
            or self.x + self.r <= 0
            or self.y - self.r >= self._sim.height
            or self.y + self.r <= 0
        ):
            self._sim.remove_particle(self)
            return

        self._collisions = {}

    def _compute_interactions(self, p: Self) -> None:
        if p == self:
            return
        is_in_group = not self.separate_group and p in self._sim.groups[self.group]
        is_linked = p in self.linked

        if (
            not self.linked_group_particles and not is_linked and is_in_group
        ) or p in self._collisions:
            return

        # Attract / repel

        direction = np.array([p.x, p.y]) - np.array([self.x, self.y])
        distance: float = np.linalg.norm(direction)
        if distance != 0:
            direction = direction / distance
        conditions: Tuple[bool, bool] = (
            p._interacts(distance),
            self._interacts(distance),
        )
        if any(conditions):
            force = self._compute_force(
                p, direction, distance, conditions, is_linked, is_in_group
            )

            self._apply_force(force)
            p._collisions[self] = -force

        if self.collision_bool and distance < self.r + p.r:
            new_speed = self._compute_collision_speed(p)
            p.v = p._compute_collision_speed(self)
            self.v = new_speed

            # Visual overlap fix
            translate_vector = direction * (distance - (self.r + p.r))
            if not self.mouse:
                delta_pos = translate_vector * (self.m / (self.m + p.m))
                self.x += delta_pos[0]
                self.y += delta_pos[1]
            if not p.mouse and not p.locked:
                delta_pos = translate_vector * (p.m / (self.m + p.m))
                p.x -= delta_pos[0]
                p.y -= delta_pos[1]

    def _compute_force(
        self,
        p: Self,
        direction: npt.NDArray[np.float_],
        distance: float,
        conditions: Tuple[bool, bool],
        is_linked: bool,
        is_in_group: bool,
    ) -> npt.NDArray[np.float_]:
        if distance == 0.0:
            if self.gravity_mode or p.gravity_mode:
                return np.zeros(2)
            force = np.random.uniform(-10, 10, 2)
            return force / np.linalg.norm(force) * -self.repel
        return direction * self._compute_magn(
            p,
            conditions,
            distance,
            is_in_group,
            is_linked,
        )

    def _calculate_magnitude(
        self,
        p: Self,
        distance: float,
        repel_r: Optional[float],
        is_in_group: bool,
        is_linked: bool,
    ) -> float:
        return self._calc_magnitude(
            part=p,
            distance=distance,
            repel_r=p.repel_r if repel_r is None else repel_r,
            attr=p.attr,
            repel=p.repel,
            is_in_group=is_in_group,
            is_linked=is_linked,
            gravity=p.gravity_mode,
        )

    def _compute_magn(
        self,
        p: Self,
        conditions: Tuple[bool, bool],
        distance: float,
        is_in_group: bool,
        is_linked: bool,
    ) -> float:
        repel_r: Optional[float] = None
        if is_linked:
            repel_radius = self.link_lengths[p]
            if repel_radius != "repel":
                repel_r = repel_radius

        if self._sim.calculate_radii_diff:
            if all(conditions) and self._are_interaction_attributes_equal(p):
                # Optimization to avoid having to compute the magnitude twice
                return 2.0 * self._calculate_magnitude(
                    p=p,
                    distance=distance,
                    repel_r=repel_r,
                    is_in_group=is_in_group,
                    is_linked=is_linked,
                )
            magnitude = 0.0
            particles: List[Tuple[Particle, Particle]] = [(self, p), (p, self)]
            for condition, (particle_a, particle_b) in zip(conditions, particles):
                if condition:
                    magnitude += particle_a._calculate_magnitude(
                        p=particle_b,
                        distance=distance,
                        repel_r=repel_r,
                        is_in_group=is_in_group,
                        is_linked=is_linked,
                    )
            return magnitude
        repel_r_ = max(self.repel_r, p.repel_r) if repel_r is None else repel_r
        magnitude = self._calc_magnitude(
            part=p,
            distance=distance,
            repel_r=repel_r_,
            attr=p.attr + self.attr,
            repel=p.repel + self.repel,
            is_in_group=is_in_group,
            is_linked=is_linked,
            gravity=self.gravity_mode or p.gravity_mode,
        )
        return magnitude


class Link(NamedTuple):
    particle_a: Particle
    particle_b: Particle
    percentage: float
