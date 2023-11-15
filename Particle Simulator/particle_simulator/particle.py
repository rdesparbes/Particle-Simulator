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

        self.collisions: List[Particle] = []
        self.forces: List[npt.NDArray[np.float_]] = []
        self.linked: List[Particle] = []
        self.link_lengths: Dict[Particle, Union[Literal["repel"], float]] = {}
        self.sim = sim

    def return_dict(self, index_source: Sequence[Self]) -> Dict[str, Any]:
        dictionary: Dict[str, Any] = super().__dict__.copy()
        del dictionary["sim"]
        del dictionary["collisions"]
        del dictionary["forces"]

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
        repel_r: float,
        attr: float,
        repel: float,
        is_in_group: bool,
        is_linked: bool,
        gravity: bool,
    ) -> float:
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
            if self.sim.stress_visualization:
                if max_force > 0.0:
                    percentage: float = round(abs(magnitude) / max_force, 2)
                else:
                    percentage = 1.0 if max_force == 0.0 else 0.0

                self.sim.link_colors.append(
                    Link(
                        particle_a=self, particle_b=part, percentage=min(percentage, 1)
                    )
                )

            if 0 <= max_force <= abs(magnitude):
                self.sim.unlink([self, part])

        return magnitude

    def update(self, near_particles: Iterable[Self]) -> None:
        if not self.sim.paused:
            self.a = self.sim.g_vector * np.sign(self.m)  # Gravity

            self._apply_force(self.sim.wind_force * self.r)

            for force in self.forces:
                self._apply_force(force)

            if not self.locked:
                for near_particle in near_particles:
                    self._compute_interactions(near_particle)

                if not self.mouse:
                    self.v += np.clip(self.a, -2, 2) * self.sim.speed
                    self.v += (
                        np.random.uniform(-1, 1, 2)
                        * self.sim.temperature
                        * self.sim.speed
                    )
                    self.v *= self.sim.air_res_calc
                    self.x += self.v[0] * self.sim.speed
                    self.y += self.v[1] * self.sim.speed

        if self.mouse:
            delta_mx = self.sim.mx - self.sim.prev_mx
            delta_my = self.sim.my - self.sim.prev_my
            self.x += delta_mx
            self.y += delta_my
            if not self.sim.paused:
                self.v = np.divide([delta_mx, delta_my], self.sim.speed)

        if self.sim.right and self.x + self.r >= self.sim.width:
            self.v *= [-self.bounciness, 1 - self.sim.ground_friction]
            self.x = self.sim.width - self.r
        if self.sim.left and self.x - self.r <= 0:
            self.v *= [-self.bounciness, 1 - self.sim.ground_friction]
            self.x = self.r
        if self.sim.bottom and self.y + self.r >= self.sim.height:
            self.v *= [1 - self.sim.ground_friction, -self.bounciness]
            self.y = self.sim.height - self.r
        if self.sim.top and self.y - self.r <= 0:
            self.v *= [1 - self.sim.ground_friction, -self.bounciness]
            self.y = self.r

        if self.sim.void_edges and (
            self.x - self.r >= self.sim.width
            or self.x + self.r <= 0
            or self.y - self.r >= self.sim.height
            or self.y + self.r <= 0
        ):
            self.sim.remove_particle(self)
            return

        self.collisions = []
        self.forces = []

    def _compute_interactions(self, p: Self) -> None:
        if p == self:
            return
        is_in_group = not self.separate_group and p in self.sim.groups[self.group]
        is_linked = p in self.linked

        if (
            not self.linked_group_particles and not is_linked and is_in_group
        ) or p in self.collisions:
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
                p,
                conditions,
                direction,
                distance,
                is_in_group,
                is_linked,
            )

            self._apply_force(force)
            p.forces.append(-force)
            p.collisions.append(self)

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
        conditions: Tuple[bool, bool],
        direction: npt.NDArray[np.float_],
        distance: float,
        is_in_group: bool,
        is_linked: bool,
    ) -> npt.NDArray[np.float_]:
        if distance == 0.0:
            if self.gravity_mode or p.gravity_mode:
                return np.zeros(2)
            force = np.random.uniform(-10, 10, 2)
            return force / np.linalg.norm(force) * -self.repel

        repel_r: Optional[float] = None
        if is_linked:
            repel_radius = self.link_lengths[p]
            if repel_radius != "repel":
                repel_r = repel_radius

        if self.sim.calculate_radii_diff:
            force = np.zeros(2)
            particles: Tuple[Particle, Particle] = p, self
            for i, particle in enumerate(particles):
                if conditions[i]:
                    repel_r_: float = particle.repel_r if repel_r is None else repel_r

                    if i == 1 and self._are_interaction_attributes_equal(p):
                        # Optimization to avoid having to compute the same force twice:
                        force *= 2
                    else:
                        magnitude = self._calc_magnitude(
                            part=particle,
                            distance=distance,
                            repel_r=repel_r_,
                            attr=particle.attr,
                            repel=particle.repel,
                            is_in_group=is_in_group,
                            is_linked=is_linked,
                            gravity=particle.gravity_mode,
                        )
                        force += magnitude * direction
            return force

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
        return magnitude * direction


class Link(NamedTuple):
    particle_a: Particle
    particle_b: Particle
    percentage: float
