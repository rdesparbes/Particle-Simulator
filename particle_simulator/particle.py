import random
from typing import (
    Self,
    Optional,
    Sequence,
    Dict,
    Any,
)

import numpy as np
import numpy.typing as npt

from .particle_data import ParticleData
from .simulation_data import SimulationData


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

    def return_dict(self, index_source: Sequence[Self]) -> Dict[str, Any]:
        dictionary: Dict[str, Any] = self.to_dict()
        dictionary["link_lengths"] = {
            index_source.index(particle): value
            for particle, value in self.link_lengths.items()
            if particle in index_source
        }
        return dictionary

    def _compute_delta_velocity(
        self, sim_data: SimulationData, force: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        forces = [force, sim_data.wind_force * self.radius]
        acceleration = np.sum(forces, axis=0) / self.mass + sim_data.g_vector

        return (
            np.clip(acceleration, -2, 2)
            + np.random.uniform(-1, 1, 2) * sim_data.temperature
        )

    def update(self, sim_data: SimulationData, force: Optional[npt.NDArray[np.float_]] = None) -> None:
        if self.mouse:
            self.velocity = sim_data.delta_mouse_pos
            dx, dy = self.velocity
            self.x += dx
            self.y += dy
        elif force is not None and not self.locked:
            self.velocity += self._compute_delta_velocity(sim_data, force)
            self.velocity *= sim_data.air_res_calc
            dx, dy = self.velocity * sim_data.speed
            self.x += dx
            self.y += dy

        if sim_data.right and self.x_max >= sim_data.width:
            self.velocity *= [-self.bounciness, 1 - sim_data.ground_friction]
            self.x = sim_data.width - self.radius
        if sim_data.left and self.x_min <= 0:
            self.velocity *= [-self.bounciness, 1 - sim_data.ground_friction]
            self.x = self.radius
        if sim_data.bottom and self.y_max >= sim_data.height:
            self.velocity *= [1 - sim_data.ground_friction, -self.bounciness]
            self.y = sim_data.height - self.radius
        if sim_data.top and self.y_min <= 0:
            self.velocity *= [1 - sim_data.ground_friction, -self.bounciness]
            self.y = self.radius

        self._collisions = {}
