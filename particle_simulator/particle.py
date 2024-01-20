import random
from typing import (
    Self,
    Optional,
    Sequence,
    Dict,
)

import numpy as np
import numpy.typing as npt

from .particle_data import ParticleData
from .particle_properties import ParticleProperties
from .simulation_data import SimulationData


class Particle(ParticleData):
    def __init__(
        self,
        x: float,
        y: float,
        radius: float = 4.0,
        color: Optional[Sequence[int]] = None,
        props: Optional[ParticleProperties] = None,
        velocity: Optional[npt.NDArray[np.float_]] = None,
        link_lengths: Optional[Dict[Self, Optional[float]]] = None,
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
        if props is None:
            props = ParticleProperties()
        super().__init__(
            x=x,
            y=y,
            radius=radius,
            color=color,
            props=props,
            velocity=velocity,
            link_lengths=link_lengths,
        )

    def _compute_delta_velocity(
        self, sim_data: SimulationData, force: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        forces = [force, sim_data.wind_force * self.radius]
        acceleration = np.sum(forces, axis=0) / self.props.mass + sim_data.g_vector

        return (
            np.clip(acceleration, -2, 2)
            + np.random.uniform(-1, 1, 2) * sim_data.temperature
        )

    def update(
        self, sim_data: SimulationData, force: Optional[npt.NDArray[np.float_]] = None
    ) -> None:
        if self.mouse:
            self.velocity = sim_data.delta_mouse_pos
            dx, dy = self.velocity
            self.x += dx
            self.y += dy
        elif force is not None and not self.props.locked:
            self.velocity += self._compute_delta_velocity(sim_data, force)
            self.velocity *= sim_data.air_res_calc
            dx, dy = self.velocity * sim_data.speed
            self.x += dx
            self.y += dy

        if sim_data.right and self.x_max >= sim_data.width:
            self.velocity *= [-self.props.bounciness, 1 - sim_data.ground_friction]
            self.x = sim_data.width - self.radius
        if sim_data.left and self.x_min <= 0:
            self.velocity *= [-self.props.bounciness, 1 - sim_data.ground_friction]
            self.x = self.radius
        if sim_data.bottom and self.y_max >= sim_data.height:
            self.velocity *= [1 - sim_data.ground_friction, -self.props.bounciness]
            self.y = sim_data.height - self.radius
        if sim_data.top and self.y_min <= 0:
            self.velocity *= [1 - sim_data.ground_friction, -self.props.bounciness]
            self.y = self.radius

        self._collisions = {}
