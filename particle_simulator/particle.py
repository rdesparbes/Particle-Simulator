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
