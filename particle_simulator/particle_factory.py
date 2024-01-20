from dataclasses import dataclass
from typing import Tuple, Optional, Dict

from particle_simulator.particle_properties import ParticleProperties


@dataclass(kw_only=True)
class ParticleFactory:
    color: Tuple[int, int, int]
    props: ParticleProperties
    velocity: Tuple[float, float]
    radius: float = 4.0


@dataclass(kw_only=True)
class ParticleBuilder(ParticleFactory):
    x: float
    y: float
    radius: float
    link_indices_lengths: Dict[int, Optional[float]]
