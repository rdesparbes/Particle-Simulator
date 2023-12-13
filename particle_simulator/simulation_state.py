from dataclasses import dataclass, field
from typing import (
    Tuple,
    List,
    Dict,
)

import numpy as np
from numpy import typing as npt

from particle_simulator.particle import Particle


@dataclass
class SimulationState:
    width: int = 650
    height: int = 600
    temperature: float = 0.0
    g: float = 0.1
    g_dir: npt.NDArray[np.float_] = field(default_factory=lambda: np.array([0.0, 1.0]))
    wind_force: npt.NDArray[np.float_] = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )
    air_res: float = 0.05
    ground_friction: float = 0.0
    speed: float = 1.0
    use_grid: bool = True
    calculate_radii_diff: bool = False
    top: bool = True
    bottom: bool = True
    left: bool = True
    right: bool = True
    void_edges: bool = False
    bg_color: Tuple[Tuple[int, int, int], str] = ((255, 255, 255), "#ffffff")
    stress_visualization: bool = False
    code: str = 'print("Hello World")'
    particles: List[Particle] = field(default_factory=list)
    groups: Dict[str, List[Particle]] = field(default_factory=lambda: {"group1": []})

    @property
    def g_vector(self) -> npt.NDArray[np.float_]:
        return self.g * self.g_dir

    @property
    def air_res_calc(self) -> float:
        return (1 - self.air_res) ** self.speed
