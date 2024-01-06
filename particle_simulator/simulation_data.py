from dataclasses import field, dataclass
from typing import Tuple

import numpy as np
from numpy import typing as npt


@dataclass(kw_only=True)
class SimulationData:
    stress_visualization: bool = False
    bg_color: Tuple[Tuple[int, int, int], str] = ((255, 255, 255), "#ffffff")
    void_edges: bool = False
    right: bool = True
    left: bool = True
    bottom: bool = True
    top: bool = True
    calculate_radii_diff: bool = False
    use_grid: bool = True
    speed: float = 1.0
    ground_friction: float = 0.0
    air_res: float = 0.05
    wind_force: npt.NDArray[np.float_] = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )
    g_dir: npt.NDArray[np.float_] = field(default_factory=lambda: np.array([0.0, 1.0]))
    g: float = 0.1
    temperature: float = 0.0
    height: int = 600
    width: int = 650

    paused: bool = True
    mr: float = 5.0
    mx: int = 0
    my: int = 0
    prev_mx: int = 0
    prev_my: int = 0

    @property
    def g_vector(self) -> npt.NDArray[np.float_]:
        return self.g * self.g_dir

    @property
    def air_res_calc(self) -> float:
        return (1 - self.air_res) ** self.speed
