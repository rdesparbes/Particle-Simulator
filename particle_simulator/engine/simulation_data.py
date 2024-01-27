from dataclasses import field, dataclass
from typing import Tuple

import numpy as np
from numpy import typing as npt


@dataclass(kw_only=True)
class SimulationData:
    # Data that must be saved and loaded
    stress_visualization: bool = False
    bg_color: Tuple[int, int, int] = (255, 255, 255)
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
    code: str = 'print("Hello World")'

    @property
    def g_vector(self) -> npt.NDArray[np.float_]:
        return self.g * self.g_dir

    @property
    def air_res_calc(self) -> float:
        return (1.0 - self.air_res) ** self.speed

    def set_code(self, code: str) -> None:
        self.code = code
