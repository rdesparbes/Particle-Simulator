from dataclasses import field, dataclass
from typing import Tuple

import numpy as np
from numpy import typing as npt

from particle_simulator.geometry import Rectangle


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

    # Data needed by particles
    height: int = 600
    width: int = 650
    paused: bool = True
    mx: int = 0
    my: int = 0
    prev_mx: int = 0
    prev_my: int = 0

    # Not needed by particles, but closely related to mx/my:
    mr: float = 5.0

    @property
    def g_vector(self) -> npt.NDArray[np.float_]:
        return self.g * self.g_dir

    @property
    def air_res_calc(self) -> float:
        return (1.0 - self.air_res) ** self.speed

    def set_code(self, code: str) -> None:
        self.code = code

    @property
    def delta_mouse_pos(self) -> npt.NDArray[np.float_]:
        return np.subtract([self.mx, self.my], [self.prev_mx, self.prev_my]).astype(
            np.float_
        )

    @property
    def rectangle(self) -> Rectangle:
        return Rectangle(x_min=0, y_min=0, x_max=self.width, y_max=self.height)
