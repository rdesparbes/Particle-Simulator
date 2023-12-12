import math
import time
from typing import (
    Tuple,
    Optional,
    List,
    Dict,
    Any,
    Collection,
    Iterable,
    Literal,
    Union,
)

import numpy as np
from numpy import typing as npt

from particle_simulator.error import Error
from particle_simulator.grid import Grid
from particle_simulator.particle import Link, Particle


class SimulationState:
    def __init__(
        self,
        width: int = 650,
        height: int = 600,
        gridres: Tuple[int, int] = (50, 50),
        temperature: float = 0,
        g: float = 0.1,
        air_res: float = 0.05,
        ground_friction: float = 0,
        fps_update_delay: float = 0.5,
    ):
        self.width = width
        self.height = height

        self.temperature = temperature
        self.g = g  # gravity
        self.g_dir: npt.NDArray[np.float_] = np.array([0.0, 1.0])
        self.g_vector: npt.NDArray[np.float_] = np.array([0.0, -g])
        self.wind_force: npt.NDArray[np.float_] = np.array([0.0, 0.0])
        self.air_res = air_res
        self.air_res_calc = 1.0 - self.air_res
        self.ground_friction = ground_friction
        self.speed: float = 1.0

        self.fps = 0
        self.fps_update_delay = fps_update_delay
        self.mx, self.my = 0, 0
        self.prev_mx, self.prev_my = 0, 0
        self.rotate_mode = False
        self.min_spawn_delay = 0.05
        self.min_hold_delay = 1
        self.last_mouse_time = 0
        self.mr = 5
        self.mouse_down = False
        self.mouse_down_start: Optional[float] = None
        self.shift = False
        self.start_save = False
        self.start_load = False
        self.paused = True
        self.toggle_pause = False
        self.running = True
        self.focus = True
        self.error: Optional[Error] = None
        self.use_grid = True
        self.calculate_radii_diff = False

        self.top = True
        self.bottom = True
        self.left = True
        self.right = True
        self.void_edges = False

        self.bg_color = [255, 255, 255], "#ffffff"
        self.stress_visualization = False
        self.link_colors: List[Link] = []

        self.code: str = 'print("Hello World")'
        self.grid = Grid(*gridres, height=height, width=width)

        self.start_time = time.time()
        self.prev_time = self.start_time

        self.particles: List[Particle] = []
        self.selection: List[Particle] = []
        self.clipboard: List[Dict[str, Any]] = []
        self.pasting = False
        self.groups: Dict[str, List[Particle]] = {"group1": []}

    @staticmethod
    def _rotate_2d(
        x: float, y: float, cx: float, cy: float, angle: float
    ) -> Tuple[float, float]:
        angle_rad = -np.radians(angle)
        dist_x = x - cx
        dist_y = y - cy
        current_angle = math.atan2(dist_y, dist_x)
        angle_rad += current_angle
        radius = np.sqrt(dist_x**2 + dist_y**2)
        x = cx + radius * np.cos(angle_rad)
        y = cy + radius * np.sin(angle_rad)

        return x, y

    def toggle_paused(self) -> None:
        self.toggle_pause = True

    def _select_particle(self, particle: Particle) -> None:
        if particle in self.selection:
            return
        self.selection.append(particle)

    def remove_particle(self, particle: Particle) -> None:
        self.particles.remove(particle)
        if particle in self.selection:
            self.selection.remove(particle)
        for p in particle.link_lengths:
            del p.link_lengths[particle]
        self.groups[particle.group].remove(particle)
        del particle

    def _copy_selected(self) -> None:
        self.clipboard = []
        for p in self.selection:
            dictionary = p.return_dict(index_source=self.selection)
            dictionary["x"] -= self.mx
            dictionary["y"] -= self.my
            self.clipboard.append(dictionary)

    def _cut(self) -> None:
        self._copy_selected()
        temp = self.selection.copy()
        for p in temp:
            self.remove_particle(p)

    def link_selection(self, fit_link: bool = False) -> None:
        self.link(self.selection, fit_link=fit_link)
        self.selection = []

    def unlink_selection(self) -> None:
        self.unlink(self.selection)
        self.selection = []

    @staticmethod
    def link(
        particles: List[Particle],
        fit_link: bool = False,
        distance: Union[None, float, Literal["repel"]] = None,
    ) -> None:
        Particle.link(particles, fit_link, distance)

    @staticmethod
    def unlink(particles: Collection[Particle]) -> None:
        Particle.unlink(particles)

    def change_link_lengths(self, particles: Iterable[Particle], amount: float) -> None:
        for p in particles:
            for link, value in p.link_lengths.items():
                if value != "repel":
                    self.link([p, link], fit_link=True, distance=value + amount)

    def set_code(self, code) -> None:
        self.code = code

    def execute(self, code: str) -> None:
        try:
            exec(code)
        except Exception as error:
            self.error = Error("Code-Error", error)

    def _simulate_step(self):
        self.link_colors = []
        self.g_vector = self.g_dir * self.g
        self.air_res_calc = (1 - self.air_res) ** self.speed
        if self.use_grid:
            self.grid.init_grid(self.particles)
        if self.toggle_pause:
            self.paused = not self.paused

            if not self.paused:
                self.selection = []
            self.toggle_pause = False
        for particle in self.particles:
            if not particle.interacts:
                near_particles = []
            elif particle.interacts_will_all:
                near_particles = self.particles
            elif self.use_grid:
                near_particles = self.grid.return_particles(particle)
            else:
                near_particles = self.particles
            particle.update(near_particles)
