import math
from dataclasses import dataclass, field, asdict
from typing import (
    Tuple,
    List,
    Dict,
    Literal,
    Union,
    Collection,
    Iterable,
    Optional,
    Callable,
)

import numpy as np
from numpy import typing as npt

from particle_simulator.error import Error
from particle_simulator.particle import Particle
from particle_simulator.particle_state import ParticleState


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

    paused: bool = True
    toggle_pause: bool = False
    selection: List[Particle] = field(default_factory=list)
    error: Optional[Error] = None
    show_fps: bool = True
    show_num: bool = True
    show_links: bool = True
    add_group_callbacks: List[Callable[[str], None]] = field(default_factory=list)
    grid_res_x: int = 50
    grid_res_y: int = 50

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

    @property
    def g_vector(self) -> npt.NDArray[np.float_]:
        return self.g * self.g_dir

    @property
    def air_res_calc(self) -> float:
        return (1 - self.air_res) ** self.speed

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

    def link_selection(self, fit_link: bool = False) -> None:
        self.link(self.selection, fit_link=fit_link)
        self.selection = []

    def unlink_selection(self) -> None:
        self.unlink(self.selection)
        self.selection = []

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

    def add_group(self) -> str:
        for i in range(1, len(self.groups) + 2):
            name = f"group{i}"
            if name not in self.groups:
                self.groups[name] = []
                return name
        assert False  # Unreachable (pigeonhole principle)

    def select_group(self, name: str) -> None:
        self.selection = list(self.groups[name])

    def _get_group(self, name: str) -> List[Particle]:
        try:
            return self.groups[name]
        except KeyError:
            new_group = []
            self.groups[name] = new_group
            for callback in self.add_group_callbacks:
                callback(name)
            return new_group

    def register_particle(self, particle: Particle) -> None:
        self._get_group(particle.group).append(particle)
        self.particles.append(particle)

    def _replace_particle(
        self, p: Particle, particle_settings: ParticleState
    ) -> Particle:
        temp_link_lengths = p.link_lengths.copy()
        px, py = p.x, p.y
        self.remove_particle(p)
        p = Particle(self, px, py, **asdict(particle_settings))
        self.register_particle(p)
        for link, length in temp_link_lengths.items():
            self.link([link, p], fit_link=length != "repel", distance=length)
        return p
