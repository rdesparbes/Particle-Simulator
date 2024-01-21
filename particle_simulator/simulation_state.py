from collections import deque
from dataclasses import dataclass, field
from typing import (
    Tuple,
    List,
    Dict,
    Literal,
    Collection,
    Iterable,
    Optional,
    Callable,
    NamedTuple,
    Deque,
)

import numpy as np
import numpy.typing as npt

from particle_simulator.conversion import builders_to_particles, particles_to_builders
from particle_simulator.error import Error
from particle_simulator.geometry import Rectangle
from particle_simulator.grid import Grid
from particle_simulator.particle import (
    Particle,
    link_particles,
    unlink_particles,
    ComputeMagnitudeStrategy,
    radii_compute_magnitude_strategy,
    default_compute_magnitude_strategy,
)
from particle_simulator.particle_factory import ParticleBuilder
from particle_simulator.simulation_data import SimulationData

Mode = Literal["SELECT", "MOVE", "ADD"]


class Link(NamedTuple):
    particle_a: Particle
    particle_b: Particle
    percentage: float


@dataclass(kw_only=True)
class SimulationState(SimulationData):
    # Mutable collections:
    particles: List[Particle] = field(default_factory=list)
    groups: Dict[str, List[Particle]] = field(default_factory=lambda: {"group1": []})
    selection: List[Particle] = field(default_factory=list)
    clipboard: List[ParticleBuilder] = field(default_factory=list)
    errors: Deque[Error] = field(default_factory=deque)
    create_group_callbacks: List[Callable[[str], None]] = field(default_factory=list)
    # Geometry:
    height: int = 600
    width: int = 650
    grid_res_x: int = 50
    grid_res_y: int = 50
    # States:
    paused: bool = True
    toggle_pause: bool = False
    running: bool = True
    pasting: bool = True
    # Display:
    show_fps: bool = True
    show_num: bool = True
    show_links: bool = True
    # Mouse:
    mx: int = 0
    my: int = 0
    mr: float = 5.0
    prev_mx: int = 0
    prev_my: int = 0
    mouse_mode: Mode = "MOVE"
    min_spawn_delay: float = 0.05

    @property
    def delta_mouse_pos(self) -> npt.NDArray[np.float_]:
        return np.subtract([self.mx, self.my], [self.prev_mx, self.prev_my]).astype(
            np.float_
        )

    @staticmethod
    def _link(
        particles: List[Particle],
        fit_link: bool = False,
        distance: Optional[float] = None,
    ) -> None:
        link_particles(particles, fit_link, distance)

    @staticmethod
    def _unlink(particles: Collection[Particle]) -> None:
        unlink_particles(particles)

    def rotate_selection(self, x: float, y: float, angle: float) -> None:
        for p in self.selection:
            p.rotate(x, y, angle=angle)

    def copy_selection(self) -> None:
        self.clipboard = []
        for factory in particles_to_builders(self.selection):
            factory.x -= self.mx
            factory.y -= self.my
            self.clipboard.append(factory)

    def cut_selection(self) -> None:
        self.copy_selection()
        self.remove_selection()

    def paste(self) -> None:
        self.pasting = True
        particles: List[Particle] = []
        for particle in builders_to_particles(self.clipboard):
            particle.x += self.mx
            particle.y += self.my
            particle.mouse = True
            self.register_particle(particle)
        self.selection = particles

    def toggle_paused(self) -> None:
        self.toggle_pause = True

    def link_selection(self, fit_link: bool = False) -> None:
        self._link(self.selection, fit_link=fit_link)
        self.selection = []

    def unlink_selection(self) -> None:
        self._unlink(self.selection)
        self.selection = []

    def select_particle(self, particle: Particle) -> None:
        if particle in self.selection:
            return
        self.selection.append(particle)

    def remove_particle(self, particle: Particle) -> None:
        self.particles.remove(particle)
        try:
            self.selection.remove(particle)
        except ValueError:
            pass
        for p in particle.link_lengths:
            del p.link_lengths[particle]
        self.groups[particle.props.group].remove(particle)
        del particle

    def remove_selection(self) -> None:
        temp = self.selection.copy()
        for p in temp:
            self.remove_particle(p)

    def select_all(self) -> None:
        for p in self.particles:
            self.select_particle(p)

    def lock_selection(self) -> None:
        for p in self.selection:
            p.props.locked = True

    def unlock_selection(self) -> None:
        for p in self.selection:
            p.props.locked = False

    def change_link_lengths(self, particles: Iterable[Particle], amount: float) -> None:
        for p in particles:
            for link, length in p.link_lengths.items():
                if length is not None:
                    self._link([p, link], fit_link=True, distance=length + amount)

    def execute(self, code: str) -> None:
        try:
            exec(code)
        except Exception as error:
            self.errors.append(Error("Code-Error", error))

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
            new_group: List[Particle] = []
            self.groups[name] = new_group
            for create_group_callback in self.create_group_callbacks:
                create_group_callback(name)
            return new_group

    def register_particle(self, particle: Particle) -> None:
        self._get_group(particle.props.group).append(particle)
        self.particles.append(particle)

    @property
    def rectangle(self) -> Rectangle:
        return Rectangle(x_min=0, y_min=0, x_max=self.width, y_max=self.height)

    def _is_out_of_bounds(self, rectangle: Rectangle) -> bool:
        return self.void_edges and self.rectangle.isdisjoint(rectangle)

    def _compute_delta_velocity(
        self, particle: Particle, force: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        forces = [force, self.wind_force * particle.radius]
        acceleration = np.sum(forces, axis=0) / particle.props.mass + self.g_vector

        return (
            np.clip(acceleration, -2, 2)
            + np.random.uniform(-1, 1, 2) * self.temperature
        )

    def update_mouse_pos(self, new_mouse_pos: Tuple[int, int]) -> None:
        self.prev_mx, self.prev_my = self.mx, self.my
        self.mx, self.my = new_mouse_pos

    def _compute_force(
        self, particle: Particle, near_particles: Iterable[Particle], links: List[Link]
    ) -> npt.NDArray[np.float_]:
        force = np.zeros(2)
        if self.calculate_radii_diff:
            compute_magnitude_strategy: ComputeMagnitudeStrategy = (
                radii_compute_magnitude_strategy
            )
        else:
            compute_magnitude_strategy = default_compute_magnitude_strategy
        for near_particle, interaction in particle.iter_interactions(
            near_particles, compute_magnitude_strategy
        ):
            force += interaction.force
            particle.compute_collision(near_particle)
            near_particle._collisions[particle] = -interaction.force
            if interaction.link_percentage is not None:
                if interaction.link_percentage > 1.0:
                    unlink_particles([particle, near_particle])
                elif self.stress_visualization:
                    links.append(
                        Link(particle, near_particle, interaction.link_percentage)
                    )

        force += np.sum(list(particle._collisions.values()), axis=0)
        particle._collisions = {}
        return force

    def _update(
        self, particle: Particle, force: Optional[npt.NDArray[np.float_]] = None
    ) -> None:
        if particle.mouse:
            particle.velocity = self.delta_mouse_pos
            dx, dy = particle.velocity
            particle.x += dx
            particle.y += dy
        elif force is not None and not particle.props.locked:
            particle.velocity += self._compute_delta_velocity(particle, force)
            particle.velocity *= self.air_res_calc
            dx, dy = particle.velocity * self.speed
            particle.x += dx
            particle.y += dy

        if self.right and particle.x_max >= self.width:
            particle.velocity *= [-particle.props.bounciness, 1 - self.ground_friction]
            particle.x = self.width - particle.radius
        if self.left and particle.x_min <= 0:
            particle.velocity *= [-particle.props.bounciness, 1 - self.ground_friction]
            particle.x = particle.radius
        if self.bottom and particle.y_max >= self.height:
            particle.velocity *= [1 - self.ground_friction, -particle.props.bounciness]
            particle.y = self.height - particle.radius
        if self.top and particle.y_min <= 0:
            particle.velocity *= [1 - self.ground_friction, -particle.props.bounciness]
            particle.y = particle.radius

    def simulate_step(self) -> List[Link]:
        links: List[Link] = []
        grid: Optional[Grid] = None
        if self.use_grid:
            grid = Grid(
                self.grid_res_x,
                self.grid_res_y,
                height=self.height,
                width=self.width,
            )
            grid.extend(self.particles)
        if self.toggle_pause:
            self.paused = not self.paused

            if not self.paused:
                self.selection = []
            self.toggle_pause = False
        for particle in self.particles:
            if not particle.interacts:
                near_particles: Iterable[Particle] = []
            elif particle.interacts_with_all:
                near_particles = self.particles
            elif grid is not None:
                near_particles = grid.return_particles(particle)
            else:
                near_particles = self.particles
            if self.paused:
                force: Optional[npt.NDArray[np.float_]] = None
            else:
                force = self._compute_force(particle, near_particles, links)
            self._update(particle, force)
            if self._is_out_of_bounds(particle.rectangle):
                self.remove_particle(particle)
        return links
