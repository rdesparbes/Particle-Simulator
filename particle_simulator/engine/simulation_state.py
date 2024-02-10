import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import (
    Tuple,
    List,
    Dict,
    Collection,
    Iterable,
    Optional,
    Deque,
    Mapping,
)

import numpy as np
import numpy.typing as npt

from particle_simulator.engine.conversion import (
    builders_to_particles,
    particles_to_builders,
)
from particle_simulator.engine.error import Error
from particle_simulator.engine.event import event, eventclass
from particle_simulator.engine.geometry import Rectangle, Circle
from particle_simulator.engine.grid import Grid
from particle_simulator.engine.interaction_transformer import (
    InteractionTransformer,
    apply_collisions,
    remove_broken_links,
)
from particle_simulator.engine.particle import (
    Particle,
    link_particles,
    unlink_particles,
    ComputeMagnitudeStrategy,
    radii_compute_magnitude_strategy,
    default_compute_magnitude_strategy,
)
from particle_simulator.engine.particle_factory import ParticleBuilder, ParticleFactory
from particle_simulator.engine.particle_interaction import ParticleInteraction
from particle_simulator.engine.simulation_data import SimulationData
from particle_simulator.mouse_mode import Mode


@eventclass
@dataclass(kw_only=True)
class SimulationState(SimulationData):
    # Mutable collections:
    particles: List[Particle] = field(default_factory=list)
    groups: Dict[str, List[Particle]] = field(default_factory=dict)
    selection: List[Particle] = field(default_factory=list)
    errors: Deque[Error] = field(default_factory=deque, init=False)
    _clipboard: List[ParticleBuilder] = field(default_factory=list, init=False)
    # Geometry:
    height: int = 600
    width: int = 650
    grid_res_x: int = 50
    grid_res_y: int = 50
    # States:
    paused: bool = True
    running: bool = True
    pasting: bool = True
    _toggle_pause: bool = False
    # Display:
    show_fps: bool = True
    show_num: bool = True
    show_links: bool = True
    # Mouse:
    mx: int = 0
    my: int = 0
    mr: float = 5.0
    _prev_mx: int = field(default=0, init=False)
    _prev_my: int = field(default=0, init=False)
    _mouse_mode: Mode = field(default="MOVE", init=False)
    min_spawn_delay: float = 0.05

    @property
    def mouse_mode(self) -> Mode:
        return self._mouse_mode

    @mouse_mode.setter
    def mouse_mode(self, value: Mode) -> None:
        self._mouse_mode = value
        self.on_mouse_mode_changed()

    @property
    def _delta_mouse_pos(self) -> npt.NDArray[np.float_]:
        return np.subtract([self.mx, self.my], [self._prev_mx, self._prev_my]).astype(
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

    def rotate_selection(self, factor: float) -> None:
        for p in self.selection:
            p.rotate(self.mx, self.my, factor * self.mr)

    def update_mouse_radius(self, factor: float) -> None:
        self.mr = max(self.mr * 2**factor, 1)

    def copy_selection(self) -> None:
        self._clipboard = []
        for factory in particles_to_builders(self.selection):
            factory.x -= self.mx
            factory.y -= self.my
            self._clipboard.append(factory)

    def cut_selection(self) -> None:
        self.copy_selection()
        self.remove_selection()

    def paste(self) -> None:
        self.pasting = True
        particles: List[Particle] = []
        for particle in builders_to_particles(self._clipboard):
            particle.x += self.mx
            particle.y += self.my
            particle.mouse = True
            self.register_particle(particle)
        self.selection = particles

    def toggle_paused(self) -> None:
        self._toggle_pause = True

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

    def _iter_in_range(self, circle: Circle) -> Iterable[Particle]:
        for particle in self.particles:
            if particle.circle.is_in_range(circle):
                yield particle

    def select_in_range(self) -> None:
        for p in self._iter_in_range(self._cursor_circle):
            self.select_particle(p)

    def remove_in_range(self) -> None:
        temp = self.particles.copy()
        for p in temp:
            if p.circle.is_in_range(self._cursor_circle):
                self.remove_particle(p)

    @property
    def _cursor_circle(self) -> Circle:
        return Circle(self.mx, self.my, self.mr)

    def select_or_reset_in_range(self) -> None:
        any_in_range = False
        for p in self._iter_in_range(self._cursor_circle):
            self.select_particle(p)
            any_in_range = True
        if not any_in_range:
            self.selection = []

    def move_in_range(self) -> None:
        any_selected = False
        for p in self._iter_in_range(self._cursor_circle):
            p.mouse = True
            if p in self.selection:
                any_selected = True
        if any_selected:
            for particle in self.selection:
                particle.mouse = True

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

    def create_particle(self, factory: ParticleFactory) -> None:
        particle = Particle(
            x=self.mx,
            y=self.my,
            radius=factory.radius,
            color=factory.color,
            props=factory.props,
            velocity=np.array(factory.velocity),
        )
        self.register_particle(particle)

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

    @event
    def on_group_created(self, name: str) -> str:
        return name

    def _get_group(self, name: str) -> List[Particle]:
        try:
            return self.groups[name]
        except KeyError:
            new_group: List[Particle] = []
            self.groups[name] = new_group
            self.on_group_created(name)
            return new_group

    def register_particle(self, particle: Particle) -> None:
        self._get_group(particle.props.group).append(particle)
        self.particles.append(particle)

    @property
    def _rectangle(self) -> Rectangle:
        return Rectangle(x_min=0, y_min=0, x_max=self.width, y_max=self.height)

    def _is_out_of_bounds(self, rectangle: Rectangle) -> bool:
        return self.void_edges and self._rectangle.isdisjoint(rectangle)

    def _compute_acceleration(
        self,
        particle: Particle,
        force: npt.NDArray[np.float_],
        max_acceleration_magnitude: float = 2.0,
    ) -> npt.NDArray[np.float_]:
        forces = [force, self.wind_force * particle.radius]
        acceleration = np.sum(forces, axis=0) / particle.props.mass + self.g_vector
        acc_magnitude = math.hypot(*acceleration)
        if acc_magnitude > 0.0:
            clipped_magnitude = min(acc_magnitude, max_acceleration_magnitude)
            acceleration = acceleration / acc_magnitude * clipped_magnitude
        return acceleration + np.random.normal(scale=0.75, size=2) * self.temperature

    def update_mouse_pos(self, new_mouse_pos: Tuple[int, int]) -> None:
        self._prev_mx, self._prev_my = self.mx, self.my
        self.mx, self.my = new_mouse_pos

    @staticmethod
    def _update_interactions(
        interactions: Dict[Particle, Dict[Particle, ParticleInteraction]],
        particle: Particle,
        near_particles: Iterable[Particle],
        compute_magnitude_strategy: ComputeMagnitudeStrategy,
    ) -> None:
        for near_particle in near_particles:
            if particle.props.locked or near_particle in interactions[particle]:
                continue
            interaction = particle.compute_interaction(
                near_particle, compute_magnitude_strategy
            )
            if interaction is None:
                continue
            interactions[particle][near_particle] = interaction
            interactions[near_particle][particle] = -interaction

    def _update(self, particle: Particle) -> None:
        if particle.mouse:
            particle.velocity = self._delta_mouse_pos
            dx, dy = particle.velocity
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

    def _compute_interactions(
        self,
    ) -> Dict[Particle, Dict[Particle, ParticleInteraction]]:
        if self.calculate_radii_diff:
            compute_magnitude_strategy: ComputeMagnitudeStrategy = (
                radii_compute_magnitude_strategy
            )
        else:
            compute_magnitude_strategy = default_compute_magnitude_strategy
        grid: Optional[Grid] = None
        if self.use_grid:
            grid = Grid(
                self.grid_res_x,
                self.grid_res_y,
                height=self.height,
                width=self.width,
            )
            grid.extend(self.particles)
        interactions: Dict[Particle, Dict[Particle, ParticleInteraction]] = defaultdict(
            dict
        )
        for particle in self.particles:
            if not particle.interacts:
                near_particles: Iterable[Particle] = []
            elif particle.interacts_with_all:
                near_particles = self.particles
            elif grid is not None:
                near_particles = grid.return_particles(particle)
            else:
                near_particles = self.particles
            self._update_interactions(
                interactions, particle, near_particles, compute_magnitude_strategy
            )
        return interactions

    def _apply_forces(
        self, particle: Particle, interactions: Mapping[Particle, ParticleInteraction]
    ) -> None:
        if particle.mouse or particle.props.locked:
            return
        force: npt.NDArray[np.float_] = sum(
            (interaction.force for interaction in interactions.values()),
            start=np.zeros(2),
        )
        particle.velocity += self._compute_acceleration(particle, force)
        particle.velocity *= self.air_res_calc
        dx, dy = particle.velocity * self.speed
        particle.x += dx
        particle.y += dy

    @event
    def on_pause_toggle(self) -> bool:
        self.paused = not self.paused
        if not self.paused:
            self.selection = []
        self._toggle_pause = False
        return self.paused

    @event
    def on_mouse_mode_changed(self) -> Mode:
        return self._mouse_mode

    def simulate_step(
        self, additional_transformers: Iterable[InteractionTransformer] = ()
    ) -> None:
        if self._toggle_pause:
            self.on_pause_toggle()
        if not self.paused:
            transformers: List[InteractionTransformer] = [
                apply_collisions,
                remove_broken_links,
                self._apply_forces,
            ]
            transformers.extend(additional_transformers)
            inter_dict = self._compute_interactions()
            for transformer in transformers:
                for particle, interactions in inter_dict.items():
                    transformer(particle, interactions)
        for particle in self.particles:
            self._update(particle)
            if self._is_out_of_bounds(particle.rectangle):
                self.remove_particle(particle)
