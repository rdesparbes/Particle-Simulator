import time
from dataclasses import asdict
from functools import partial
from typing import (
    Optional,
    Iterable,
    List,
    Protocol,
    Tuple,
)

import numpy as np
import numpy.typing as npt

from particle_simulator.engine.conversion import (
    builders_to_particles,
    particles_to_builders,
)
from particle_simulator.engine.error import Error
from particle_simulator.engine.interaction_transformer import (
    compute_links,
    InteractionTransformer,
    Link,
)
from particle_simulator.engine.particle import Particle
from particle_simulator.engine.particle_factory import ParticleFactory
from particle_simulator.engine.simulation_state import SimulationState
from .controller_state import ControllerState
from .io import sim_pickle
from .painter import paint_image
from .sim_gui_settings import SimGUISettings


class SimulationGUI(Protocol):
    def register_sim(self, sim: SimulationState) -> None:
        pass

    def get_sim_settings(self) -> SimGUISettings:
        pass

    def set_sim_settings(self, sim_settings: SimGUISettings) -> None:
        pass

    def get_particle_factory(self) -> ParticleFactory:
        pass

    def set_particle_factory(self, particle_factory: ParticleFactory) -> None:
        pass

    def update(
        self,
        image: npt.NDArray[np.uint8],
        fps: Optional[float] = None,
    ) -> None:
        pass

    def get_mouse_pos(self) -> Tuple[int, int]:
        pass


class Simulation:
    def __init__(
        self,
        state: SimulationState,
        gui: SimulationGUI,
        fps_update_delay: float = 0.5,
    ):
        self.state: SimulationState = state
        self.gui = gui
        self.fps = 0.0
        self.fps_update_delay = fps_update_delay
        self.rotate_mode = False
        self.last_particle_added_time = 0.0
        self.prev_fps_update_time = time.time()
        self.prev_time = self.prev_fps_update_time

    def mouse_button_1_pressed(self) -> None:
        if self.state.mouse_mode == "SELECT":
            self.state.select_or_reset_in_range()
        elif self.state.mouse_mode == "MOVE":
            self.state.move_in_range()
        elif self.state.mouse_mode == "ADD":
            self.state.selection = []
            self.add_particle()

    def mouse_button_1_pressed_while_moving(self) -> None:
        if self.state.mouse_mode == "SELECT":
            self.state.select_in_range()
        elif (
            self.state.mouse_mode == "ADD"
            and time.time() - self.last_particle_added_time
            >= self.state.min_spawn_delay
        ):
            self.add_particle()

    def mouse_button_1_released(self) -> None:
        if self.state.mouse_mode == "MOVE" or self.state.pasting:
            for p in self.state.particles:
                p.mouse = False
        self.state.pasting = False

    def on_scroll(self, factor: float) -> None:
        if self.rotate_mode:
            self.state.rotate_selection(factor)
        else:
            self.state.update_mouse_radius(factor)

    def enter_rotate_mode(self) -> None:
        self.rotate_mode = True

    def exit_rotate_mode(self) -> None:
        self.rotate_mode = False

    def _get_particle_factory(self) -> Optional[ParticleFactory]:
        try:
            return self.gui.get_particle_factory()
        except Exception as error:
            self.state.errors.append(Error("Input-Error", error))
        return None

    def _set_particles(self, particles: Iterable[Particle]) -> None:
        for p in particles:
            # Update for each particle in case of 'random' and avoid sharing properties:
            factory = self._get_particle_factory()
            if factory is not None:
                p.radius = factory.radius
                p.color = factory.color
                p.props = factory.props
                p.velocity = np.array(factory.velocity)

    def set_selected(self) -> None:
        self._set_particles(self.state.selection)

    def set_all(self) -> None:
        self._set_particles(self.state.particles)

    def to_controller_state(self) -> ControllerState:
        return ControllerState(
            sim_data=self.state,
            gui_settings=self.gui.get_sim_settings(),
            gui_particle_state=self.gui.get_particle_factory(),
            particles=particles_to_builders(self.state.particles),
        )

    def from_controller_state(self, controller_state: ControllerState) -> None:
        self.state = SimulationState(**asdict(controller_state.sim_data))

        self.gui.register_sim(self.state)
        self.gui.set_particle_factory(controller_state.gui_particle_state)
        self.gui.set_sim_settings(controller_state.gui_settings)

        for particle in builders_to_particles(controller_state.particles):
            self.state.register_particle(particle)

    def add_particle(self) -> None:
        factory = self._get_particle_factory()
        if factory is not None:
            self.state.create_particle(factory)
            self.last_particle_added_time = time.time()

    def _paint_image(self, link_colors: Iterable[Link]) -> npt.NDArray[np.uint8]:
        if not self.state.stress_visualization or self.state.paused:
            return paint_image(self.state, link_colors=None)
        return paint_image(self.state, link_colors)

    def save(self, filename: str) -> None:
        controller_state = self.to_controller_state()
        try:
            sim_pickle.dump(controller_state, filename=filename)
        except Exception as error:
            self.state.errors.append(Error("Saving-Error", error))

    def load(self, filename: str) -> None:
        if not self.state.paused:
            self.state.toggle_paused()
        try:
            controller_state = sim_pickle.load(filename)
            self.from_controller_state(controller_state)
        except Exception as error:
            self.state.errors.append(Error("Loading-Error", error))

    @property
    def _fps_update_time(self) -> float:
        return self.prev_fps_update_time + self.fps_update_delay

    def _update_timings(self, new_time: float) -> None:
        if new_time >= self._fps_update_time:
            try:
                self.fps = 1.0 / (new_time - self.prev_time)
            except ZeroDivisionError:
                pass
            self.prev_fps_update_time = new_time
        self.prev_time = new_time

    def simulate(self) -> None:
        while self.state.running:
            self.state.update_mouse_pos(self.gui.get_mouse_pos())
            links: List[Link] = []
            transformers: List[InteractionTransformer] = []
            if self.state.stress_visualization:
                transformers.append(partial(compute_links, links=links))
            self.state.simulate_step(transformers)
            self._update_timings(new_time=time.time())
            image = self._paint_image(links)
            self.gui.update(image, self.fps)
