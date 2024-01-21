import time
import tkinter as tk
from dataclasses import asdict
from functools import partial
from typing import (
    Optional,
    Tuple,
    Iterable,
    Callable,
)

import numpy as np
import numpy.typing as npt

from .controller_state import ControllerState
from .conversion import builders_to_particles, particles_to_builders
from .error import Error
from .geometry import Circle
from .gui import GUI
from .painter import paint_image
from .particle import Particle
from .particle_factory import ParticleFactory
from .simulation_state import SimulationState, Link


def _no_event(action: Callable[[], None]) -> Callable[[tk.Event], None]:
    def wrapper(_event: tk.Event) -> None:
        return action()

    return wrapper


class Simulation:
    def __init__(
        self,
        width: int = 650,
        height: int = 600,
        title: str = "Simulation",
        gridres: Tuple[int, int] = (50, 50),
        temperature: float = 0,
        g: float = 0.1,
        air_res: float = 0.05,
        ground_friction: float = 0,
        fps_update_delay: float = 0.5,
    ):
        self.state: SimulationState = SimulationState(
            width=width,
            height=height,
            temperature=temperature,
            g=g,
            air_res=air_res,
            ground_friction=ground_friction,
            grid_res_x=gridres[0],
            grid_res_y=gridres[1],
        )

        self.fps = 0.0
        self.fps_update_delay = fps_update_delay
        self.rotate_mode = False
        self.last_particle_added_time = 0.0
        self.prev_fps_update_time = time.time()
        self.prev_time = self.prev_fps_update_time

        self.gui = GUI(self.state, title)
        self.state.create_group_callbacks.append(self.gui.create_group)

        self.gui.save_btn.configure(command=self.save)
        self.gui.load_btn.configure(command=self.load)
        self.gui.set_selected_btn.configure(command=self.set_selected)
        self.gui.set_all_btn.configure(command=self.set_all)

        # Keyboard- and mouse-controls
        self.gui.canvas.bind("<B1-Motion>", self._mouse_m)
        self.gui.canvas.bind("<Button-1>", self._mouse_p)
        self.gui.canvas.bind("<ButtonRelease-1>", self._mouse_r)
        self.gui.canvas.bind("<B3-Motion>", self._right_mouse)
        self.gui.canvas.bind("<Button-3>", self._right_mouse)
        self.gui.canvas.bind("<MouseWheel>", self._on_scroll)

        self.gui.tk.bind("<space>", _no_event(self.state.toggle_paused))
        self.gui.tk.bind("<Delete>", _no_event(self.state.remove_selection))
        self.gui.tk.bind("<Control-a>", _no_event(self.state.select_all))
        self.gui.tk.bind("<Control-c>", _no_event(self.state.copy_selected))
        self.gui.tk.bind("<Control-x>", _no_event(self.state.cut))
        self.gui.tk.bind("<Control-v>", _no_event(self.state.paste))
        self.gui.tk.bind("<Control-l>", _no_event(self.state.lock_selection))
        self.gui.tk.bind(
            "<Control-Shift-KeyPress-L>", _no_event(self.state.unlock_selection)
        )
        self.gui.tk.bind("<l>", _no_event(self.state.link_selection))
        self.gui.tk.bind(
            "<Alt_R><l>", _no_event(partial(self.state.link_selection, fit_link=True))
        )
        self.gui.tk.bind("<Shift-L>", _no_event(self.state.unlink_selection))
        self.gui.tk.bind("<KeyPress-r>", _no_event(self._enter_rotate_mode))
        self.gui.tk.bind("<KeyRelease-r>", _no_event(self._exit_rotate_mode))
        self.gui.tk.bind("<Control-s>", _no_event(self.save))
        self.gui.tk.bind("<Control-o>", _no_event(self.load))

    def _iter_in_range(self, circle: Circle) -> Iterable[Particle]:
        for particle in self.state.particles:
            if particle.circle.is_in_range(circle):
                yield particle

    def _mouse_p(self, event: tk.Event) -> None:
        mouse_circle = Circle(event.x, event.y, self.state.mr)
        if self.state.mouse_mode == "SELECT":
            selected = False
            for p in self._iter_in_range(mouse_circle):
                self.state.select_particle(p)
                selected = True
            if not selected:
                self.state.selection = []
        elif self.state.mouse_mode == "MOVE":
            selected = False
            for p in self._iter_in_range(mouse_circle):
                p.mouse = True
                if p in self.state.selection:
                    selected = True
            if selected:
                for particle in self.state.selection:
                    particle.mouse = True
        elif self.state.mouse_mode == "ADD":
            self.state.selection = []
            self.add_particle(event.x, event.y)

    def _mouse_m(self, event: tk.Event) -> None:
        if self.state.mouse_mode == "SELECT":
            mouse_circle = Circle(event.x, event.y, self.state.mr)
            for p in self._iter_in_range(mouse_circle):
                self.state.select_particle(p)
        elif (
            self.state.mouse_mode == "ADD"
            and time.time() - self.last_particle_added_time
            >= self.state.min_spawn_delay
        ):
            self.add_particle(event.x, event.y)

    def _mouse_r(self, _event: tk.Event) -> None:
        if self.state.mouse_mode == "MOVE" or self.state.pasting:
            for p in self.state.particles:
                p.mouse = False
        self.state.pasting = False

    def _right_mouse(self, event: tk.Event) -> None:
        temp = self.state.particles.copy()
        mouse_circle = Circle(event.x, event.y, self.state.mr)
        for p in temp:
            if p.circle.is_in_range(mouse_circle):
                self.state.remove_particle(p)

    def _on_scroll(self, event: tk.Event) -> None:
        if self.rotate_mode:
            for p in self.state.selection:
                p.x, p.y = self.state.rotate_2d(
                    p.x, p.y, event.x, event.y, angle=event.delta / 500 * self.state.mr
                )
        else:
            self.state.mr = max(self.state.mr * 2 ** (event.delta / 500), 1)

    def _enter_rotate_mode(self) -> None:
        self.rotate_mode = True

    def _exit_rotate_mode(self) -> None:
        self.rotate_mode = False

    def _get_particle_settings(self) -> Optional[ParticleFactory]:
        try:
            return self.gui.get_particle_settings()
        except Exception as error:
            self.state.errors.append(Error("Input-Error", error))
        return None

    def _set_particles(self, particles: Iterable[Particle]) -> None:
        for p in particles:
            # Update for each particle in case of 'random' and avoid sharing properties:
            factory = self._get_particle_settings()
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
            gui_particle_state=self.gui.get_particle_settings(),
            particles=particles_to_builders(self.state.particles),
        )

    def from_controller_state(self, controller_state: ControllerState) -> None:
        self.state = SimulationState(**asdict(controller_state.sim_data))
        self.gui.register_sim(self.state)
        self.state.create_group_callbacks = [self.gui.create_group]
        self.gui.set_particle_settings(controller_state.gui_particle_state)

        self.gui.groups_entry["values"] = []
        self.gui.set_sim_settings(controller_state.gui_settings)

        for p in self.state.particles.copy():
            self.state.remove_particle(p)

        self.state.groups = {}
        for particle in builders_to_particles(controller_state.particles):
            self.state.register_particle(particle)

    def add_particle(self, x: float, y: float) -> None:
        p = self._get_particle_settings()
        if p is not None:
            particle = Particle(
                x=x,
                y=y,
                radius=p.radius,
                color=p.color,
                props=p.props,
                velocity=np.array(p.velocity),
            )
            self.state.register_particle(particle)
            self.last_particle_added_time = time.time()

    def _paint_image(self, link_colors: Iterable[Link]) -> npt.NDArray[np.uint8]:
        if not self.state.stress_visualization or self.state.paused:
            return paint_image(self.state, None)
        return paint_image(self.state, link_colors)

    def save(self, filename: Optional[str] = None) -> None:
        controller_state = self.to_controller_state()
        try:
            self.gui.save_manager.save(controller_state, filename=filename)
        except Exception as error:
            self.state.errors.append(Error("Saving-Error", error))

    def load(self, filename: Optional[str] = None) -> None:
        if not self.state.paused:
            self.state.toggle_paused()
        try:
            controller_state = self.gui.save_manager.load(filename=filename)
            if controller_state is None:
                return
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
            links = self.state.simulate_step()
            self._update_timings(new_time=time.time())
            image = self._paint_image(links)
            self.gui.update(image, self.fps)
