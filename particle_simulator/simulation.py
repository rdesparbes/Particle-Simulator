import time
import tkinter as tk
from dataclasses import asdict
from typing import (
    Optional,
    Tuple,
    List,
    Union,
    Iterable,
)

import numpy as np
import numpy.typing as npt
from pynput.keyboard import Listener, Key, KeyCode

from .controller_state import ControllerState
from .conversion import builders_to_particles, particles_to_builders
from .error import Error
from .geometry import Circle
from .gui import GUI
from .painter import paint_image
from .particle import Particle
from .particle_factory import ParticleFactory, ParticleBuilder
from .simulation_state import SimulationState, Link


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
        self.shift = False
        self.start_save = False
        self.start_load = False
        self.prev_fps_update_time = time.time()
        self.prev_time = self.prev_fps_update_time
        self.clipboard: List[ParticleBuilder] = []
        self.pasting = False

        self.gui = GUI(self.state, title)
        self.state.create_group_callbacks.append(self.gui.create_group)
        self._link_colors: List[Link] = []

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

        self.listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _copy_selected(self) -> None:
        self.clipboard = []
        for factory in particles_to_builders(self.state.selection):
            factory.x -= self.state.mx
            factory.y -= self.state.my
            self.clipboard.append(factory)

    def _cut(self) -> None:
        self._copy_selected()
        self.state.remove_selection()

    def _iter_in_range(self, circle: Circle) -> Iterable[Particle]:
        for particle in self.state.particles:
            if particle.circle.is_in_range(circle):
                yield particle

    def _mouse_p(self, event: tk.Event) -> None:
        self.gui.canvas.focus_set()
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
        if self.state.mouse_mode == "MOVE" or self.pasting:
            for p in self.state.particles:
                p.mouse = False
        self.pasting = False

    def _right_mouse(self, event: tk.Event) -> None:
        self.gui.canvas.focus_set()
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

    def _on_press(self, key: Union[Key, KeyCode, None]) -> None:
        if not self.state.focus:
            return
        # SPACE to pause
        if key == Key.space:
            self.state.toggle_paused()
        # DELETE to delete
        elif key == Key.delete:
            self.state.remove_selection()
        elif key in {Key.shift_l, Key.shift_r}:
            self.shift = True
        # CTRL + A to select all
        elif KeyCode.from_char(str(key)).char == r"'\x01'":
            self.state.select_all()
        # CTRL + C to copy
        elif KeyCode.from_char(str(key)).char == r"'\x03'":
            self._copy_selected()
        # CTRL + V to copy
        elif KeyCode.from_char(str(key)).char == r"'\x16'":
            self._paste()
        # CTRL + X to cut
        elif KeyCode.from_char(str(key)).char == r"'\x18'":
            self._cut()
        # CTRL + L and CTRL + SHIFT + L to lock and 'unlock'
        elif KeyCode.from_char(str(key)).char == r"'\x0c'" and not self.shift:
            self.state.lock_selection()
        elif KeyCode.from_char(str(key)).char == r"'\x0c'" and self.shift:
            self.state.unlock_selection()
        # L to link, SHIFT + L to unlink and ALT GR + L to fit-link
        elif KeyCode.from_char(str(key)).char == "'l'":
            self.state.link_selection()
        elif KeyCode.from_char(str(key)).char == "<76>":
            self.state.link_selection(fit_link=True)
        elif KeyCode.from_char(str(key)).char == "'L'":
            self.state.unlink_selection()
        # R to enter rotate-mode
        elif KeyCode.from_char(str(key)).char == "'r'":
            self.rotate_mode = True
        # CTRL + S to save
        elif KeyCode.from_char(str(key)).char == r"'\x13'":
            self.start_save = True  # Threading-issues
        # CTRL + O to load / open
        elif KeyCode.from_char(str(key)).char == r"'\x0f'":
            self.start_load = True

    def _on_release(self, key: Union[Key, KeyCode, None]) -> None:
        if key in {Key.shift_l, Key.shift_r}:
            self.shift = False
        elif KeyCode.from_char(str(key)).char == "'r'":
            self.rotate_mode = False

    def _get_particle_settings(self) -> Optional[ParticleFactory]:
        try:
            return self.gui.get_particle_settings()
        except Exception as error:
            self.state.error = Error("Input-Error", error)
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

    def _paste(self) -> None:
        self.pasting = True
        particles: List[Particle] = []
        for particle in builders_to_particles(self.clipboard):
            particle.x += self.state.mx
            particle.y += self.state.my
            particle.mouse = True
            self.state.register_particle(particle)
        self.state.selection = particles

    def _paint_image(self, link_colors: Iterable[Link]) -> npt.NDArray[np.uint8]:
        if not self.state.stress_visualization or self.state.paused:
            return paint_image(self.state, None)
        return paint_image(self.state, link_colors)

    def save(self, filename: Optional[str] = None) -> None:
        controller_state = self.to_controller_state()
        try:
            self.gui.save_manager.save(controller_state, filename=filename)
        except Exception as error:
            self.state.error = Error("Saving-Error", error)

    def load(self, filename: Optional[str] = None) -> None:
        if not self.state.paused:
            self.state.toggle_paused()
        try:
            controller_state = self.gui.save_manager.load(filename=filename)
            if controller_state is None:
                return
            self.from_controller_state(controller_state)
        except Exception as error:
            self.state.error = Error("Loading-Error", error)

    def _handle_save_manager(self) -> None:
        if self.start_save:
            self.save()
            self.start_save = False
        if self.start_load:
            self.load()
            self.start_load = False

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

    def _update_mouse(self) -> None:
        self.state.focus = self.gui.get_focus()
        self.state.update_mouse_pos(self.gui.get_mouse_pos())

    def simulate(self) -> None:
        while self.state.running:
            self._handle_save_manager()
            self._update_mouse()
            links = self.state.simulate_step()
            self._update_timings(new_time=time.time())
            image = self._paint_image(links)
            self.gui.update(
                image,
                paused=self.state.paused,
                fps=self.fps,
                particle_count=len(self.state.particles),
                error=self.state.error,
            )
            self.state.error = None
