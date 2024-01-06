import time
import tkinter as tk
from dataclasses import asdict
from typing import (
    Optional,
    Any,
    Dict,
    Tuple,
    List,
)

import cv2
import numpy as np
import numpy.typing as npt
from pynput.keyboard import Listener, Key, KeyCode

from . import sim_pickle
from .controller_state import ControllerState
from .error import Error
from .grid import Grid
from .gui import GUI
from .particle import Particle
from .particle_factory import ParticleFactory
from .sim_pickle import (
    SimPickle,
)
from .simulation_state import SimulationState


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
        self.state = SimulationState(
            width=width,
            height=height,
            temperature=temperature,
            g=g,
            air_res=air_res,
            ground_friction=ground_friction,
        )

        self.fps = 0.0
        self.fps_update_delay = fps_update_delay
        self.rotate_mode = False
        self.last_particle_added_time = 0.0
        self.mouse_down = False
        self.mouse_down_start: Optional[float] = None
        self.shift = False
        self.start_save = False
        self.start_load = False
        self.focus = True
        self.grid = Grid(*gridres, height=height, width=width)
        self.prev_fps_update_time = time.time()
        self.prev_time = self.prev_fps_update_time
        self.clipboard: List[Dict[str, Any]] = []
        self.pasting = False

        self.gui = GUI(self.state, title, gridres)
        self.state.add_group_callbacks.append(self.gui.create_group)

        self.gui.save_btn.configure(command=self.save)
        self.gui.load_btn.configure(command=self.load)
        self.gui.set_selected_btn.configure(command=self.set_selected)
        self.gui.set_all_btn.configure(command=self.set_all)
        self.gui.grid_res_x_value.trace("w", self._update_grid)
        self.gui.grid_res_y_value.trace("w", self._update_grid)

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
        for p in self.state.selection:
            dictionary = p.return_dict(index_source=self.state.selection)
            dictionary["x"] -= self.state.mx
            dictionary["y"] -= self.state.my
            self.clipboard.append(dictionary)

    def _cut(self) -> None:
        self._copy_selected()
        temp = self.state.selection.copy()
        for p in temp:
            self.state.remove_particle(p)

    def _simulate_step(self):
        self.state.link_colors = []
        if self.state.use_grid:
            self.grid.init_grid(self.state.particles)
        if self.state.toggle_pause:
            self.state.paused = not self.state.paused

            if not self.state.paused:
                self.state.selection = []
            self.state.toggle_pause = False
        for particle in self.state.particles:
            if not particle.interacts:
                near_particles = []
            elif particle.interacts_with_all:
                near_particles = self.state.particles
            elif self.state.use_grid:
                near_particles = self.grid.return_particles(particle)
            else:
                near_particles = self.state.particles
            particle.update(near_particles)
            if particle.is_out_of_bounds():
                self.state.remove_particle(particle)

    def _mouse_p_part(self, particle: Particle, x: int, y: int) -> bool:
        if particle.distance(x, y) <= max(self.state.mr, particle.radius):
            if self.state.mouse_mode == "SELECT":
                self.state.select_particle(particle)
                return True

            particle.mouse = True
            if particle in self.state.selection:
                return True
        return False

    def _mouse_p(self, event: tk.Event) -> None:
        self.gui.canvas.focus_set()
        self.mouse_down_start = time.time()
        self.mouse_down = True
        if self.state.mouse_mode in {"SELECT", "MOVE"}:
            selected = any(
                self._mouse_p_part(p, event.x, event.y) for p in self.state.particles
            )
            if not selected:
                self.state.selection = []
            elif self.state.mouse_mode == "MOVE":
                for particle in self.state.selection:
                    particle.mouse = True
        elif self.state.mouse_mode == "ADD":
            if len(self.state.selection) > 0:
                self.state.selection = []

            self.add_particle(event.x, event.y)

    def _mouse_m(self, event: tk.Event) -> None:
        if self.state.mouse_mode == "SELECT":
            for p in self.state.particles:
                self._mouse_p_part(p, event.x, event.y)
        elif (
            self.state.mouse_mode == "ADD"
            and time.time() - self.last_particle_added_time
            >= self.state.min_spawn_delay
        ):
            self.add_particle(event.x, event.y)

    def _mouse_r(self, _event: tk.Event) -> None:
        self.mouse_down = False
        if self.state.mouse_mode == "MOVE" or self.pasting:
            for p in self.state.particles:
                p.mouse = False
        self.pasting = False

    def _right_mouse(self, event: tk.Event) -> None:
        self.gui.canvas.focus_set()
        temp = self.state.particles.copy()
        for p in temp:
            if np.sqrt((event.x - p.x) ** 2 + (event.y - p.y) ** 2) <= max(
                self.state.mr, p.radius
            ):
                self.state.remove_particle(p)

    def _on_scroll(self, event: tk.Event) -> None:
        if self.rotate_mode:
            for p in self.state.selection:
                p.x, p.y = self.state.rotate_2d(
                    p.x, p.y, event.x, event.y, event.delta / 500 * self.state.mr
                )
        else:
            self.state.mr = max(self.state.mr * 2 ** (event.delta / 500), 1)

    def _on_press(self, key: Key) -> None:
        if not self.focus:
            return
        # SPACE to pause
        if key == Key.space:
            self.state.toggle_paused()
        # DELETE to delete
        elif key == Key.delete:
            temp = self.state.selection.copy()
            for p in temp:
                self.state.remove_particle(p)
        elif key == Key.shift_l or key == Key.shift_r:
            self.shift = True
        # CTRL + A to select all
        elif KeyCode.from_char(str(key)).char == r"'\x01'":
            for p in self.state.particles:
                self.state.select_particle(p)
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
            for p in self.state.selection:
                p.locked = True
        elif KeyCode.from_char(str(key)).char == r"'\x0c'" and self.shift:
            for p in self.state.selection:
                p.locked = False
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

    def _on_release(self, key: Key) -> None:
        if key in {Key.shift_l, Key.shift_r}:
            self.shift = False
        elif KeyCode.from_char(str(key)).char == "'r'":
            self.rotate_mode = False

    def update_grid(self, row_count: int, col_count: int) -> None:
        self.state.grid_res_x = col_count
        self.state.grid_res_y = row_count
        self.grid = Grid(
            row_count,
            col_count,
            height=self.state.height,
            width=self.state.width,
        )

    def _update_grid(self, *_event) -> None:
        row_count = self.gui.grid_res_x_value.get()
        col_count = self.gui.grid_res_y_value.get()
        self.update_grid(row_count, col_count)

    def _get_particle_settings(self) -> Optional[ParticleFactory]:
        try:
            particle_settings = self.gui.get_particle_settings()
            if particle_settings.radius is None:
                particle_settings.radius = self.state.mr
            return particle_settings
        except Exception as error:
            self.state.error = Error("Input-Error", error)
        return None

    def set_selected(self) -> None:
        particle_settings = self._get_particle_settings()
        if particle_settings is None:
            return
        temp = self.state.selection.copy()
        for p in temp:
            p = self.state.replace_particle(p, particle_settings)
            self.state.selection.append(p)

    def set_all(self) -> None:
        temp = self.state.particles.copy()
        for p in temp:
            particle_settings = (
                self._get_particle_settings()
            )  # Update for each particle in case of 'random'
            if particle_settings is not None:
                self.state.replace_particle(p, particle_settings)

    def to_dict(self) -> SimPickle:
        controller_state = ControllerState(
            sim_data=self.state,
            gui_settings=self.gui.get_sim_settings(),
            gui_particle_state=self.gui.get_particle_settings(),
            particles=self.state.particles,
        )
        return sim_pickle.to_dict(controller_state)

    def from_dict(self, data: SimPickle) -> None:
        controller_state = sim_pickle.from_dict(data)
        self.state = SimulationState(**asdict(controller_state.sim_data))
        self.gui.register_sim(self.state)
        self.state.add_group_callbacks = [self.gui.create_group]
        self.gui.set_particle_settings(controller_state.gui_particle_state)

        self.gui.group_indices = []
        self.gui.groups_entry["values"] = []
        self.gui.set_sim_settings(controller_state.gui_settings)

        for p in self.state.particles.copy():
            self.state.remove_particle(p)

        self.state.groups = {}
        particles = [
            Particle(self.state, **asdict(particle_data))
            for particle_data in controller_state.particles
        ]
        for particle in particles:
            particle.link_lengths = {
                particles[index]: value
                for index, value in particle.link_indices_lengths.items()
            }
            particle.link_indices_lengths = {}
            self.state.register_particle(particle)

    def add_particle(self, x: float, y: float) -> None:
        particle_factory = self._get_particle_settings()
        if particle_factory is not None:
            p = Particle(self.state, x=x, y=y, **asdict(particle_factory))
            self.state.register_particle(p)
            self.last_particle_added_time = time.time()

    def _paste(self) -> None:
        self.pasting = True
        temp_particles = []
        for data in self.clipboard:
            p = Particle(self.state, x=0, y=0, group=data["group"])
            self.state.register_particle(p)
            temp_particles.append(p)

        for i, data in enumerate(self.clipboard):
            d = data.copy()
            particle = temp_particles[i]
            d["x"] += self.state.mx
            d["y"] += self.state.my
            for key, value in d.items():
                try:
                    vars(particle)[key] = value.copy()
                except AttributeError:
                    vars(particle)[key] = value

            particle.link_lengths = {
                temp_particles[index]: value
                for index, value in particle.link_lengths.items()
            }
            particle.mouse = True
        self.state.selection = temp_particles

    def _draw_image(self) -> npt.NDArray[np.uint8]:
        image = np.full(
            (self.state.height, self.state.width, 3),
            self.state.bg_color[0],
            dtype=np.uint8,
        )
        if self.state.show_links:
            if self.state.stress_visualization and not self.state.paused:
                for p1, p2, percentage in self.state.link_colors:
                    color = [max(255 * percentage, 235)] + [235 * (1 - percentage)] * 2
                    cv2.line(
                        image,
                        (int(p1.x), int(p1.y)),
                        (int(p2.x), int(p2.y)),
                        color,
                        1,
                    )
            else:
                for p1 in self.state.particles:
                    for p2 in p1.link_lengths:
                        cv2.line(
                            image,
                            (int(p1.x), int(p1.y)),
                            (int(p2.x), int(p2.y)),
                            [235] * 3,
                            1,
                        )
        for particle in self.state.particles:
            cv2.circle(
                image,
                (int(particle.x), int(particle.y)),
                int(particle.radius),
                [int(c) for c in particle.color],
                -1,
            )
        for particle in self.state.selection:
            cv2.circle(
                image,
                (int(particle.x), int(particle.y)),
                int(particle.radius),
                [0, 0, 255],
                2,
            )
        cv2.circle(image, (self.state.mx, self.state.my), int(self.state.mr), [127] * 3)
        return image

    def save(self, filename: Optional[str] = None) -> None:
        try:
            data = self.to_dict()
            self.gui.save_manager.save(data, filename=filename)
        except Exception as error:
            self.state.error = Error("Saving-Error", error)

    def load(self, filename: Optional[str] = None) -> None:
        if not self.state.paused:
            self.state.toggle_paused()
        try:
            data = self.gui.save_manager.load(filename=filename)
            if data is None:
                return
            self.from_dict(data)
        except Exception as error:
            self.state.error = Error("Loading-Error", error)

    def _handle_save_manager(self):
        if self.start_save:
            self.save()
            self.start_save = False
        if self.start_load:
            self.load()
            self.start_load = False

    @property
    def _fps_update_time(self) -> float:
        return self.prev_fps_update_time + self.fps_update_delay

    def _update_timings(self, new_time: float):
        if new_time >= self._fps_update_time:
            try:
                self.fps = 1.0 / (new_time - self.prev_time)
            except ZeroDivisionError:
                pass
            self.prev_fps_update_time = new_time
        self.prev_time = new_time

    def _update_mouse_position(self):
        self.state.prev_mx, self.state.prev_my = self.state.mx, self.state.my
        self.state.mx, self.state.my = self.gui.get_mouse_pos()

    def simulate(self) -> None:
        while self.state.running:
            self.focus = self.gui.get_focus()
            self._handle_save_manager()
            self._simulate_step()
            self._update_timings(new_time=time.time())
            self._update_mouse_position()
            image = self._draw_image()
            self.gui.update(
                image,
                paused=self.state.paused,
                fps=self.fps,
                particle_count=len(self.state.particles),
                error=self.state.error,
            )
            self.state.error = None
