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

from .error import Error
from .grid import Grid
from .gui import GUI, Mode, CANVAS_X, CANVAS_Y
from .particle import Particle, Link
from .particle_state import ParticleState
from .sim_pickle import (
    SimPickle,
    ParticleSettings,
    AttributeType,
    SimSettings,
)
from .simulation_state import SimulationState


class Simulation(SimulationState):
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
        super().__init__(
            width=width,
            height=height,
            temperature=temperature,
            g=g,
            air_res=air_res,
            ground_friction=ground_friction,
        )

        self.fps = 0.0
        self.fps_update_delay = fps_update_delay
        self.mx, self.my = 0, 0
        self.prev_mx, self.prev_my = 0, 0
        self.rotate_mode = False
        self.min_spawn_delay = 0.05
        self.last_particle_added_time = 0.0
        self.mr = 5.0
        self.mouse_down = False
        self.mouse_down_start: Optional[float] = None
        self.shift = False
        self.start_save = False
        self.start_load = False
        self.running = True
        self.focus = True
        self.link_colors: List[Link] = []
        self.grid = Grid(*gridres, height=height, width=width)
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.clipboard: List[Dict[str, Any]] = []
        self.pasting = False

        self.gui = GUI(self, title, gridres)
        self.mouse_mode: Mode = "MOVE"

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

    def _simulate_step(self):
        self.link_colors = []
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

    def _mouse_p_part(self, particle: Particle, x: int, y: int) -> bool:
        if np.sqrt((x - particle.x) ** 2 + (y - particle.y) ** 2) <= max(
            int(self.mr), particle.r
        ):
            if self.mouse_mode == "SELECT":
                self._select_particle(particle)
                return True

            particle.mouse = True
            if particle in self.selection:
                return True
        return False

    def _mouse_p(self, event: tk.Event) -> None:
        self.gui.canvas.focus_set()
        self.mouse_down_start = time.time()
        self.mouse_down = True
        if self.mouse_mode == "SELECT" or self.mouse_mode == "MOVE":
            selected = any(
                self._mouse_p_part(p, event.x, event.y) for p in self.particles
            )
            if not selected:
                self.selection = []
            elif self.mouse_mode == "MOVE":
                for particle in self.selection:
                    particle.mouse = True
        elif self.mouse_mode == "ADD":
            if len(self.selection) > 0:
                self.selection = []

            self.add_particle(event.x, event.y)

    def _mouse_m(self, event: tk.Event) -> None:
        if self.mouse_mode == "SELECT":
            for p in self.particles:
                self._mouse_p_part(p, event.x, event.y)
        elif (
            self.mouse_mode == "ADD"
            and time.time() - self.last_particle_added_time >= self.min_spawn_delay
        ):
            self.add_particle(event.x, event.y)

    def _mouse_r(self, _event: tk.Event) -> None:
        self.mouse_down = False
        if self.mouse_mode == "MOVE" or self.pasting:
            for p in self.particles:
                p.mouse = False
        self.pasting = False

    def _right_mouse(self, event: tk.Event) -> None:
        self.gui.canvas.focus_set()
        temp = self.particles.copy()
        for p in temp:
            if np.sqrt((event.x - p.x) ** 2 + (event.y - p.y) ** 2) <= max(
                int(self.mr), p.r
            ):
                self.remove_particle(p)

    def _on_scroll(self, event: tk.Event) -> None:
        if self.rotate_mode:
            for p in self.selection:
                p.x, p.y = self._rotate_2d(
                    p.x, p.y, event.x, event.y, event.delta / 500 * self.mr
                )
        else:
            self.mr = max(self.mr * 2 ** (event.delta / 500), 1)

    def _on_press(self, key: Key) -> None:
        if not self.focus:
            return
        # SPACE to pause
        if key == Key.space:
            self.toggle_paused()
        # DELETE to delete
        elif key == Key.delete:
            temp = self.selection.copy()
            for p in temp:
                self.remove_particle(p)
        elif key == Key.shift_l or key == Key.shift_r:
            self.shift = True
        # CTRL + A to select all
        elif KeyCode.from_char(str(key)).char == r"'\x01'":
            for p in self.particles:
                self._select_particle(p)
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
            for p in self.selection:
                p.locked = True
        elif KeyCode.from_char(str(key)).char == r"'\x0c'" and self.shift:
            for p in self.selection:
                p.locked = False
        # L to link, SHIFT + L to unlink and ALT GR + L to fit-link
        elif KeyCode.from_char(str(key)).char == "'l'":
            self.link_selection()
        elif KeyCode.from_char(str(key)).char == "<76>":
            self.link_selection(fit_link=True)
        elif KeyCode.from_char(str(key)).char == "'L'":
            self.unlink_selection()
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
        if key == Key.shift_l or key == Key.shift_r:
            self.shift = False
        elif KeyCode.from_char(str(key)).char == "'r'":
            self.rotate_mode = False

    def update_grid(self, row_count: int, col_count: int) -> None:
        self.grid_res_x = col_count
        self.grid_res_y = row_count
        self.grid = Grid(
            row_count,
            col_count,
            height=self.height,
            width=self.width,
        )

    def _inputs2dict(self) -> Optional[ParticleState]:
        try:
            kwargs = self.gui.inputs2dict()
            if kwargs.radius is None:
                kwargs.radius = self.mr
            return kwargs
        except Exception as error:
            self.error = Error("Input-Error", error)
        return None

    def set_selected(self) -> None:
        kwargs = self._inputs2dict()
        if kwargs is None:
            return
        temp = self.selection.copy()
        for p in temp:
            p = self._replace_particle(p, kwargs)
            self.selection.append(p)

    def set_all(self) -> None:
        temp = self.particles.copy()
        for p in temp:
            kwargs = self._inputs2dict()  # Update for each particle in case of 'random'
            if kwargs is not None:
                self._replace_particle(p, kwargs)

    def to_dict(self) -> SimPickle:
        sim_settings: SimSettings = {
            "gravity_entry": (self.gui.gravity_entry.get(), "entry"),
            "air_res_entry": (self.gui.air_res_entry.get(), "entry"),
            "friction_entry": (self.gui.friction_entry.get(), "entry"),
            "temp_sc": (self.gui.temp_sc.get(), "set"),
            "speed_sc": (self.gui.speed_sc.get(), "set"),
            "show_fps": (self.gui.show_fps.get(), "set"),
            "show_num": (self.gui.show_num.get(), "set"),
            "show_links": (self.gui.show_links.get(), "set"),
            "top_bool": (self.gui.top_bool.get(), "set"),
            "bottom_bool": (self.gui.bottom_bool.get(), "set"),
            "left_bool": (self.gui.left_bool.get(), "set"),
            "right_bool": (self.gui.right_bool.get(), "set"),
            "grid_bool": (self.gui.grid_bool.get(), "set"),
            "grid_res_x_value": (self.gui.grid_res_x_value.get(), "set"),
            "grid_res_y_value": (self.gui.grid_res_y_value.get(), "set"),
            "delay_entry": (self.gui.delay_entry.get(), "entry"),
            "calculate_radii_diff_bool": (
                self.gui.calculate_radii_diff_bool.get(),
                "set",
            ),
            "g_dir": (self.g_dir, "var"),
            "wind_force": (self.wind_force, "var"),
            "stress_visualization": (self.stress_visualization, "var"),
            "bg_color": (self.bg_color, "var"),
            "void_edges": (self.void_edges, "var"),
            "code": (self.code, "var"),
        }
        p = self.gui.inputs2dict()
        particle_settings: ParticleSettings = {
            "radius_entry": (str(p.radius), "entry"),
            "color_entry": (str(p.color), "entry"),
            "mass_entry": (str(p.mass), "entry"),
            "velocity_x_entry": (str(p.velocity[0]), "entry"),
            "velocity_y_entry": (str(p.velocity[1]), "entry"),
            "bounciness_entry": (str(p.bounciness), "entry"),
            "do_collision_bool": (p.collisions, "set"),
            "locked_bool": (p.locked, "set"),
            "linked_group_bool": (p.linked_group_particles, "set"),
            "attr_r_entry": (str(p.attract_r), "entry"),
            "repel_r_entry": (str(p.repel_r), "entry"),
            "attr_strength_entry": (str(p.attraction_strength), "entry"),
            "gravity_mode_bool": (p.gravity_mode, "set"),
            "repel_strength_entry": (str(p.repulsion_strength), "entry"),
            "link_attr_break_entry": (str(p.link_attr_breaking_force), "entry"),
            "link_repel_break_entry": (str(p.link_repel_breaking_force), "entry"),
            "groups_entry": (str(p.group), "entry"),
            "separate_group_bool": (p.separate_group, "set"),
        }

        return {
            "particles": [
                particle.return_dict(index_source=self.particles)
                for particle in self.particles
            ],
            "particle-settings": particle_settings,
            "sim-settings": sim_settings,
        }

    def from_dict(self, data: SimPickle) -> None:
        all_settings: Dict[str, Tuple[Any, AttributeType]] = {
            **data["particle-settings"],
            **data["sim-settings"],
        }
        self.gui.group_indices = []
        self.gui.groups_entry["values"] = []
        for key, (attribute_value, attribute_type) in all_settings.items():
            if attribute_type == "set":
                getattr(self.gui, key).set(attribute_value)
            elif attribute_type == "var":
                setattr(self, key, attribute_value)
            else:
                # Temporary fix to make the tests pass
                if key == "radius_entry" and attribute_value == "None":
                    attribute_value = "scroll"
                getattr(self.gui, key).delete(0, tk.END)
                getattr(self.gui, key).insert(0, attribute_value)

        for p in self.particles.copy():
            self.remove_particle(p)

        self.groups = {}
        for i, d in enumerate(data["particles"]):
            p = Particle(self, 0, 0, group=d["group"])
            for key, value in d.items():
                setattr(p, key, value)
            p.init_constants()
            self.register_particle(p)

        for particle in self.particles:
            particle.link_lengths = {
                self.particles[index]: value
                for index, value in particle.link_lengths.items()
            }

    def add_particle(self, x: float, y: float) -> None:
        kwargs = self._inputs2dict()
        if kwargs is not None:
            p = Particle(self, x, y, **asdict(kwargs))
            self.register_particle(p)
            self.last_particle_added_time = time.time()

    def _paste(self) -> None:
        self.pasting = True
        temp_particles = []
        for data in self.clipboard:
            p = Particle(self, 0, 0, group=data["group"])
            self.register_particle(p)
            temp_particles.append(p)

        for i, data in enumerate(self.clipboard):
            d = data.copy()
            particle = temp_particles[i]
            d["x"] += self.mx
            d["y"] += self.my
            for key, value in d.items():
                try:
                    vars(particle)[key] = value.copy()
                except AttributeError:
                    vars(particle)[key] = value

            particle.init_constants()
            particle.link_lengths = {
                temp_particles[index]: value
                for index, value in particle.link_lengths.items()
            }
            particle.mouse = True
        self.selection = temp_particles

    def _update_attributes(self) -> None:
        # Should be handled on the GUI side with callbacks
        self.g = float(self.gui.gravity_entry.get())
        self.air_res = float(self.gui.air_res_entry.get())
        self.ground_friction = float(self.gui.friction_entry.get())
        self.min_spawn_delay = float(self.gui.delay_entry.get())

        self.temperature = self.gui.temp_sc.get()
        self.speed = self.gui.speed_sc.get()

        self.use_grid = self.gui.grid_bool.get()
        self.calculate_radii_diff = self.gui.calculate_radii_diff_bool.get()
        self.top = self.gui.top_bool.get()
        self.bottom = self.gui.bottom_bool.get()
        self.left = self.gui.left_bool.get()
        self.right = self.gui.right_bool.get()

    def _draw_image(self, show_links: bool) -> npt.NDArray[np.uint8]:
        image = np.full((self.height, self.width, 3), self.bg_color[0], dtype=np.uint8)
        if show_links:
            if self.stress_visualization and not self.paused:
                for p1, p2, percentage in self.link_colors:
                    color = [max(255 * percentage, 235)] + [235 * (1 - percentage)] * 2
                    cv2.line(
                        image,
                        (int(p1.x), int(p1.y)),
                        (int(p2.x), int(p2.y)),
                        color,
                        1,
                    )
            else:
                for p1 in self.particles:
                    for p2 in p1.link_lengths:
                        cv2.line(
                            image,
                            (int(p1.x), int(p1.y)),
                            (int(p2.x), int(p2.y)),
                            [235] * 3,
                            1,
                        )
        for particle in self.particles:
            cv2.circle(
                image,
                (int(particle.x), int(particle.y)),
                int(particle.r),
                particle.color,
                -1,
            )
        for particle in self.selection:
            cv2.circle(
                image,
                (int(particle.x), int(particle.y)),
                int(particle.r),
                [0, 0, 255],
                2,
            )
        cv2.circle(image, (self.mx, self.my), int(self.mr), [127] * 3)
        return image

    def _update_focus(self):
        try:
            self.focus = isinstance(self.gui.tk.focus_displayof(), (tk.Canvas, tk.Tk))
        except KeyError:
            # Combobox
            self.focus = False

    def save(self, filename: Optional[str] = None) -> None:
        try:
            data = self.to_dict()
            self.gui.save_manager.save(data, filename=filename)
        except Exception as error:
            self.error = Error("Saving-Error", error)

    def load(self, filename: Optional[str] = None) -> None:
        if not self.paused:
            self.toggle_paused()
        try:
            data = self.gui.save_manager.load(filename=filename)
            if data is None:
                return
            self.from_dict(data)
        except Exception as error:
            self.error = Error("Loading-Error", error)

    def _handle_save_manager(self):
        if self.start_save:
            self.save()
            self.start_save = False
        if self.start_load:
            self.load()
            self.start_load = False

    def _update_timings(self):
        if time.time() - self.start_time >= self.fps_update_delay:
            try:
                self.fps = 1 / (time.time() - self.prev_time)
            except ZeroDivisionError:
                pass
            self.start_time = time.time()
        self.prev_time = time.time()

    def _update_mouse_position(self):
        self.prev_mx, self.prev_my = self.mx, self.my
        self.mx = self.gui.tk.winfo_pointerx() - self.gui.tk.winfo_rootx() - CANVAS_X
        self.my = self.gui.tk.winfo_pointery() - self.gui.tk.winfo_rooty() - CANVAS_Y

    def simulate(self) -> None:
        while self.running:
            self._update_attributes()
            self._update_focus()
            self._handle_save_manager()
            self._simulate_step()
            self._update_timings()
            self._update_mouse_position()
            image = self._draw_image(self.gui.show_links.get())
            self.gui.update(
                image,
                paused=self.paused,
                fps=self.fps,
                particle_count=len(self.particles),
                error=self.error,
            )
            self.error = None
