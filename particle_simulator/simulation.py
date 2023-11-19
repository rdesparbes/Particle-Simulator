import math
import time
import tkinter as tk
from tkinter import messagebox
from typing import (
    List,
    Collection,
    Iterable,
    Optional,
    Any,
    Dict,
    Tuple,
)

import cv2
import numpy as np
import numpy.typing as npt
from PIL import ImageTk, Image
from pynput.keyboard import Listener, Key, KeyCode

from .error import Error
from .grid import Grid
from .gui import GUI, Mode
from .particle import Particle, Link
from .save_manager import SaveManager


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
        distance: Optional[float] = None,
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
            gridres=gridres,
            temperature=temperature,
            g=g,
            air_res=air_res,
            ground_friction=ground_friction,
            fps_update_delay=fps_update_delay,
        )
        self.gui = GUI(self, title, gridres)
        self.save_manager = SaveManager(self)
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

    def _mouse_p_part(self, particle: Particle, event: tk.Event) -> bool:
        if np.sqrt((event.x - particle.x) ** 2 + (event.y - particle.y) ** 2) <= max(
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
            selected = any(self._mouse_p_part(p, event) for p in self.particles)
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
                self._mouse_p_part(p, event)
        elif (
            self.mouse_mode == "ADD"
            and time.time() - self.last_mouse_time >= self.min_spawn_delay
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
        if self.focus:
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
        self.grid = Grid(
            row_count,
            col_count,
            height=self.height,
            width=self.width,
        )

    def add_group(self) -> str:
        for i in range(1, len(self.groups) + 2):
            name = f"group{i}"
            if name not in self.groups:
                self.groups[name] = []
                return name
        assert False  # Unreachable (pigeonhole principle)

    def select_group(self, name: str) -> None:
        self.selection = list(self.groups[name])

    def _inputs2dict(self) -> Optional[Dict[str, Any]]:
        try:
            kwargs = self.gui.inputs2dict()
            if "radius" not in kwargs:
                kwargs["radius"] = self.mr
            return kwargs
        except Exception as error:
            self.error = Error("Input-Error", error)
        return None

    def register_particle(self, particle: Particle) -> None:
        try:
            self.groups[particle.group].append(particle)
        except KeyError:
            self.groups[particle.group] = [particle]
            self.gui.create_group(particle.group)
        self.particles.append(particle)

    def _replace_particle(self, p: Particle, kwargs: Dict[str, Any]) -> Particle:
        temp_link_lengths = p.link_lengths.copy()
        px, py = p.x, p.y
        self.remove_particle(p)
        p = Particle(self, px, py, **kwargs)
        self.register_particle(p)
        for link, length in temp_link_lengths.items():
            self.link([link, p], fit_link=length != "repel", distance=length)
        return p

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

    def copy_from_selected(self) -> None:
        variable_names = {
            "radius_entry": ["r", "entry"],
            "color_entry": ["color", "entry"],
            "mass_entry": ["m", "entry"],
            "velocity_x_entry": ["v[0]", "entry"],
            "velocity_y_entry": ["v[1]", "entry"],
            "bounciness_entry": ["bounciness", "entry"],
            "do_collision_bool": ["collision_bool", "set"],
            "locked_bool": ["locked", "set"],
            "linked_group_bool": ["linked_group_particles", "set"],
            "attr_r_entry": ["attr_r", "entry"],
            "repel_r_entry": ["repel_r", "entry"],
            "attr_strength_entry": ["attr", "entry"],
            "repel_strength_entry": ["repel", "entry"],
            "link_attr_break_entry": ["link_attr_breaking_force", "entry"],
            "link_repel_break_entry": ["link_repel_breaking_force", "entry"],
            "groups_entry": ["group", "entry"],
            "separate_group_bool": ["separate_group", "set"],
            "gravity_mode_bool": ["gravity_mode", "set"],
        }
        particle_settings = variable_names.copy()

        for i, p in enumerate(self.selection):
            for key, value in variable_names.items():
                val = eval("p." + value[0])

                if i == 0:
                    particle_settings[key] = val

                same = particle_settings[key] == val
                if value[1] == "set":
                    if same:
                        getattr(self.gui, key).set(val)
                    else:
                        getattr(self.gui, key).set(False)
                else:
                    if same:
                        getattr(self.gui, key).delete(0, tk.END)
                        getattr(self.gui, key).insert(0, str(val))
                    else:
                        getattr(self.gui, key).delete(0, tk.END)

    def add_particle(self, x: float, y: float) -> None:
        kwargs = self._inputs2dict()
        if kwargs is not None:
            p = Particle(self, x, y, **kwargs)
            self.register_particle(p)
            self.last_mouse_time = time.time()

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

    def _draw_image(self) -> npt.NDArray[np.uint8]:
        image = np.full((self.height, self.width, 3), self.bg_color[0], dtype=np.uint8)
        if self.gui.show_links.get():
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
                particle.r,
                particle.color,
                -1,
            )
        for particle in self.selection:
            cv2.circle(
                image,
                (int(particle.x), int(particle.y)),
                particle.r,
                [0, 0, 255],
                2,
            )
        cv2.circle(image, (self.mx, self.my), int(self.mr), [127] * 3)
        return image

    def _update_gui(self, image: npt.NDArray[np.uint8]) -> None:
        if self.error is not None:
            messagebox.showerror(self.error.name, str(self.error.exception))
            self.error = None
        photo = ImageTk.PhotoImage(
            image=Image.fromarray(image.astype(np.uint8)), master=self.gui.tk
        )
        self.gui.pause_button.config(
            image=self.gui.play_photo if self.paused else self.gui.pause_photo
        )

        self.gui.canvas.delete("all")
        self.gui.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        if self.gui.show_fps.get():
            self.gui.canvas.create_text(
                10,
                10,
                text=f"FPS: {round(self.fps, 2)}",
                anchor="nw",
                font=("Helvetica", 9, "bold"),
            )
        if self.gui.show_num.get():
            self.gui.canvas.create_text(
                10,
                25,
                text=f"Particles: {len(self.particles)}",
                anchor="nw",
                font=("Helvetica", 9, "bold"),
            )

        self.gui.update()

    def simulate(self):
        while self.running:
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

            if (
                self.mouse_down
                and time.time() - self.mouse_down_start >= self.min_hold_delay
            ):
                event = tk.Event()
                event.x, event.y = self.mx, self.my
                self._mouse_m(event)

            try:
                self.focus = isinstance(
                    self.gui.tk.focus_displayof(), (tk.Canvas, tk.Tk)
                )
            except KeyError:
                # Combobox
                self.focus = False

            if self.start_save:
                self.save_manager.save()
                self.start_save = False

            if self.start_load:
                self.save_manager.load()
                self.start_load = False

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

            if time.time() - self.start_time >= self.fps_update_delay:
                try:
                    self.fps = 1 / (time.time() - self.prev_time)
                except ZeroDivisionError:
                    pass
                self.start_time = time.time()
            self.prev_time = time.time()

            self.prev_mx, self.prev_my = self.mx, self.my
            self.mx = self.gui.tk.winfo_pointerx() - self.gui.tk.winfo_rootx()
            self.my = self.gui.tk.winfo_pointery() - self.gui.tk.winfo_rooty() - 30

            image = self._draw_image()

            self._update_gui(image)
