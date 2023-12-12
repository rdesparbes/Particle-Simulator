import time
import tkinter as tk
from tkinter import messagebox
from typing import (
    Optional,
    Any,
    Dict,
    Tuple,
    Literal,
)

import cv2
import numpy as np
import numpy.typing as npt
from PIL import ImageTk, Image
from pynput.keyboard import Listener, Key, KeyCode

from .error import Error
from .grid import Grid
from .gui import GUI, Mode
from .particle import Particle
from .save_manager import SaveManager
from .simulation_state import SimulationState


_CANVAS_Y = 30  # The Y coordinates of the top-left corner of the canvas
AttributeType = Literal["set", "entry", "var"]


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
        variable_names: Dict[str, Tuple[str, AttributeType]] = {
            "radius_entry": ("r", "entry"),
            "color_entry": ("color", "entry"),
            "mass_entry": ("m", "entry"),
            "velocity_x_entry": ("v[0]", "entry"),
            "velocity_y_entry": ("v[1]", "entry"),
            "bounciness_entry": ("bounciness", "entry"),
            "do_collision_bool": ("collision_bool", "set"),
            "locked_bool": ("locked", "set"),
            "linked_group_bool": ("linked_group_particles", "set"),
            "attr_r_entry": ("attr_r", "entry"),
            "repel_r_entry": ("repel_r", "entry"),
            "attr_strength_entry": ("attr", "entry"),
            "repel_strength_entry": ("repel", "entry"),
            "link_attr_break_entry": ("link_attr_breaking_force", "entry"),
            "link_repel_break_entry": ("link_repel_breaking_force", "entry"),
            "groups_entry": ("group", "entry"),
            "separate_group_bool": ("separate_group", "set"),
            "gravity_mode_bool": ("gravity_mode", "set"),
        }
        particle_settings = variable_names.copy()

        for i, p in enumerate(self.selection):
            for key, (attribute_name, attribute_type) in variable_names.items():
                val = eval("p." + attribute_name)

                if i == 0:
                    particle_settings[key] = val

                same = particle_settings[key] == val
                if attribute_type == "set":
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

    def to_dict(self) -> Dict[str, Any]:
        sim_settings = {
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

        particle_settings = {
            "radius_entry": (self.gui.radius_entry.get(), "entry"),
            "color_entry": (self.gui.color_entry.get(), "entry"),
            "mass_entry": (self.gui.mass_entry.get(), "entry"),
            "velocity_x_entry": (self.gui.velocity_x_entry.get(), "entry"),
            "velocity_y_entry": (self.gui.velocity_y_entry.get(), "entry"),
            "bounciness_entry": (self.gui.bounciness_entry.get(), "entry"),
            "do_collision_bool": (self.gui.do_collision_bool.get(), "set"),
            "locked_bool": (self.gui.locked_bool.get(), "set"),
            "linked_group_bool": (self.gui.linked_group_bool.get(), "set"),
            "attr_r_entry": (self.gui.attr_r_entry.get(), "entry"),
            "repel_r_entry": (self.gui.repel_r_entry.get(), "entry"),
            "attr_strength_entry": (
                self.gui.attr_strength_entry.get(),
                "entry",
            ),
            "gravity_mode_bool": (self.gui.gravity_mode_bool.get(), "set"),
            "repel_strength_entry": (
                self.gui.repel_strength_entry.get(),
                "entry",
            ),
            "link_attr_break_entry": (
                self.gui.link_attr_break_entry.get(),
                "entry",
            ),
            "link_repel_break_entry": (
                self.gui.link_repel_break_entry.get(),
                "entry",
            ),
            "groups_entry": (self.gui.groups_entry.get(), "entry"),
            "separate_group_bool": (
                self.gui.separate_group_bool.get(),
                "set",
            ),
        }

        return {
            "particles": [
                particle.return_dict(index_source=self.particles)
                for particle in self.particles
            ],
            "particle-settings": particle_settings,
            "sim-settings": sim_settings,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        for key, (attribute_name, attribute_type) in list(
            data["particle-settings"].items()
        ) + list(data["sim-settings"].items()):
            if attribute_type == "set":
                getattr(self.gui, key).set(attribute_name)
            elif attribute_type == "var":
                setattr(self, key, attribute_name)
            else:
                getattr(self.gui, key).delete(0, tk.END)
                getattr(self.gui, key).insert(0, attribute_name)

        temp = self.particles.copy()
        for p in temp:
            self.remove_particle(p)

        self.groups = {}
        self.gui.group_indices = []
        self.gui.groups_entry["values"] = []
        for i in range(len(data["particles"])):
            p = Particle(self, 0, 0, group=data["particles"][i]["group"])
            self.register_particle(p)

        for i, d in enumerate(data["particles"]):
            particle = self.particles[i]

            for key, value in d.items():
                setattr(particle, key, value)
            particle.init_constants()

            particle.link_lengths = {
                self.particles[index]: value
                for index, value in particle.link_lengths.items()
            }

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

    def simulate(self) -> None:
        while self.running:
            self._update_attributes()

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

            self._simulate_step()

            if time.time() - self.start_time >= self.fps_update_delay:
                try:
                    self.fps = 1 / (time.time() - self.prev_time)
                except ZeroDivisionError:
                    pass
                self.start_time = time.time()
            self.prev_time = time.time()

            self.prev_mx, self.prev_my = self.mx, self.my
            self.mx = self.gui.tk.winfo_pointerx() - self.gui.tk.winfo_rootx()
            self.my = (
                self.gui.tk.winfo_pointery() - self.gui.tk.winfo_rooty() - _CANVAS_Y
            )

            image = self._draw_image()

            self._update_gui(image)
