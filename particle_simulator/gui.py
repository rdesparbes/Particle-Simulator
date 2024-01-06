import tkinter as tk
from tkinter import colorchooser, messagebox
from typing import Literal, Sequence, Dict, Any, Optional, Union, Tuple

import numpy as np
import numpy.typing as npt
from PIL import ImageTk, Image

from .code_window import CodeWindow
from .error import Error
from .extra_window import ExtraWindow
from .gui_widgets import GUIWidgets
from .particle_data import ParticleData
from .particle_factory import ParticleFactory
from .sim_gui_settings import SimGUISettings
from .simulation_state import SimulationState


class GUI(GUIWidgets):
    def __init__(
        self, sim: SimulationState, title: str, gridres: Tuple[int, int]
    ) -> None:
        super().__init__(sim.width, sim.height, title, gridres)
        self.tk.protocol("WM_DELETE_WINDOW", self.destroy)
        self._register_sim(sim)
        self._set_callbacks()
        self.sim = sim

    def _set_callbacks(self) -> None:
        self.select_btn.configure(command=self._set_select_mode)
        self.move_btn.configure(command=self._set_move_mode)
        self.add_btn.configure(command=self._set_add_mode)
        self.code_btn.configure(command=self._create_code_window)
        self.gravity_entry.configure(command=self._set_gravity)
        self.air_res_entry.configure(command=self._set_air_res)
        self.friction_entry.configure(command=self._set_ground_friction)
        self.temp_sc.configure(command=self._set_temperature)
        self.speed_sc.configure(command=self._set_speed)
        self.fps_chk.configure(command=self._set_show_fps)
        self.num_chk.configure(command=self._set_show_num)
        self.links_chk.configure(command=self._set_show_links)
        self.top_chk.configure(command=self._set_top)
        self.bottom_chk.configure(command=self._set_bottom)
        self.left_chk.configure(command=self._set_left)
        self.right_chk.configure(command=self._set_right)
        self.grid_chk.configure(command=self._set_use_grid)
        self.delay_entry.configure(command=self._set_min_spawn_delay)
        self.calculate_radii_diff_chk.configure(command=self._set_calculate_radii_diff)
        self.extra_btn.configure(command=self._create_extra_window)
        self.group_add_btn.configure(command=self.add_group)
        self.copy_selected_btn.configure(command=self._copy_from_selected)

    def _register_sim(self, sim: SimulationState) -> None:
        self.pause_button.configure(
            image=self.play_photo if sim.paused else self.pause_photo,
            command=sim.toggle_paused,
        )
        self.load_btn.configure(command=sim.link_selection)
        self.unlink_btn.configure(command=sim.unlink_selection)
        self.gravity_entry.delete(0, tk.END)
        self.gravity_entry.insert(0, str(sim.g))
        self.air_res_entry.delete(0, tk.END)
        self.air_res_entry.insert(0, str(sim.air_res))
        self.friction_entry.delete(0, tk.END)
        self.friction_entry.insert(0, str(sim.ground_friction))
        self.temp_sc.set(sim.temperature)
        self.speed_sc.set(sim.speed)
        self.show_fps.set(sim.show_fps)
        self.show_num.set(sim.show_num)
        self.show_links.set(sim.show_links)
        self.top_bool.set(sim.top)
        self.bottom_bool.set(sim.bottom)
        self.left_bool.set(sim.left)
        self.right_bool.set(sim.right)
        self.delay_entry.delete(0, tk.END)
        self.delay_entry.insert(0, str(sim.min_spawn_delay))
        self._set_color(sim.bg_color[1])
        self.group_select_btn.configure(
            command=lambda: sim.select_group(self.groups_entry.get())
        )

    def register_sim(self, sim: SimulationState) -> None:
        self._register_sim(sim)
        self.sim = sim

    def _set_air_res(self) -> None:
        self.sim.air_res = float(self.air_res_entry.get())

    def _set_gravity(self) -> None:
        self.sim.g = float(self.gravity_entry.get())

    def _set_ground_friction(self) -> None:
        self.sim.ground_friction = float(self.friction_entry.get())

    def _set_temperature(self, new_temp: str) -> None:
        self.sim.temperature = float(new_temp)

    def _set_speed(self, new_speed: str) -> None:
        self.sim.speed = float(new_speed)

    def _set_show_fps(self) -> None:
        self.sim.show_fps = self.show_fps.get()

    def _set_show_num(self) -> None:
        self.sim.show_num = self.show_num.get()

    def _set_show_links(self) -> None:
        self.sim.show_links = self.show_links.get()

    def _set_top(self) -> None:
        self.sim.top = self.top_bool.get()

    def _set_bottom(self) -> None:
        self.sim.bottom = self.bottom_bool.get()

    def _set_left(self) -> None:
        self.sim.left = self.left_bool.get()

    def _set_right(self) -> None:
        self.sim.right = self.right_bool.get()

    def _set_use_grid(self) -> None:
        self.sim.use_grid = self.grid_bool.get()

    def _set_min_spawn_delay(self) -> None:
        self.sim.min_spawn_delay = float(self.delay_entry.get())

    def _set_calculate_radii_diff(self) -> None:
        self.sim.calculate_radii_diff = self.calculate_radii_diff_bool.get()

    def _set_select_mode(self) -> None:
        self.sim.mouse_mode = "SELECT"
        self.gui_canvas.itemconfig(self.select_rect, state="normal")
        self.gui_canvas.itemconfig(self.move_rect, state="hidden")
        self.gui_canvas.itemconfig(self.add_rect, state="hidden")

    def _set_move_mode(self) -> None:
        self.sim.mouse_mode = "MOVE"
        self.gui_canvas.itemconfig(self.select_rect, state="hidden")
        self.gui_canvas.itemconfig(self.move_rect, state="normal")
        self.gui_canvas.itemconfig(self.add_rect, state="hidden")

    def _set_add_mode(self) -> None:
        self.sim.mouse_mode = "ADD"
        self.gui_canvas.itemconfig(self.select_rect, state="hidden")
        self.gui_canvas.itemconfig(self.move_rect, state="hidden")
        self.gui_canvas.itemconfig(self.add_rect, state="normal")

    def _create_extra_window(self) -> None:
        self.extra_window = ExtraWindow(self.sim, str(self.path))

    def _create_code_window(self) -> None:
        self.code_window = CodeWindow()
        self.code_window.set_code(self.sim.code)
        self.code_window.set_exec_callback(self.sim.execute)
        self.code_window.set_save_callback(self.sim.set_code)

    @staticmethod
    def _extract_group_index(name: str) -> int:
        return int(name.replace("group", ""))

    def add_group(self) -> None:
        name = self.sim.add_group()
        self.create_group(name)
        self.groups_entry.current(self._extract_group_index(name) - 1)

    def create_group(self, name: str) -> None:
        self.group_indices.append(self._extract_group_index(name))
        self.group_indices.sort()
        self.groups_entry["values"] = [f"group{i}" for i in self.group_indices]

    def ask_color_entry(self, *_event):
        color, color_exa = colorchooser.askcolor(title="Choose color")
        if color is not None:
            self.color_entry.delete(0, tk.END)
            self.color_entry.insert(0, str(list(color)))
            self.tab2_canvas.itemconfig(self.part_color_rect, fill=color_exa)

    def get_focus(self):
        try:
            return isinstance(self.tk.focus_displayof(), (tk.Canvas, tk.Tk))
        except KeyError:
            return False

    def get_sim_settings(self) -> SimGUISettings:
        return SimGUISettings(
            show_fps=self.show_fps.get(),
            show_num=self.show_num.get(),
            show_links=self.show_links.get(),
            grid_res_x=int(self.grid_res_x.get()),
            grid_res_y=int(self.grid_res_y.get()),
            delay=float(self.delay_entry.get()),
        )

    def set_sim_settings(self, sim_settings: SimGUISettings) -> None:
        s = sim_settings

        self.show_fps.set(s.show_fps)
        self.show_num.set(s.show_num)
        self.show_links.set(s.show_links)
        self.grid_res_x_value.set(s.grid_res_x)
        self.grid_res_y_value.set(s.grid_res_y)
        self._set_entry(self.delay_entry, str(s.delay))

    def get_particle_settings(self) -> ParticleFactory:
        radius_str: str = self.radius_entry.get()
        if radius_str == "scroll":
            radius: Optional[float] = None
        else:
            radius = float(self.radius_entry.get())

        color_str: str = self.color_entry.get()
        if color_str == "random":
            color: Union[Tuple[int, int, int], Literal["random"]] = color_str
        else:
            color = tuple(map(int, eval(color_str)))

        return ParticleFactory(
            color=color,
            mass=float(self.mass_entry.get()),
            velocity=(
                float(self.velocity_x_entry.get()),
                float(self.velocity_y_entry.get()),
            ),
            bounciness=float(self.bounciness_entry.get()),
            attract_r=float(self.attr_r_entry.get()),
            repel_r=float(self.repel_r_entry.get()),
            attraction_strength=float(self.attr_strength_entry.get()),
            repulsion_strength=float(self.repel_strength_entry.get()),
            link_attr_breaking_force=float(self.link_attr_break_entry.get()),
            link_repel_breaking_force=float(self.link_repel_break_entry.get()),
            collisions=self.do_collision_bool.get(),
            locked=self.locked_bool.get(),
            linked_group_particles=self.linked_group_bool.get(),
            group=self.groups_entry.get(),
            separate_group=self.separate_group_bool.get(),
            gravity_mode=self.gravity_mode_bool.get(),
            radius=radius,
        )

    @staticmethod
    def _set_entry(entry: Union[tk.Entry, tk.Spinbox], text: str) -> None:
        entry.delete(0, tk.END)
        entry.insert(0, text)

    def set_particle_settings(self, particle_settings: ParticleFactory) -> None:
        p = particle_settings
        self._set_entry(self.color_entry, str(p.color))
        self._set_entry(self.mass_entry, str(p.mass))
        self._set_entry(self.velocity_x_entry, str(p.velocity[0]))
        self._set_entry(self.velocity_y_entry, str(p.velocity[1]))
        self._set_entry(self.bounciness_entry, str(p.bounciness))
        self._set_entry(self.attr_r_entry, str(p.attract_r))
        self._set_entry(self.repel_r_entry, str(p.repel_r))
        self._set_entry(self.attr_strength_entry, str(p.attraction_strength))
        self._set_entry(self.repel_strength_entry, str(p.repulsion_strength))
        self._set_entry(self.link_attr_break_entry, str(p.link_attr_breaking_force))
        self._set_entry(self.link_repel_break_entry, str(p.link_repel_breaking_force))
        self.do_collision_bool.set(p.collisions)
        self.locked_bool.set(p.locked)
        self.linked_group_bool.set(p.linked_group_particles)
        self._set_entry(self.groups_entry, p.group)
        self.separate_group_bool.set(p.separate_group)
        self.gravity_mode_bool.set(p.gravity_mode)
        self._set_entry(
            self.radius_entry, "scroll" if p.radius is None else str(p.radius)
        )

    def _copy_from_selected(self) -> None:
        selection: Sequence[ParticleData] = self.sim.selection

        particle_settings: Dict[str, Any] = {}
        for i, p in enumerate(selection):
            variable_names: Dict[str, Any] = {
                "radius_entry": p.radius,
                "color_entry": p.color,
                "mass_entry": p.mass,
                "velocity_x_entry": p.velocity[0],
                "velocity_y_entry": p.velocity[1],
                "bounciness_entry": p.bounciness,
                "do_collision_bool": p.collisions,
                "locked_bool": p.locked,
                "linked_group_bool": p.linked_group_particles,
                "attr_r_entry": p.attract_r,
                "repel_r_entry": p.repel_r,
                "attr_strength_entry": p.attraction_strength,
                "repel_strength_entry": p.repulsion_strength,
                "link_attr_break_entry": p.link_attr_breaking_force,
                "link_repel_break_entry": p.link_repel_breaking_force,
                "groups_entry": p.group,
                "separate_group_bool": p.separate_group,
                "gravity_mode_bool": p.gravity_mode,
            }
            for gui_attr, part_val in variable_names.items():
                if i == 0:
                    particle_settings[gui_attr] = part_val

                same = particle_settings[gui_attr] == part_val
                widget: tk.Widget = getattr(self, gui_attr)
                if isinstance(widget, tk.BooleanVar):
                    if same:
                        widget.set(part_val)
                    else:
                        widget.set(False)
                elif isinstance(widget, (tk.Entry, tk.Spinbox)):
                    widget.delete(0, tk.END)
                    if same:
                        widget.insert(0, str(part_val))
                else:
                    raise NotImplementedError(f"Unexpected widget: {type(widget)}")

    def _set_color(self, color: str) -> None:
        self.tab2_canvas.itemconfig(self.part_color_rect, fill=color)

    def change_color_entry(self, *event):
        try:
            color = eval(self.color_var.get())
            color_str = "#%02x%02x%02x" % tuple(color)
            self._set_color(color_str)
        except:
            if self.color_var.get() == "random" or self.color_var.get() == "":
                self._set_color("#ffffff")

    def _update(self):
        if self.code_window is not None:
            self.code_window.tk.update()
        if self.extra_window is not None:
            self.extra_window.update()

        self.tk.update()

    def update(
        self,
        image: npt.NDArray[np.uint8],
        paused: bool = True,
        fps: Optional[float] = None,
        particle_count: Optional[int] = None,
        error: Optional[Error] = None,
    ) -> None:
        if error is not None:
            messagebox.showerror(error.name, str(error.exception))
        photo = ImageTk.PhotoImage(image=Image.fromarray(image.astype(np.uint8)))
        self.pause_button.config(image=self.play_photo if paused else self.pause_photo)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        if self.show_fps.get() and fps is not None:
            self.canvas.create_text(
                10,
                10,
                text=f"FPS: {round(fps, 2)}",
                anchor="nw",
                font=("Helvetica", 9, "bold"),
            )
        if self.sim.show_num and particle_count is not None:
            self.canvas.create_text(
                10,
                25,
                text=f"Particles: {particle_count}",
                anchor="nw",
                font=("Helvetica", 9, "bold"),
            )

        self._update()

    def destroy(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.sim.running = False
            self.tk.destroy()
