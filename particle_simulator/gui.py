import bisect
import os
import tkinter as tk
from tkinter import messagebox
from typing import Sequence, Dict, Any, Optional, List

import numpy as np
import numpy.typing as npt
from PIL import ImageTk, Image

from .code_window import CodeWindow
from .color import generate_random
from .error import Error
from .extra_window import ExtraWindow
from .gui_widgets import GUIWidgets
from .particle import Particle
from .particle_factory import ParticleFactory
from .particle_properties import ParticleProperties
from .save_manager import SaveManager
from .sim_gui_settings import SimGUISettings
from .simulation_state import SimulationState


class GUI(GUIWidgets):
    def __init__(self, sim: SimulationState, title: str) -> None:
        super().__init__(sim.width, sim.height, title)
        self.save_manager = SaveManager(file_location=os.path.dirname(self.path))
        self.code_window: Optional[CodeWindow] = None
        self.extra_window: Optional[ExtraWindow] = None
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
        self.group_add_btn.configure(command=self._add_group)
        self.copy_selected_btn.configure(command=self._copy_from_selected)
        self.grid_res_x.configure(command=self._set_grid_x)
        self.grid_res_y.configure(command=self._set_grid_y)

    def _register_sim(self, sim: SimulationState) -> None:
        self.pause_button.configure(
            image=self.play_photo if sim.paused else self.pause_photo,
            command=sim.toggle_paused,
        )
        self.link_btn.configure(command=sim.link_selection)
        self.unlink_btn.configure(command=sim.unlink_selection)
        self._set_entry(self.gravity_entry, str(sim.g))
        self._set_entry(self.air_res_entry, str(sim.air_res))
        self._set_entry(self.friction_entry, str(sim.ground_friction))
        self.temp_sc.set(sim.temperature)
        self.speed_sc.set(sim.speed)
        self.show_fps.set(sim.show_fps)
        self.show_num.set(sim.show_num)
        self.show_links.set(sim.show_links)
        self.top_bool.set(sim.top)
        self.bottom_bool.set(sim.bottom)
        self.left_bool.set(sim.left)
        self.right_bool.set(sim.right)
        self._set_entry(self.delay_entry, str(sim.min_spawn_delay))
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

    def _set_grid_x(self) -> None:
        self.sim.grid_res_x = int(self.grid_res_x.get())

    def _set_grid_y(self, *_args: Any) -> None:
        self.sim.grid_res_y = int(self.grid_res_y.get())

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

    def _add_group(self) -> None:
        name = self.sim.add_group()
        index = self._create_group(name)
        self.groups_entry.current(index)

    def _create_group(self, name: str) -> int:
        try:
            return self.groups_entry["values"].index(name)
        except ValueError:
            group_names: List[str] = list(self.groups_entry["values"])
            index = bisect.bisect(group_names, name)
            group_names.insert(index, name)
            self.groups_entry["values"] = group_names
            return index

    def create_group(self, name: str) -> None:
        self._create_group(name)

    def get_focus(self) -> bool:
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
        self._set_entry(self.grid_res_x, str(s.grid_res_x))
        self._set_entry(self.grid_res_y, str(s.grid_res_y))
        self._set_entry(self.delay_entry, str(s.delay))

    def get_particle_settings(self) -> ParticleFactory:
        props = ParticleProperties(
            mass=float(self.mass_entry.get()),
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
        )
        color = self._parse_color()
        if color is None:
            color = generate_random()
        factory = ParticleFactory(
            color=color,
            props=props,
            radius=float(self.radius_entry.get()),
            velocity=(
                float(self.velocity_x_entry.get()),
                float(self.velocity_y_entry.get()),
            ),
        )
        return factory

    def _set_particle_properties(self, p: ParticleProperties) -> None:
        self._set_entry(self.mass_entry, str(p.mass))
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

    def set_particle_settings(self, particle_settings: ParticleFactory) -> None:
        p = particle_settings
        self.color_var.set(str(p.color))
        self._set_particle_properties(p.props)
        self._set_entry(self.velocity_x_entry, str(p.velocity[0]))
        self._set_entry(self.velocity_y_entry, str(p.velocity[1]))
        self._set_entry(self.radius_entry, str(p.radius))

    def _copy_from_selected(self) -> None:
        selection: Sequence[Particle] = self.sim.selection

        particle_settings: Dict[str, Any] = {}
        for i, p in enumerate(selection):
            variable_names: Dict[str, Any] = {
                "radius_entry": p.radius,
                "color_var": p.color,
                "mass_entry": p.props.mass,
                "velocity_x_entry": p.velocity[0],
                "velocity_y_entry": p.velocity[1],
                "bounciness_entry": p.props.bounciness,
                "do_collision_bool": p.props.collisions,
                "locked_bool": p.props.locked,
                "linked_group_bool": p.props.linked_group_particles,
                "attr_r_entry": p.props.attract_r,
                "repel_r_entry": p.props.repel_r,
                "attr_strength_entry": p.props.attraction_strength,
                "repel_strength_entry": p.props.repulsion_strength,
                "link_attr_break_entry": p.props.link_attr_breaking_force,
                "link_repel_break_entry": p.props.link_repel_breaking_force,
                "groups_entry": p.props.group,
                "separate_group_bool": p.props.separate_group,
                "gravity_mode_bool": p.props.gravity_mode,
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
                elif isinstance(widget, tk.StringVar):
                    if same:
                        widget.set(str(part_val))
                    else:
                        widget.set("random")
                else:
                    raise NotImplementedError(f"Unexpected widget: {type(widget)}")

    def _update(self) -> None:
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

    def destroy(self) -> None:
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.sim.running = False
            self.tk.destroy()
