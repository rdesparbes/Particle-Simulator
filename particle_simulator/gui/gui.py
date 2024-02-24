import bisect
import os
import tkinter as tk
from functools import partial
from tkinter import messagebox
from typing import Dict, Any, Optional, List, Set, Protocol

import numpy as np
import numpy.typing as npt
from PIL import ImageTk, Image

from particle_simulator.color import generate_random
from particle_simulator.engine.particle import Particle
from particle_simulator.engine.particle_factory import ParticleFactory
from particle_simulator.engine.particle_properties import ParticleProperties
from particle_simulator.engine.simulation_state import SimulationState
from particle_simulator.io.save_manager import SaveManager
from particle_simulator.sim_gui_settings import SimGUISettings
from .code_window import CodeWindow
from .extra_window import ExtraWindow
from .gui_widgets import GUIWidgets
from ..utils import any_args


class _SimController(Protocol):
    def mouse_button_1_pressed(self) -> None:
        pass

    def mouse_button_1_pressed_while_moving(self) -> None:
        pass

    def mouse_button_1_released(self) -> None:
        pass

    def on_scroll(self, factor: float) -> None:
        pass

    def enter_rotate_mode(self) -> None:
        pass

    def exit_rotate_mode(self) -> None:
        pass

    def set_selected(self) -> None:
        pass

    def set_all(self) -> None:
        pass

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass


class GUI(GUIWidgets):
    def __init__(self, sim: SimulationState, title: str = "Simulation") -> None:
        super().__init__(title, sim.width, sim.height)
        self.save_manager = SaveManager(file_location=os.path.dirname(self._path))
        self.code_window: Optional[CodeWindow] = None
        self.extra_window: Optional[ExtraWindow] = None
        self.tk.protocol("WM_DELETE_WINDOW", self._destroy)
        self.sim = sim
        self._register_sim()
        self._set_callbacks()

    def _bind_sim_events(self) -> None:
        self.canvas.bind("<B3-Motion>", any_args(self.sim.remove_in_range))
        self.canvas.bind("<Button-3>", any_args(self.sim.remove_in_range))

        self.tk.bind("<space>", any_args(self.sim.toggle_paused))
        self.tk.bind("<Delete>", any_args(self.sim.remove_selection))
        self.tk.bind("<Control-a>", any_args(self.sim.select_all))
        self.tk.bind("<Control-c>", any_args(self.sim.copy_selection))
        self.tk.bind("<Control-x>", any_args(self.sim.cut_selection))
        self.tk.bind("<Control-v>", any_args(self.sim.paste))
        self.tk.bind("<Control-l>", any_args(self.sim.lock_selection))
        self.tk.bind("<Control-Shift-KeyPress-L>", any_args(self.sim.unlock_selection))
        self.tk.bind("<l>", any_args(self.sim.link_selection))
        self.tk.bind(
            "<Alt_R><l>", any_args(partial(self.sim.link_selection, fit_link=True))
        )
        self.tk.bind("<Shift-L>", any_args(self.sim.unlink_selection))

    def register_controller(self, controller: _SimController) -> None:
        self.canvas.bind(
            "<B1-Motion>", any_args(controller.mouse_button_1_pressed_while_moving)
        )
        self.canvas.bind("<Button-1>", any_args(controller.mouse_button_1_pressed))
        self.canvas.bind(
            "<ButtonRelease-1>", any_args(controller.mouse_button_1_released)
        )
        self.canvas.bind(
            "<MouseWheel>", lambda event: controller.on_scroll(event.delta / 500)
        )

        self.tk.bind("<KeyPress-r>", any_args(controller.enter_rotate_mode))
        self.tk.bind("<KeyRelease-r>", any_args(controller.exit_rotate_mode))
        self.tk.bind("<Control-s>", any_args(controller.save))
        self.tk.bind("<Control-o>", any_args(controller.load))

        self._bar_canvas.save_btn.configure(command=controller.save)
        self._bar_canvas.load_btn.configure(command=controller.load)
        self._particle_tab.set_selected_btn.configure(command=controller.set_selected)
        self._particle_tab.set_all_btn.configure(command=controller.set_all)

    def _set_callbacks(self) -> None:
        self._bar_canvas.select_btn.configure(command=self._set_select_mode)
        self._bar_canvas.move_btn.configure(command=self._set_move_mode)
        self._bar_canvas.add_btn.configure(command=self._set_add_mode)
        self._bar_canvas.code_btn.configure(command=self._create_code_window)

        self._sim_tab.gravity_var.trace("w", any_args(self._set_gravity))
        self._sim_tab.air_res_var.trace("w", any_args(self._set_air_res))
        self._sim_tab.friction_var.trace("w", any_args(self._set_ground_friction))
        self._sim_tab.temp_var.trace("w", any_args(self._set_temperature))
        self._sim_tab.speed_var.trace("w", any_args(self._set_speed))
        self._sim_tab.fps_chk.configure(command=self._set_show_fps)
        self._sim_tab.num_chk.configure(command=self._set_show_num)
        self._sim_tab.links_chk.configure(command=self._set_show_links)
        self._sim_tab.top_chk.configure(command=self._set_top)
        self._sim_tab.bottom_chk.configure(command=self._set_bottom)
        self._sim_tab.left_chk.configure(command=self._set_left)
        self._sim_tab.right_chk.configure(command=self._set_right)
        self._sim_tab.grid_chk.configure(command=self._set_use_grid)
        self._sim_tab.delay_var.trace("w", any_args(self._set_min_spawn_delay))
        self._sim_tab.calculate_radii_diff_chk.configure(
            command=self._set_calculate_radii_diff
        )
        self._sim_tab.extra_btn.configure(command=self._create_extra_window)
        self._sim_tab.grid_res_x_var.trace("w", any_args(self._set_grid_x))
        self._sim_tab.grid_res_y_var.trace("w", any_args(self._set_grid_y))

        self._particle_tab.group_add_btn.configure(command=self._add_group)
        self._particle_tab.copy_selected_btn.configure(command=self._copy_from_selected)

    def _register_sim(self) -> None:
        self.sim.on_group_created.subscribe(self._create_group)

        self.sim.on_pause_toggle.subscribe(self._bar_canvas.set_paused)
        self._bar_canvas.on_pause_button_pressed.subscribe(
            lambda _: self.sim.toggle_paused()
        )
        self._bar_canvas.set_paused(self.sim.paused)
        self._bar_canvas.link_btn.configure(command=self.sim.link_selection)
        self._bar_canvas.unlink_btn.configure(command=self.sim.unlink_selection)
        self.sim.on_mouse_mode_changed.subscribe(self._bar_canvas.set_mode)
        self._bar_canvas.set_mode(self.sim.mouse_mode)

        self._sim_tab.gravity_var.set(self.sim.g)
        self._sim_tab.air_res_var.set(self.sim.air_res)
        self._sim_tab.friction_var.set(self.sim.ground_friction)
        self._sim_tab.temp_var.set(self.sim.temperature)
        self._sim_tab.speed_var.set(self.sim.speed)
        self._sim_tab.show_fps.set(self.sim.show_fps)
        self._sim_tab.show_num.set(self.sim.show_num)
        self._sim_tab.show_links.set(self.sim.show_links)
        self._sim_tab.top_bool.set(self.sim.top)
        self._sim_tab.bottom_bool.set(self.sim.bottom)
        self._sim_tab.left_bool.set(self.sim.left)
        self._sim_tab.right_bool.set(self.sim.right)
        self._sim_tab.delay_var.set(self.sim.min_spawn_delay)

        self._particle_tab.group_select_btn.configure(
            command=lambda: self.sim.select_group(self._particle_tab.groups_var.get())
        )
        groups = sorted(self.sim.groups)
        self._particle_tab.groups_entry["values"] = groups
        if groups:
            self._particle_tab.groups_entry.current(0)

        self._bind_sim_events()

    def register_sim(self, sim: SimulationState) -> None:
        self.sim = sim
        self._register_sim()

    def _set_air_res(self) -> None:
        self.sim.air_res = self._sim_tab.air_res_var.get()

    def _set_gravity(self) -> None:
        self.sim.g = self._sim_tab.gravity_var.get()

    def _set_ground_friction(self) -> None:
        self.sim.ground_friction = self._sim_tab.friction_var.get()

    def _set_temperature(self) -> None:
        self.sim.temperature = self._sim_tab.temp_var.get()

    def _set_speed(self) -> None:
        self.sim.speed = self._sim_tab.speed_var.get()

    def _set_show_fps(self) -> None:
        self.sim.show_fps = self._sim_tab.show_fps.get()

    def _set_show_num(self) -> None:
        self.sim.show_num = self._sim_tab.show_num.get()

    def _set_show_links(self) -> None:
        self.sim.show_links = self._sim_tab.show_links.get()

    def _set_top(self) -> None:
        self.sim.top = self._sim_tab.top_bool.get()

    def _set_bottom(self) -> None:
        self.sim.bottom = self._sim_tab.bottom_bool.get()

    def _set_left(self) -> None:
        self.sim.left = self._sim_tab.left_bool.get()

    def _set_right(self) -> None:
        self.sim.right = self._sim_tab.right_bool.get()

    def _set_use_grid(self) -> None:
        self.sim.use_grid = self._sim_tab.grid_bool.get()

    def _set_grid_x(self) -> None:
        self.sim.grid_res_x = self._sim_tab.grid_res_x_var.get()

    def _set_grid_y(self) -> None:
        self.sim.grid_res_y = self._sim_tab.grid_res_y_var.get()

    def _set_min_spawn_delay(self) -> None:
        self.sim.min_spawn_delay = self._sim_tab.delay_var.get()

    def _set_calculate_radii_diff(self) -> None:
        self.sim.calculate_radii_diff = self._sim_tab.calculate_radii_diff_bool.get()

    def _set_select_mode(self) -> None:
        self.sim.mouse_mode = "SELECT"

    def _set_move_mode(self) -> None:
        self.sim.mouse_mode = "MOVE"

    def _set_add_mode(self) -> None:
        self.sim.mouse_mode = "ADD"

    def _create_extra_window(self) -> None:
        self.extra_window = ExtraWindow(self.sim)

    def _create_code_window(self) -> None:
        self.code_window = CodeWindow()
        self.code_window.set_code(self.sim.code)
        self.code_window.on_exec.subscribe(self.sim.execute)
        self.code_window.on_save.subscribe(self.sim.set_code)

    def _add_group(self) -> None:
        name = self.sim.add_group()
        index = self._create_group_id(name)
        self._particle_tab.groups_entry.current(index)

    def _create_group_id(self, name: str) -> int:
        try:
            return self._particle_tab.groups_entry["values"].index(name)
        except ValueError:
            group_names: List[str] = list(self._particle_tab.groups_entry["values"])
            index = bisect.bisect(group_names, name)
            group_names.insert(index, name)
            self._particle_tab.groups_entry["values"] = group_names
            return index

    def _create_group(self, name: str) -> None:
        index = self._create_group_id(name)
        if not self._particle_tab.groups_entry.get():
            self._particle_tab.groups_entry.current(index)

    def get_sim_settings(self) -> SimGUISettings:
        return SimGUISettings(
            show_fps=self._sim_tab.show_fps.get(),
            show_num=self._sim_tab.show_num.get(),
            show_links=self._sim_tab.show_links.get(),
            grid_res_x=self._sim_tab.grid_res_x_var.get(),
            grid_res_y=self._sim_tab.grid_res_y_var.get(),
            delay=self._sim_tab.delay_var.get(),
        )

    def set_sim_settings(self, sim_settings: SimGUISettings) -> None:
        s = sim_settings

        self._sim_tab.show_fps.set(s.show_fps)
        self._sim_tab.show_num.set(s.show_num)
        self._sim_tab.show_links.set(s.show_links)
        self._sim_tab.grid_res_x_var.set(s.grid_res_x)
        self._sim_tab.grid_res_y_var.set(s.grid_res_y)
        self._sim_tab.delay_var.set(s.delay)

    def get_particle_factory(self) -> ParticleFactory:
        props = ParticleProperties(
            mass=self._particle_tab.mass_var.get(),
            bounciness=self._particle_tab.bounciness_var.get(),
            attract_r=self._particle_tab.attr_r_var.get(),
            repel_r=self._particle_tab.repel_r_var.get(),
            attraction_strength=self._particle_tab.attr_strength_var.get(),
            repulsion_strength=self._particle_tab.repel_strength_var.get(),
            link_attr_breaking_force=self._particle_tab.link_attr_break_var.get(),
            link_repel_breaking_force=self._particle_tab.link_repel_break_var.get(),
            collisions=self._particle_tab.do_collision_bool.get(),
            locked=self._particle_tab.locked_bool.get(),
            linked_group_particles=self._particle_tab.linked_group_bool.get(),
            group=self._particle_tab.groups_var.get(),
            separate_group=self._particle_tab.separate_group_bool.get(),
            gravity_mode=self._particle_tab.gravity_mode_bool.get(),
        )
        color = self._particle_tab._parse_color()
        if color is None:
            color = generate_random()
        factory = ParticleFactory(
            color=color,
            props=props,
            radius=self._particle_tab.radius_var.get(),
            velocity=(
                self._particle_tab.velocity_x_var.get(),
                self._particle_tab.velocity_y_var.get(),
            ),
        )
        return factory

    def _set_particle_properties(self, p: ParticleProperties) -> None:
        self._particle_tab.mass_var.set(p.mass)
        self._particle_tab.bounciness_var.set(p.bounciness)
        self._particle_tab.attr_r_var.set(p.attract_r)
        self._particle_tab.repel_r_var.set(p.repel_r)
        self._particle_tab.attr_strength_var.set(p.attraction_strength)
        self._particle_tab.repel_strength_var.set(p.repulsion_strength)
        self._particle_tab.link_attr_break_var.set(p.link_attr_breaking_force)
        self._particle_tab.link_repel_break_var.set(p.link_repel_breaking_force)
        self._particle_tab.do_collision_bool.set(p.collisions)
        self._particle_tab.locked_bool.set(p.locked)
        self._particle_tab.linked_group_bool.set(p.linked_group_particles)
        self._particle_tab.groups_var.set(p.group)
        self._particle_tab.separate_group_bool.set(p.separate_group)
        self._particle_tab.gravity_mode_bool.set(p.gravity_mode)

    def set_particle_settings(self, particle_settings: ParticleFactory) -> None:
        p = particle_settings
        self._particle_tab.color_var.set(str(p.color))
        self._set_particle_properties(p.props)
        self._particle_tab.velocity_x_var.set(p.velocity[0])
        self._particle_tab.velocity_y_var.set(p.velocity[1])
        self._particle_tab.radius_var.set(p.radius)

    @staticmethod
    def _part_to_dict(p: Particle) -> Dict[str, Any]:
        return {
            "radius_var": p.radius,
            "color_var": p.color,
            "mass_var": p.props.mass,
            "velocity_x_var": p.velocity[0],
            "velocity_y_var": p.velocity[1],
            "bounciness_var": p.props.bounciness,
            "do_collision_bool": p.props.collisions,
            "locked_bool": p.props.locked,
            "linked_group_bool": p.props.linked_group_particles,
            "attr_r_var": p.props.attract_r,
            "repel_r_var": p.props.repel_r,
            "attr_strength_var": p.props.attraction_strength,
            "repel_strength_var": p.props.repulsion_strength,
            "link_attr_break_var": p.props.link_attr_breaking_force,
            "link_repel_break_var": p.props.link_repel_breaking_force,
            "groups_var": p.props.group,
            "separate_group_bool": p.props.separate_group,
            "gravity_mode_bool": p.props.gravity_mode,
        }

    def _copy_from_selected(self) -> None:
        particle_settings: Optional[Dict[str, Any]] = None
        for p in self.sim.selection:
            variable_names = self._part_to_dict(p)
            if particle_settings is None:
                particle_settings = variable_names
            to_remove: Set[str] = set()
            for gui_attr, part_val in particle_settings.items():
                variable: tk.Variable = getattr(self._particle_tab, gui_attr)
                if variable_names[gui_attr] == part_val:
                    variable.set(part_val)
                else:
                    self._set_variable_default(variable)
                    to_remove.add(gui_attr)
            for gui_attr in to_remove:
                del particle_settings[gui_attr]

    def _set_variable_default(self, variable: tk.Variable) -> None:
        if isinstance(variable, tk.BooleanVar):
            variable.set(False)
        elif isinstance(variable, tk.IntVar):
            variable.set(0)
        elif isinstance(variable, tk.DoubleVar):
            variable.set(0.0)
        elif isinstance(variable, tk.StringVar):
            variable.set("")
        else:
            raise NotImplementedError(f"Unexpected variable: {type(variable)}")

    def _update(self) -> None:
        if self.code_window is not None:
            self.code_window.tk.update()
        if self.extra_window is not None:
            self.extra_window.update()

        self.tk.update()

    def update(
        self,
        image: npt.NDArray[np.uint8],
        fps: Optional[float] = None,
    ) -> None:
        while self.sim.errors:
            error = self.sim.errors.popleft()
            messagebox.showerror(error.name, str(error.exception))
        photo = ImageTk.PhotoImage(image=Image.fromarray(image.astype(np.uint8)))

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        if self.sim.show_fps and fps is not None:
            self.canvas.create_text(
                10,
                10,
                text=f"FPS: {round(fps, 2)}",
                anchor="nw",
                font=("Helvetica", 9, "bold"),
            )
        if self.sim.show_num:
            self.canvas.create_text(
                10,
                25,
                text=f"Particles: {len(self.sim.particles)}",
                anchor="nw",
                font=("Helvetica", 9, "bold"),
            )

        self._update()

    def _destroy(self) -> None:
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.sim.running = False
            self.tk.destroy()
