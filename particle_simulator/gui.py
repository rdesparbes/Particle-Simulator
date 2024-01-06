import os
import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
from typing import Literal, Sequence, Dict, Any, Optional, Union, Tuple

import numpy as np
import numpy.typing as npt
from PIL import ImageTk, Image

from .code_window import CodeWindow
from .error import Error
from .extra_window import ExtraWindow
from .particle_data import ParticleData
from .particle_factory import ParticleFactory
from .save_manager import SaveManager
from .sim_gui_settings import SimGUISettings
from .simulation_state import SimulationState

CANVAS_X = 0  # The X coordinate of the top-left corner of the canvas
CANVAS_Y = 30  # The Y coordinate of the top-left corner of the canvas


class GUI:
    def __init__(
        self, sim: SimulationState, title: str, gridres: Tuple[int, int]
    ) -> None:
        width = sim.width
        height = sim.height
        self.path = os.path.split(os.path.abspath(__file__))[0]

        self.tk = tk.Tk()
        self.tk.title(title)
        self.tk.resizable(False, False)
        self.tk.protocol("WM_DELETE_WINDOW", self.destroy)
        self.gui_canvas = tk.Canvas(self.tk, width=width + 200, height=height + 30)
        self.gui_canvas.pack()
        self.canvas = tk.Canvas(self.tk, width=width, height=height)
        self.canvas.place(x=CANVAS_X, y=CANVAS_Y)

        self.code_window: Optional[CodeWindow] = None
        self.extra_window: Optional[ExtraWindow] = None

        self.toolbar = self.gui_canvas.create_rectangle(0, 0, width, 30, fill="#1f3333")
        self.gui_canvas.create_line(80, 0, 80, 30, fill="grey30")

        self.play_photo = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/play.gif"), master=self.tk
        ).subsample(8, 8)
        self.pause_photo = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/pause.gif"), master=self.tk
        ).subsample(7, 7)
        self.pause_button = tk.Button(
            self.gui_canvas,
            image=self.pause_photo,
            cursor="hand2",
            border="0",
            bg="#1f3333",
            activebackground="#1f3333",
        )
        self.pause_button.place(x=40, y=16, anchor="center")

        x = 125
        self.select_img = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/select.gif"), master=self.tk
        ).subsample(57, 57)
        self.select_btn = tk.Button(
            self.tk,
            image=self.select_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._set_select_mode,
        )
        self.select_btn.place(x=x, y=16, anchor="center")
        self.select_rect = self.gui_canvas.create_rectangle(
            x - 12, 3, x + 12, 27, outline="blue", state="hidden"
        )

        x = 165
        self.move_img = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/move.gif"), master=self.tk
        ).subsample(42, 42)
        self.move_btn = tk.Button(
            self.tk,
            image=self.move_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._set_move_mode,
        )
        self.move_btn.place(x=x, y=16, anchor="center")
        self.move_rect = self.gui_canvas.create_rectangle(
            x - 12, 3, x + 12, 27, outline="blue"
        )

        x = 205
        self.add_img = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/add.gif"), master=self.tk
        ).subsample(36, 36)
        self.add_btn = tk.Button(
            self.tk,
            image=self.add_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._set_add_mode,
        )
        self.add_btn.place(x=x, y=15, anchor="center")
        self.add_rect = self.gui_canvas.create_rectangle(
            x - 13, 3, x + 11, 27, outline="blue", state="hidden"
        )

        self.link_btn = tk.Button(
            self.tk,
            text="LINK",
            font=("Helvetica", 8, "bold"),
            cursor="hand2",
            fg="khaki1",
            bg="#1f3333",
            activebackground="#1f3333",
            relief="flat",
        )
        self.link_btn.place(x=250, y=16, anchor="center")

        self.unlink_btn = tk.Button(
            self.tk,
            text="UNLINK",
            font=("Helvetica", 8, "bold"),
            cursor="hand2",
            fg="blue violet",
            bg="#1f3333",
            activebackground="#1f3333",
            relief="flat",
        )
        self.unlink_btn.place(x=300, y=16, anchor="center")

        self.save_img = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/save.gif"), master=self.tk
        ).subsample(28, 28)
        self.save_manager = SaveManager(file_location=os.path.dirname(self.path))
        self.save_btn = tk.Button(
            self.tk,
            image=self.save_img,
            cursor="hand2",
            bg="#1f3333",
            activebackground="#1f3333",
            relief="flat",
        )
        self.save_btn.place(x=width - 110, y=16, anchor="center")

        self.load_img = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/load.gif"), master=self.tk
        ).subsample(32, 32)
        self.load_btn = tk.Button(
            self.tk,
            image=self.load_img,
            cursor="hand2",
            bg="#1f3333",
            activebackground="#1f3333",
            relief="flat",
        )
        self.load_btn.place(x=width - 75, y=16, anchor="center")

        self.code_img = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/code.gif"), master=self.tk
        ).subsample(13, 13)
        self.code_btn = tk.Button(
            self.tk,
            image=self.code_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._create_code_window,
        )
        self.code_btn.place(x=width - 25, y=16, anchor="center")

        # layout sidebar-GUI
        self.tabControl = ttk.Notebook(self.tk)
        self.tab1 = ttk.Frame(self.tabControl, relief="flat")
        self.tabControl.add(self.tab1, text="Sim-Settings")
        self.tab2 = ttk.Frame(
            self.tabControl, relief="flat", width=200, height=height + 30
        )
        self.tabControl.add(self.tab2, text="Particle-Settings")
        self.tabControl.place(x=width, y=0)

        # layout self.tab1
        self.tab1_canvas = tk.Canvas(self.tab1, width=200, height=height)
        self.tab1_canvas.pack()

        tk.Label(self.tab1, text="Gravity:", font=("helvetica", 8)).place(
            x=7, y=20, anchor="nw"
        )
        self.gravity_entry = tk.Spinbox(
            self.tab1,
            width=7,
            from_=0,
            to=1,
            increment=0.1,
            command=self._set_gravity,
        )
        self.gravity_entry.place(x=100, y=20)

        tk.Label(self.tab1, text="Air Resistance:", font=("helvetica", 8)).place(
            x=7, y=50, anchor="nw"
        )
        self.air_res_entry = tk.Spinbox(
            self.tab1,
            width=7,
            from_=0,
            to=1,
            increment=0.01,
            command=self._set_air_res,
        )
        self.air_res_entry.place(x=100, y=50)

        tk.Label(self.tab1, text="Ground Friction:", font=("helvetica", 8)).place(
            x=7, y=80, anchor="nw"
        )
        self.friction_entry = tk.Spinbox(
            self.tab1,
            width=7,
            from_=0,
            to=1,
            increment=0.01,
            command=self._set_ground_friction,
        )
        self.friction_entry.place(x=100, y=80)

        self.temp_sc = tk.Scale(
            self.tab1,
            from_=0,
            to=5,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            length=175,
            width=10,
            tickinterval=1,
            fg="gray65",
            activebackground="midnight blue",
            cursor="hand2",
            command=self._set_temperature,
        )
        self.temp_sc.place(x=100, y=153, anchor="center")
        tk.Label(self.tab1, text="Temperature:", font=("helvetica", 8)).place(
            x=7, y=110, anchor="nw"
        )

        self.speed_sc = tk.Scale(
            self.tab1,
            from_=0,
            to=3,
            orient=tk.HORIZONTAL,
            resolution=0.01,
            length=175,
            width=10,
            tickinterval=1,
            fg="gray65",
            activebackground="midnight blue",
            cursor="hand2",
            command=self._set_speed,
        )
        self.speed_sc.place(x=100, y=233, anchor="center")
        tk.Label(self.tab1, text="Simulation Speed:", font=("helvetica", 8)).place(
            x=7, y=190, anchor="nw"
        )

        self.show_fps = tk.BooleanVar(self.tk)
        self.fps_chk = tk.Checkbutton(
            self.tab1,
            text="Display FPS",
            font=("helvetica", 8),
            variable=self.show_fps,
            command=self._set_show_fps,
        )
        self.fps_chk.place(x=10, y=260, anchor="nw")

        self.show_num = tk.BooleanVar(self.tk)
        self.num_chk = tk.Checkbutton(
            self.tab1,
            text="Display # Particles",
            font=("helvetica", 8),
            variable=self.show_num,
            command=self._set_show_num,
        )
        self.num_chk.place(x=10, y=280, anchor="nw")

        self.show_links = tk.BooleanVar(self.tk)
        self.links_chk = tk.Checkbutton(
            self.tab1,
            text="Display links",
            font=("helvetica", 8),
            variable=self.show_links,
            command=self._set_show_links,
        )
        self.links_chk.place(x=10, y=300, anchor="nw")

        self.tab1_canvas.create_text(
            100, 335, text="Blocking Edges", font=("helvetica", 9), anchor="center"
        )
        self.tab1_canvas.create_line(10, 345, 190, 345, fill="grey50")

        self.top_bool = tk.BooleanVar(self.tk)
        self.top_chk = tk.Checkbutton(
            self.tab1,
            text="top",
            font=("helvetica", 8),
            variable=self.top_bool,
            command=self._set_top,
        )
        self.top_chk.place(x=30, y=350, anchor="nw")

        self.bottom_bool = tk.BooleanVar(self.tk)
        self.bottom_chk = tk.Checkbutton(
            self.tab1,
            text="bottom",
            font=("helvetica", 8),
            variable=self.bottom_bool,
            command=self._set_bottom,
        )
        self.bottom_chk.place(x=110, y=350, anchor="nw")

        self.left_bool = tk.BooleanVar(self.tk)
        self.left_chk = tk.Checkbutton(
            self.tab1,
            text="left",
            font=("helvetica", 8),
            variable=self.left_bool,
            command=self._set_left,
        )
        self.left_chk.place(x=30, y=370, anchor="nw")

        self.right_bool = tk.BooleanVar(self.tk)
        self.right_chk = tk.Checkbutton(
            self.tab1,
            text="right",
            font=("helvetica", 8),
            variable=self.right_bool,
            command=self._set_right,
        )
        self.right_chk.place(x=110, y=370, anchor="nw")

        self.tab1_canvas.create_text(
            100, 405, text="Optimization", font=("helvetica", 9), anchor="center"
        )
        self.tab1_canvas.create_line(10, 415, 190, 415, fill="grey50")

        self.grid_bool = tk.BooleanVar(self.tk, True)
        self.grid_chk = tk.Checkbutton(
            self.tab1,
            text="Use Grid",
            font=("helvetica", 8),
            variable=self.grid_bool,
            command=self._set_use_grid,
        )
        self.grid_chk.place(x=10, y=425, anchor="nw")

        tk.Label(self.tab1, text="Grid-Res:", font=("helvetica", 8)).place(
            x=7, y=455, anchor="nw"
        )
        tk.Label(self.tab1, text="X:", font=("helvetica", 8)).place(
            x=60, y=455, anchor="nw"
        )
        self.grid_res_x_value = tk.IntVar(self.tk, value=gridres[0])
        self.grid_res_x = tk.Spinbox(
            self.tab1,
            width=7,
            from_=1,
            to=200,
            increment=1,
            textvariable=self.grid_res_x_value,
        )
        self.grid_res_x.place(x=80, y=455)

        tk.Label(self.tab1, text="Y:", font=("helvetica", 8)).place(
            x=60, y=480, anchor="nw"
        )
        self.grid_res_y_value = tk.IntVar(self.tk, value=gridres[1])
        self.grid_res_y = tk.Spinbox(
            self.tab1,
            width=7,
            from_=1,
            to=200,
            increment=1,
            textvariable=self.grid_res_y_value,
        )
        self.grid_res_y.place(x=80, y=480)

        self.tab1_canvas.create_text(
            100, 515, text="Extra", font=("helvetica", 9), anchor="center"
        )
        self.tab1_canvas.create_line(10, 525, 190, 525, fill="grey50")

        tk.Label(self.tab1, text="Min Spawn-Delay:", font=("helvetica", 8)).place(
            x=7, y=533, anchor="nw"
        )
        self.delay_entry = tk.Spinbox(
            self.tab1,
            width=7,
            from_=0,
            to=1,
            increment=0.01,
            command=self._set_min_spawn_delay,
        )
        self.delay_entry.place(x=100, y=533)

        self.calculate_radii_diff_bool = tk.BooleanVar(self.tk, False)
        self.calculate_radii_diff_chk = tk.Checkbutton(
            self.tab1,
            text="Better Radii-Calculation",
            font=("helvetica", 8),
            variable=self.calculate_radii_diff_bool,
            command=self._set_calculate_radii_diff,
        )
        self.calculate_radii_diff_chk.place(x=7, y=553, anchor="nw")

        self.extra_img = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/dots.gif"), master=self.tk
        ).subsample(11, 11)
        self.extra_btn = tk.Button(
            self.tab1,
            image=self.extra_img,
            cursor="hand2",
            bg="#F0F0F0",
            activebackground="#F0F0F0",
            relief="flat",
            command=self._create_extra_window,
        )
        self.extra_btn.place(x=7, y=580)

        # layout tab2
        self.tab2_canvas = tk.Canvas(self.tab2, width=200, height=height)
        self.tab2_canvas.pack()

        tk.Label(self.tab2, text="Radius:", font=("helvetica", 8)).place(
            x=7, y=20, anchor="nw"
        )
        self.radius_entry = tk.Spinbox(self.tab2, width=7, from_=1, to=300, increment=1)
        self.radius_entry.delete(0, tk.END)
        self.radius_entry.insert(0, "scroll")
        self.radius_entry.place(x=100, y=20)

        tk.Label(self.tab2, text="Color:", font=("helvetica", 8)).place(
            x=7, y=50, anchor="nw"
        )
        self.color_var = tk.StringVar(self.tk, "random")
        self.color_entry = tk.Entry(self.tab2, width=8, textvariable=self.color_var)
        self.color_entry.place(x=100, y=50)
        self.color_var.trace("w", self.change_color_entry)

        self.part_color_rect = self.tab2_canvas.create_rectangle(
            160,
            48,
            180,
            68,
            fill="#ffffff",
            activeoutline="red",
            tags="part_color_rect",
        )
        self.tab2_canvas.tag_bind("part_color_rect", "<Button-1>", self.ask_color_entry)

        tk.Label(self.tab2, text="Mass:", font=("helvetica", 8)).place(
            x=7, y=80, anchor="nw"
        )
        self.mass_entry = tk.Spinbox(
            self.tab2, width=7, from_=0.1, to=100, increment=0.1
        )
        self.mass_entry.delete(0, tk.END)
        self.mass_entry.insert(0, "1")
        self.mass_entry.place(x=100, y=80)

        tk.Label(self.tab2, text="Bounciness:", font=("helvetica", 8)).place(
            x=7, y=110, anchor="nw"
        )
        self.bounciness_entry = tk.Spinbox(
            self.tab2, width=7, from_=0, to=1, increment=0.1
        )
        self.bounciness_entry.delete(0, tk.END)
        self.bounciness_entry.insert(0, "0.7")
        self.bounciness_entry.place(x=100, y=110)

        tk.Label(self.tab2, text="Velocity:", font=("helvetica", 8)).place(
            x=7, y=140, anchor="nw"
        )
        tk.Label(self.tab2, text="X:", font=("helvetica", 8)).place(
            x=60, y=140, anchor="nw"
        )
        self.velocity_x_entry = tk.Spinbox(
            self.tab2, width=7, from_=0, to=1, increment=0.1
        )
        self.velocity_x_entry.delete(0, tk.END)
        self.velocity_x_entry.insert(0, "0")
        self.velocity_x_entry.place(x=100, y=140)
        tk.Label(self.tab2, text="Y:", font=("helvetica", 8)).place(
            x=60, y=162, anchor="nw"
        )
        self.velocity_y_entry = tk.Spinbox(
            self.tab2, width=7, from_=-5, to=5, increment=0.1
        )
        self.velocity_y_entry.delete(0, tk.END)
        self.velocity_y_entry.insert(0, "0")
        self.velocity_y_entry.place(x=100, y=162)

        self.locked_bool = tk.BooleanVar(self.tk, False)
        self.locked_chk = tk.Checkbutton(
            self.tab2, text="Locked", font=("helvetica", 8), variable=self.locked_bool
        )
        self.locked_chk.place(x=7, y=190, anchor="nw")

        self.do_collision_bool = tk.BooleanVar(self.tk, False)
        self.do_collision_chk = tk.Checkbutton(
            self.tab2,
            text="Check Collisions",
            font=("helvetica", 8),
            variable=self.do_collision_bool,
        )
        self.do_collision_chk.place(x=7, y=210, anchor="nw")

        tk.Label(self.tab2, text="Attraction-radius:", font=("helvetica", 8)).place(
            x=7, y=250, anchor="nw"
        )
        self.attr_r_entry = tk.Spinbox(
            self.tab2, width=7, from_=-1, to=500, increment=1
        )
        self.attr_r_entry.delete(0, tk.END)
        self.attr_r_entry.insert(0, "-1")
        self.attr_r_entry.place(x=100, y=250)

        tk.Label(self.tab2, text="Attr-strength:", font=("helvetica", 8)).place(
            x=7, y=273, anchor="nw"
        )
        self.attr_strength_entry = tk.Spinbox(
            self.tab2, width=7, from_=0, to=50, increment=0.1
        )
        self.attr_strength_entry.delete(0, tk.END)
        self.attr_strength_entry.insert(0, "0.5")
        self.attr_strength_entry.place(x=100, y=273)

        self.gravity_mode_bool = tk.BooleanVar(self.tk, False)
        self.gravity_mode_chk = tk.Checkbutton(
            self.tab2,
            text="Gravity-Mode",
            font=("helvetica", 7),
            variable=self.gravity_mode_bool,
        )
        self.gravity_mode_chk.place(x=7, y=290, anchor="nw")

        tk.Label(self.tab2, text="Repulsion-radius:", font=("helvetica", 8)).place(
            x=7, y=313, anchor="nw"
        )
        self.repel_r_entry = tk.Spinbox(
            self.tab2, width=7, from_=0, to=500, increment=1
        )
        self.repel_r_entry.delete(0, tk.END)
        self.repel_r_entry.insert(0, "10")
        self.repel_r_entry.place(x=100, y=323)

        tk.Label(self.tab2, text="Repel-strength:", font=("helvetica", 8)).place(
            x=7, y=336, anchor="nw"
        )
        self.repel_strength_entry = tk.Spinbox(
            self.tab2, width=7, from_=0, to=50, increment=0.1
        )
        self.repel_strength_entry.delete(0, tk.END)
        self.repel_strength_entry.insert(0, "1")
        self.repel_strength_entry.place(x=100, y=346)

        self.linked_group_bool = tk.BooleanVar(self.tk, True)
        self.linked_group_chk = tk.Checkbutton(
            self.tab2,
            text="Linked to group-particles",
            font=("helvetica", 8),
            variable=self.linked_group_bool,
        )
        self.linked_group_chk.place(x=7, y=376, anchor="nw")

        tk.Label(self.tab2, text="Link-breaking-force:", font=("helvetica", 8)).place(
            x=7, y=400, anchor="nw"
        )
        tk.Label(self.tab2, text="Attr:", font=("helvetica", 8)).place(
            x=7, y=420, anchor="nw"
        )
        self.link_attr_break_entry = tk.Spinbox(
            self.tab2, width=5, from_=0, to=5000, increment=0.1
        )
        self.link_attr_break_entry.delete(0, tk.END)
        self.link_attr_break_entry.insert(0, "-1")
        self.link_attr_break_entry.place(x=40, y=420)
        tk.Label(self.tab2, text="Repel:", font=("helvetica", 8)).place(
            x=100, y=420, anchor="nw"
        )
        self.link_repel_break_entry = tk.Spinbox(
            self.tab2, width=5, from_=0, to=5000, increment=0.1
        )
        self.link_repel_break_entry.delete(0, tk.END)
        self.link_repel_break_entry.insert(0, "-1")
        self.link_repel_break_entry.place(x=140, y=420)

        tk.Label(self.tab2, text="Particle-group:", font=("helvetica", 8)).place(
            x=7, y=450, anchor="nw"
        )
        self.group_indices = [1]
        self.groups_entry = ttk.Combobox(self.tab2, width=10, values=["group1"])
        self.groups_entry.current(0)
        self.groups_entry.place(x=10, y=470, anchor="nw")

        self.group_add_btn = tk.Button(
            self.tab2,
            text="+",
            font=("Helvetica", 15, "bold"),
            cursor="hand2",
            fg="grey14",
            bg="#F0F0F0",
            activebackground="#F0F0F0",
            relief="flat",
            width=1,
            command=self.add_group,
        )
        self.group_add_btn.place(x=105, y=480, anchor="center")

        self.select_img2 = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/select2.gif"), master=self.tk
        ).subsample(54, 54)
        self.group_select_btn = tk.Button(
            self.tab2,
            image=self.select_img2,
            cursor="hand2",
            bg="#F0F0F0",
            activebackground="#F0F0F0",
            relief="flat",
        )
        self.group_select_btn.place(x=123, y=480, anchor="center")

        self.separate_group_bool = tk.BooleanVar(self.tk, False)
        self.separate_group_chk = tk.Checkbutton(
            self.tab2,
            text="Separate Group",
            font=("helvetica", 8),
            variable=self.separate_group_bool,
        )
        self.separate_group_chk.place(x=10, y=495, anchor="nw")

        self.copy_selected_btn = tk.Button(
            self.tab2,
            text="Copy from selected",
            bg="light coral",
            command=self._copy_from_selected,
        )
        self.copy_selected_btn.place(x=15, y=height - 65)
        self.set_selected_btn = tk.Button(
            self.tab2,
            text="Set Selected",
            bg="light green",
        )
        self.set_selected_btn.place(x=15, y=height - 30)
        self.set_all_btn = tk.Button(self.tab2, text="Set All", bg="light blue")
        self.set_all_btn.place(x=95, y=height - 30)

        self._register_sim(sim)
        self.sim = sim

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
