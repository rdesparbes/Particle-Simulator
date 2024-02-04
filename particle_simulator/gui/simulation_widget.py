import os
import tkinter as tk
from tkinter import ttk

from particle_simulator.gui.variable import get_double_var, get_int_var


class SimulationWidget(ttk.Frame):
    def __init__(self, master: tk.Widget, resource_path: str) -> None:
        super().__init__(master, relief="flat")
        self._sim_tab_canvas = tk.Canvas(self)
        self._sim_tab_canvas.pack(expand=True, fill="both")

        tk.Label(self._sim_tab_canvas, text="Gravity:", font=("helvetica", 8)).place(
            x=7, y=20, anchor="nw"
        )
        self._gravity_entry = tk.Spinbox(
            self._sim_tab_canvas,
            width=7,
            from_=0,
            to=1,
            increment=0.1,
        )
        self.gravity_var = get_double_var(self._gravity_entry)
        self._gravity_entry.place(x=100, y=20)

        tk.Label(
            self._sim_tab_canvas, text="Air Resistance:", font=("helvetica", 8)
        ).place(x=7, y=50, anchor="nw")
        self._air_res_entry = tk.Spinbox(
            self._sim_tab_canvas,
            width=7,
            from_=0,
            to=1,
            increment=0.01,
        )
        self.air_res_var = get_double_var(self._air_res_entry, 0.0)
        self._air_res_entry.place(x=100, y=50)

        tk.Label(
            self._sim_tab_canvas, text="Ground Friction:", font=("helvetica", 8)
        ).place(x=7, y=80, anchor="nw")
        self._friction_entry = tk.Spinbox(
            self._sim_tab_canvas,
            width=7,
            from_=0,
            to=1,
            increment=0.01,
        )
        self.friction_var = get_double_var(self._friction_entry)
        self._friction_entry.place(x=100, y=80)

        self._temp_sc = tk.Scale(
            self._sim_tab_canvas,
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
        )
        self.temp_var = get_double_var(self._temp_sc)
        self._temp_sc.place(x=100, y=153, anchor="center")
        tk.Label(
            self._sim_tab_canvas, text="Temperature:", font=("helvetica", 8)
        ).place(x=7, y=110, anchor="nw")

        self._speed_sc = tk.Scale(
            self._sim_tab_canvas,
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
        )
        self.speed_var = get_double_var(self._speed_sc)
        self._speed_sc.place(x=100, y=233, anchor="center")
        tk.Label(
            self._sim_tab_canvas, text="Simulation Speed:", font=("helvetica", 8)
        ).place(x=7, y=190, anchor="nw")

        self.show_fps = tk.BooleanVar(self)
        self.fps_chk = tk.Checkbutton(
            self._sim_tab_canvas,
            text="Display FPS",
            font=("helvetica", 8),
            variable=self.show_fps,
        )
        self.fps_chk.place(x=10, y=260, anchor="nw")

        self.show_num = tk.BooleanVar(self)
        self.num_chk = tk.Checkbutton(
            self._sim_tab_canvas,
            text="Display # Particles",
            font=("helvetica", 8),
            variable=self.show_num,
        )
        self.num_chk.place(x=10, y=280, anchor="nw")

        self.show_links = tk.BooleanVar(self)
        self.links_chk = tk.Checkbutton(
            self._sim_tab_canvas,
            text="Display links",
            font=("helvetica", 8),
            variable=self.show_links,
        )
        self.links_chk.place(x=10, y=300, anchor="nw")

        self._sim_tab_canvas.create_text(
            100, 335, text="Blocking Edges", font=("helvetica", 9), anchor="center"
        )
        self._sim_tab_canvas.create_line(10, 345, 190, 345, fill="grey50")

        self.top_bool = tk.BooleanVar(self)
        self.top_chk = tk.Checkbutton(
            self._sim_tab_canvas,
            text="top",
            font=("helvetica", 8),
            variable=self.top_bool,
        )
        self.top_chk.place(x=30, y=350, anchor="nw")

        self.bottom_bool = tk.BooleanVar(self)
        self.bottom_chk = tk.Checkbutton(
            self._sim_tab_canvas,
            text="bottom",
            font=("helvetica", 8),
            variable=self.bottom_bool,
        )
        self.bottom_chk.place(x=110, y=350, anchor="nw")

        self.left_bool = tk.BooleanVar(self)
        self.left_chk = tk.Checkbutton(
            self._sim_tab_canvas,
            text="left",
            font=("helvetica", 8),
            variable=self.left_bool,
        )
        self.left_chk.place(x=30, y=370, anchor="nw")

        self.right_bool = tk.BooleanVar(self)
        self.right_chk = tk.Checkbutton(
            self._sim_tab_canvas,
            text="right",
            font=("helvetica", 8),
            variable=self.right_bool,
        )
        self.right_chk.place(x=110, y=370, anchor="nw")

        self._sim_tab_canvas.create_text(
            100, 405, text="Optimization", font=("helvetica", 9), anchor="center"
        )
        self._sim_tab_canvas.create_line(10, 415, 190, 415, fill="grey50")

        self.grid_bool = tk.BooleanVar(self, True)
        self.grid_chk = tk.Checkbutton(
            self._sim_tab_canvas,
            text="Use Grid",
            font=("helvetica", 8),
            variable=self.grid_bool,
        )
        self.grid_chk.place(x=10, y=425, anchor="nw")

        tk.Label(self._sim_tab_canvas, text="Grid-Res:", font=("helvetica", 8)).place(
            x=7, y=455, anchor="nw"
        )
        tk.Label(self._sim_tab_canvas, text="X:", font=("helvetica", 8)).place(
            x=60, y=455, anchor="nw"
        )
        self._grid_res_x = tk.Spinbox(
            self._sim_tab_canvas,
            width=7,
            from_=1,
            to=200,
            increment=1,
        )
        self.grid_res_x_var = get_int_var(self._grid_res_x)
        self._grid_res_x.place(x=80, y=455)

        tk.Label(self._sim_tab_canvas, text="Y:", font=("helvetica", 8)).place(
            x=60, y=480, anchor="nw"
        )
        self._grid_res_y = tk.Spinbox(
            self._sim_tab_canvas,
            width=7,
            from_=1,
            to=200,
            increment=1,
        )
        self.grid_res_y_var = get_int_var(self._grid_res_y)
        self._grid_res_y.place(x=80, y=480)

        self._sim_tab_canvas.create_text(
            100, 515, text="Extra", font=("helvetica", 9), anchor="center"
        )
        self._sim_tab_canvas.create_line(10, 525, 190, 525, fill="grey50")

        tk.Label(
            self._sim_tab_canvas, text="Min Spawn-Delay:", font=("helvetica", 8)
        ).place(x=7, y=533, anchor="nw")
        self._delay_entry = tk.Spinbox(
            self._sim_tab_canvas,
            width=7,
            from_=0,
            to=1,
            increment=0.01,
        )
        self.delay_var = get_double_var(self._delay_entry)
        self._delay_entry.place(x=100, y=533)

        self.calculate_radii_diff_bool = tk.BooleanVar(self, False)
        self.calculate_radii_diff_chk = tk.Checkbutton(
            self._sim_tab_canvas,
            text="Better Radii-Calculation",
            font=("helvetica", 8),
            variable=self.calculate_radii_diff_bool,
        )
        self.calculate_radii_diff_chk.place(x=7, y=553, anchor="nw")

        self._extra_img = tk.PhotoImage(
            file=os.path.join(resource_path, "Assets/dots.gif"), master=self
        ).subsample(11, 11)
        self.extra_btn = tk.Button(
            self._sim_tab_canvas,
            image=self._extra_img,
            cursor="hand2",
            bg="#F0F0F0",
            activebackground="#F0F0F0",
            relief="flat",
        )
        self.extra_btn.place(x=7, y=580)
