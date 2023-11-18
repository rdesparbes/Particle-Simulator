import os
import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
from typing import Literal

from .code_window import CodeWindow
from .extra_window import ExtraWindow

Mode = Literal["SELECT", "MOVE", "ADD"]


class GUI:
    def __init__(self, sim, title, gridres):
        self.sim = sim
        self.path = os.path.split(os.path.abspath(__file__))[0]

        self.tk = tk.Tk()
        self.tk.title(title)
        self.tk.resizable(False, False)
        self.tk.protocol("WM_DELETE_WINDOW", self.destroy)
        self.gui_canvas = tk.Canvas(
            self.tk, width=self.sim.width + 200, height=self.sim.height + 30
        )
        self.gui_canvas.pack()
        self.canvas = tk.Canvas(self.tk, width=self.sim.width, height=self.sim.height)
        self.canvas.place(x=0, y=30)

        self.code_window = None
        self.extra_window = None

        self.toolbar = self.gui_canvas.create_rectangle(
            0, 0, self.sim.width, 30, fill="#1f3333"
        )
        self.gui_canvas.create_line(80, 0, 80, 30, fill="grey30")

        self.play_photo = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/play.gif"), master=self.tk
        ).subsample(8, 8)
        self.pause_photo = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/pause.gif"), master=self.tk
        ).subsample(7, 7)
        self.pause_button = tk.Button(
            self.gui_canvas,
            image=self.play_photo if self.sim.paused else self.pause_photo,
            cursor="hand2",
            border="0",
            bg="#1f3333",
            activebackground="#1f3333",
            command=self.sim.toggle_paused,
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
            command=lambda: self.sim.change_mode("SELECT"),
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
            command=lambda: self.sim.change_mode("MOVE"),
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
            command=lambda: self.sim.change_mode("ADD"),
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
            command=lambda: self.sim.link_selection(),
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
            command=lambda: self.sim.unlink_selection(),
        )
        self.unlink_btn.place(x=300, y=16, anchor="center")

        self.save_img = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/save.gif"), master=self.tk
        ).subsample(28, 28)
        self.save_btn = tk.Button(
            self.tk,
            image=self.save_img,
            cursor="hand2",
            bg="#1f3333",
            activebackground="#1f3333",
            relief="flat",
            command=lambda: self.sim.save_manager.save(),
        )
        self.save_btn.place(x=self.sim.width - 110, y=16, anchor="center")

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
            command=lambda: self.sim.save_manager.load(),
        )
        self.load_btn.place(x=self.sim.width - 75, y=16, anchor="center")

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
        self.code_btn.place(x=self.sim.width - 25, y=16, anchor="center")

        # layout sidebar-GUI
        self.tabControl = ttk.Notebook(self.tk)
        self.tab1 = ttk.Frame(self.tabControl, relief="flat")
        self.tabControl.add(self.tab1, text="Sim-Settings")
        self.tab2 = ttk.Frame(
            self.tabControl, relief="flat", width=200, height=self.sim.height + 30
        )
        self.tabControl.add(self.tab2, text="Particle-Settings")
        self.tabControl.place(x=self.sim.width, y=0)

        # layout self.tab1
        self.tab1_canvas = tk.Canvas(self.tab1, width=200, height=self.sim.height)
        self.tab1_canvas.pack()

        tk.Label(self.tab1, text="Gravity:", font=("helvetica", 8)).place(
            x=7, y=20, anchor="nw"
        )
        self.gravity_entry = tk.Spinbox(
            self.tab1, width=7, from_=0, to=1, increment=0.1
        )
        self.gravity_entry.delete(0, tk.END)
        self.gravity_entry.insert(0, self.sim.g)
        self.gravity_entry.place(x=100, y=20)

        tk.Label(self.tab1, text="Air Resistance:", font=("helvetica", 8)).place(
            x=7, y=50, anchor="nw"
        )
        self.air_res_entry = tk.Spinbox(
            self.tab1, width=7, from_=0, to=1, increment=0.01
        )
        self.air_res_entry.delete(0, tk.END)
        self.air_res_entry.insert(0, self.sim.air_res)
        self.air_res_entry.place(x=100, y=50)

        tk.Label(self.tab1, text="Ground Friction:", font=("helvetica", 8)).place(
            x=7, y=80, anchor="nw"
        )
        self.friction_entry = tk.Spinbox(
            self.tab1, width=7, from_=0, to=1, increment=0.01
        )
        self.friction_entry.delete(0, tk.END)
        self.friction_entry.insert(0, self.sim.ground_friction)
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
        )
        self.temp_sc.set(self.sim.temperature)
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
        )
        self.speed_sc.set(1)
        self.speed_sc.place(x=100, y=233, anchor="center")
        tk.Label(self.tab1, text="Simulation Speed:", font=("helvetica", 8)).place(
            x=7, y=190, anchor="nw"
        )

        self.show_fps = tk.BooleanVar(self.tk, True)
        self.fps_chk = tk.Checkbutton(
            self.tab1, text="Display FPS", font=("helvetica", 8), variable=self.show_fps
        )
        self.fps_chk.place(x=10, y=260, anchor="nw")

        self.show_num = tk.BooleanVar(self.tk, True)
        self.num_chk = tk.Checkbutton(
            self.tab1,
            text="Display # Particles",
            font=("helvetica", 8),
            variable=self.show_num,
        )
        self.num_chk.place(x=10, y=280, anchor="nw")

        self.show_links = tk.BooleanVar(self.tk, True)
        self.links_chk = tk.Checkbutton(
            self.tab1,
            text="Display links",
            font=("helvetica", 8),
            variable=self.show_links,
        )
        self.links_chk.place(x=10, y=300, anchor="nw")

        self.tab1_canvas.create_text(
            100, 335, text="Blocking Edges", font=("helvetica", 9), anchor="center"
        )
        self.tab1_canvas.create_line(10, 345, 190, 345, fill="grey50")

        self.top_bool = tk.BooleanVar(self.tk, True)
        self.top_chk = tk.Checkbutton(
            self.tab1, text="top", font=("helvetica", 8), variable=self.top_bool
        )
        self.top_chk.place(x=30, y=350, anchor="nw")

        self.bottom_bool = tk.BooleanVar(self.tk, True)
        self.bottom_chk = tk.Checkbutton(
            self.tab1, text="bottom", font=("helvetica", 8), variable=self.bottom_bool
        )
        self.bottom_chk.place(x=110, y=350, anchor="nw")

        self.left_bool = tk.BooleanVar(self.tk, True)
        self.left_chk = tk.Checkbutton(
            self.tab1, text="left", font=("helvetica", 8), variable=self.left_bool
        )
        self.left_chk.place(x=30, y=370, anchor="nw")

        self.right_bool = tk.BooleanVar(self.tk, True)
        self.right_chk = tk.Checkbutton(
            self.tab1, text="right", font=("helvetica", 8), variable=self.right_bool
        )
        self.right_chk.place(x=110, y=370, anchor="nw")

        self.tab1_canvas.create_text(
            100, 405, text="Optimization", font=("helvetica", 9), anchor="center"
        )
        self.tab1_canvas.create_line(10, 415, 190, 415, fill="grey50")

        self.grid_bool = tk.BooleanVar(self.tk, True)
        self.grid_chk = tk.Checkbutton(
            self.tab1, text="Use Grid", font=("helvetica", 8), variable=self.grid_bool
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

        self.grid_res_x_value.trace("w", self.sim.update_grid)
        self.grid_res_y_value.trace("w", self.sim.update_grid)

        self.tab1_canvas.create_text(
            100, 515, text="Extra", font=("helvetica", 9), anchor="center"
        )
        self.tab1_canvas.create_line(10, 525, 190, 525, fill="grey50")

        tk.Label(self.tab1, text="Min Spawn-Delay:", font=("helvetica", 8)).place(
            x=7, y=533, anchor="nw"
        )
        self.delay_entry = tk.Spinbox(self.tab1, width=7, from_=0, to=1, increment=0.01)
        self.delay_entry.delete(0, tk.END)
        self.delay_entry.insert(0, self.sim.min_spawn_delay)
        self.delay_entry.place(x=100, y=533)

        self.calculate_radii_diff_bool = tk.BooleanVar(self.tk, False)
        self.calculate_radii_diff_chk = tk.Checkbutton(
            self.tab1,
            text="Better Radii-Calculation",
            font=("helvetica", 8),
            variable=self.calculate_radii_diff_bool,
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
        self.tab2_canvas = tk.Canvas(self.tab2, width=200, height=self.sim.height)
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
            fill=self.sim.bg_color[1],
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
            command=self.sim.select_group,
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
            command=self.sim.copy_from_selected,
        )
        self.copy_selected_btn.place(x=15, y=self.sim.height - 65)
        self.set_selected_btn = tk.Button(
            self.tab2,
            text="Set Selected",
            bg="light green",
            command=self.sim.set_selected,
        )
        self.set_selected_btn.place(x=15, y=self.sim.height - 30)
        self.set_all_btn = tk.Button(
            self.tab2, text="Set All", bg="light blue", command=self.sim.set_all
        )
        self.set_all_btn.place(x=95, y=self.sim.height - 30)

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

    def change_color_entry(self, *event):
        try:
            color = eval(self.color_var.get())
            self.tab2_canvas.itemconfig(
                self.part_color_rect, fill="#%02x%02x%02x" % tuple(color)
            )
        except:
            if self.color_var.get() == "random" or self.color_var.get() == "":
                self.tab2_canvas.itemconfig(self.part_color_rect, fill="#ffffff")

    def update(self):
        if self.code_window is not None:
            self.code_window.tk.update()
        if self.extra_window is not None:
            self.extra_window.update()

        self.tk.update()

    def destroy(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.sim.running = False
            self.tk.destroy()
