import os
import re
import tkinter as tk
from tkinter import ttk, colorchooser
from typing import Optional, Tuple, Union

from particle_simulator.color import color_to_hex

_CANVAS_X = 0  # The X coordinate of the top-left corner of the canvas
_CANVAS_Y = 30  # The Y coordinate of the top-left corner of the canvas


class GUIWidgets:
    def __init__(self, width: int, height: int, title: str) -> None:
        self._path = os.path.split(os.path.abspath(__file__))[0]

        self.tk = tk.Tk()
        self.tk.title(title)
        self.tk.resizable(False, False)
        self._gui_canvas = tk.Canvas(self.tk, width=width + 200, height=height + 30)
        self._gui_canvas.pack()
        self.canvas = tk.Canvas(self.tk, width=width, height=height)
        self.canvas.place(x=_CANVAS_X, y=_CANVAS_Y)

        self._gui_canvas.create_rectangle(0, 0, width, 30, fill="#1f3333")
        self._gui_canvas.create_line(80, 0, 80, 30, fill="grey30")

        self._play_photo = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/play.gif"), master=self.tk
        ).subsample(8, 8)
        self._pause_photo = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/pause.gif"), master=self.tk
        ).subsample(7, 7)
        self.pause_button = tk.Button(
            self._gui_canvas,
            image=self._pause_photo,
            cursor="hand2",
            border="0",
            bg="#1f3333",
            activebackground="#1f3333",
        )
        self.pause_button.place(x=40, y=16, anchor="center")

        x = 125
        self._select_img = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/select.gif"), master=self.tk
        ).subsample(57, 57)
        self.select_btn = tk.Button(
            self.tk,
            image=self._select_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._set_select_mode,
        )
        self.select_btn.place(x=x, y=16, anchor="center")
        self._select_rect = self._gui_canvas.create_rectangle(
            x - 12, 3, x + 12, 27, outline="blue", state="hidden"  # type: ignore[call-overload]
        )

        x = 165
        self._move_img = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/move.gif"), master=self.tk
        ).subsample(42, 42)
        self.move_btn = tk.Button(
            self.tk,
            image=self._move_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._set_move_mode,
        )
        self.move_btn.place(x=x, y=16, anchor="center")
        self._move_rect = self._gui_canvas.create_rectangle(
            x - 12, 3, x + 12, 27, outline="blue"
        )

        x = 205
        self._add_img = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/add.gif"), master=self.tk
        ).subsample(36, 36)
        self.add_btn = tk.Button(
            self.tk,
            image=self._add_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._set_add_mode,
        )
        self.add_btn.place(x=x, y=15, anchor="center")
        self._add_rect = self._gui_canvas.create_rectangle(
            x - 13, 3, x + 11, 27, outline="blue", state="hidden"  # type: ignore[call-overload]
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

        self._save_img = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/save.gif"), master=self.tk
        ).subsample(28, 28)
        self.save_btn = tk.Button(
            self.tk,
            image=self._save_img,
            cursor="hand2",
            bg="#1f3333",
            activebackground="#1f3333",
            relief="flat",
        )
        self.save_btn.place(x=width - 110, y=16, anchor="center")

        self._load_img = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/load.gif"), master=self.tk
        ).subsample(32, 32)
        self.load_btn = tk.Button(
            self.tk,
            image=self._load_img,
            cursor="hand2",
            bg="#1f3333",
            activebackground="#1f3333",
            relief="flat",
        )
        self.load_btn.place(x=width - 75, y=16, anchor="center")

        self._code_img = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/code.gif"), master=self.tk
        ).subsample(13, 13)
        self.code_btn = tk.Button(
            self.tk,
            image=self._code_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
        )
        self.code_btn.place(x=width - 25, y=16, anchor="center")

        # layout sidebar-GUI
        self._tab_control = ttk.Notebook(self.tk)
        self._sim_tab = ttk.Frame(self._tab_control, relief="flat")
        self._tab_control.add(self._sim_tab, text="Sim-Settings")
        self._particle_tab = ttk.Frame(
            self._tab_control, relief="flat", width=200, height=height + 30
        )
        self._tab_control.add(self._particle_tab, text="Particle-Settings")
        self._tab_control.place(x=width, y=0)

        # layout self.tab1
        self._sim_tab_canvas = tk.Canvas(self._sim_tab, width=200, height=height)
        self._sim_tab_canvas.pack()

        tk.Label(self._sim_tab, text="Gravity:", font=("helvetica", 8)).place(
            x=7, y=20, anchor="nw"
        )
        self.gravity_entry = tk.Spinbox(
            self._sim_tab,
            width=7,
            from_=0,
            to=1,
            increment=0.1,
        )
        self.gravity_entry.place(x=100, y=20)

        tk.Label(self._sim_tab, text="Air Resistance:", font=("helvetica", 8)).place(
            x=7, y=50, anchor="nw"
        )
        self.air_res_entry = tk.Spinbox(
            self._sim_tab,
            width=7,
            from_=0,
            to=1,
            increment=0.01,
        )
        self.air_res_entry.place(x=100, y=50)

        tk.Label(self._sim_tab, text="Ground Friction:", font=("helvetica", 8)).place(
            x=7, y=80, anchor="nw"
        )
        self.friction_entry = tk.Spinbox(
            self._sim_tab,
            width=7,
            from_=0,
            to=1,
            increment=0.01,
        )
        self.friction_entry.place(x=100, y=80)

        self.temp_sc = tk.Scale(
            self._sim_tab,
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
        self.temp_sc.place(x=100, y=153, anchor="center")
        tk.Label(self._sim_tab, text="Temperature:", font=("helvetica", 8)).place(
            x=7, y=110, anchor="nw"
        )

        self.speed_sc = tk.Scale(
            self._sim_tab,
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
        self.speed_sc.place(x=100, y=233, anchor="center")
        tk.Label(self._sim_tab, text="Simulation Speed:", font=("helvetica", 8)).place(
            x=7, y=190, anchor="nw"
        )

        self.show_fps = tk.BooleanVar(self.tk)
        self.fps_chk = tk.Checkbutton(
            self._sim_tab,
            text="Display FPS",
            font=("helvetica", 8),
            variable=self.show_fps,
        )
        self.fps_chk.place(x=10, y=260, anchor="nw")

        self.show_num = tk.BooleanVar(self.tk)
        self.num_chk = tk.Checkbutton(
            self._sim_tab,
            text="Display # Particles",
            font=("helvetica", 8),
            variable=self.show_num,
        )
        self.num_chk.place(x=10, y=280, anchor="nw")

        self.show_links = tk.BooleanVar(self.tk)
        self.links_chk = tk.Checkbutton(
            self._sim_tab,
            text="Display links",
            font=("helvetica", 8),
            variable=self.show_links,
        )
        self.links_chk.place(x=10, y=300, anchor="nw")

        self._sim_tab_canvas.create_text(
            100, 335, text="Blocking Edges", font=("helvetica", 9), anchor="center"
        )
        self._sim_tab_canvas.create_line(10, 345, 190, 345, fill="grey50")

        self.top_bool = tk.BooleanVar(self.tk)
        self.top_chk = tk.Checkbutton(
            self._sim_tab,
            text="top",
            font=("helvetica", 8),
            variable=self.top_bool,
        )
        self.top_chk.place(x=30, y=350, anchor="nw")

        self.bottom_bool = tk.BooleanVar(self.tk)
        self.bottom_chk = tk.Checkbutton(
            self._sim_tab,
            text="bottom",
            font=("helvetica", 8),
            variable=self.bottom_bool,
        )
        self.bottom_chk.place(x=110, y=350, anchor="nw")

        self.left_bool = tk.BooleanVar(self.tk)
        self.left_chk = tk.Checkbutton(
            self._sim_tab,
            text="left",
            font=("helvetica", 8),
            variable=self.left_bool,
        )
        self.left_chk.place(x=30, y=370, anchor="nw")

        self.right_bool = tk.BooleanVar(self.tk)
        self.right_chk = tk.Checkbutton(
            self._sim_tab,
            text="right",
            font=("helvetica", 8),
            variable=self.right_bool,
        )
        self.right_chk.place(x=110, y=370, anchor="nw")

        self._sim_tab_canvas.create_text(
            100, 405, text="Optimization", font=("helvetica", 9), anchor="center"
        )
        self._sim_tab_canvas.create_line(10, 415, 190, 415, fill="grey50")

        self.grid_bool = tk.BooleanVar(self.tk, True)
        self.grid_chk = tk.Checkbutton(
            self._sim_tab,
            text="Use Grid",
            font=("helvetica", 8),
            variable=self.grid_bool,
        )
        self.grid_chk.place(x=10, y=425, anchor="nw")

        tk.Label(self._sim_tab, text="Grid-Res:", font=("helvetica", 8)).place(
            x=7, y=455, anchor="nw"
        )
        tk.Label(self._sim_tab, text="X:", font=("helvetica", 8)).place(
            x=60, y=455, anchor="nw"
        )
        self.grid_res_x = tk.Spinbox(
            self._sim_tab,
            width=7,
            from_=1,
            to=200,
            increment=1,
        )
        self.grid_res_x.place(x=80, y=455)

        tk.Label(self._sim_tab, text="Y:", font=("helvetica", 8)).place(
            x=60, y=480, anchor="nw"
        )
        self.grid_res_y = tk.Spinbox(
            self._sim_tab,
            width=7,
            from_=1,
            to=200,
            increment=1,
        )
        self.grid_res_y.place(x=80, y=480)

        self._sim_tab_canvas.create_text(
            100, 515, text="Extra", font=("helvetica", 9), anchor="center"
        )
        self._sim_tab_canvas.create_line(10, 525, 190, 525, fill="grey50")

        tk.Label(self._sim_tab, text="Min Spawn-Delay:", font=("helvetica", 8)).place(
            x=7, y=533, anchor="nw"
        )
        self.delay_entry = tk.Spinbox(
            self._sim_tab,
            width=7,
            from_=0,
            to=1,
            increment=0.01,
        )
        self.delay_entry.place(x=100, y=533)

        self.calculate_radii_diff_bool = tk.BooleanVar(self.tk, False)
        self.calculate_radii_diff_chk = tk.Checkbutton(
            self._sim_tab,
            text="Better Radii-Calculation",
            font=("helvetica", 8),
            variable=self.calculate_radii_diff_bool,
        )
        self.calculate_radii_diff_chk.place(x=7, y=553, anchor="nw")

        self._extra_img = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/dots.gif"), master=self.tk
        ).subsample(11, 11)
        self.extra_btn = tk.Button(
            self._sim_tab,
            image=self._extra_img,
            cursor="hand2",
            bg="#F0F0F0",
            activebackground="#F0F0F0",
            relief="flat",
        )
        self.extra_btn.place(x=7, y=580)

        # layout tab2
        self._particle_tab_canvas = tk.Canvas(self._particle_tab, width=200, height=height)
        self._particle_tab_canvas.pack()

        tk.Label(self._particle_tab, text="Radius:", font=("helvetica", 8)).place(
            x=7, y=20, anchor="nw"
        )
        self.radius_entry = tk.Spinbox(self._particle_tab, width=7, from_=1, to=300, increment=1)
        self._set_entry(self.radius_entry, "4")
        self.radius_entry.place(x=100, y=20)

        tk.Label(self._particle_tab, text="Color:", font=("helvetica", 8)).place(
            x=7, y=50, anchor="nw"
        )
        self.color_var = tk.StringVar(self.tk, "random")
        self._color_entry = tk.Entry(self._particle_tab, width=8, textvariable=self.color_var)
        self._color_entry.place(x=100, y=50)
        self.color_var.trace("w", self._change_color_entry)

        self._part_color_rect = self._particle_tab_canvas.create_rectangle(
            160,
            48,
            180,
            68,
            fill="#ffffff",
            activeoutline="red",
            tags="part_color_rect",
        )
        self._particle_tab_canvas.tag_bind(
            "part_color_rect", "<Button-1>", self._ask_color_entry
        )

        tk.Label(self._particle_tab, text="Mass:", font=("helvetica", 8)).place(
            x=7, y=80, anchor="nw"
        )
        self.mass_entry = tk.Spinbox(
            self._particle_tab, width=7, from_=0.1, to=100, increment=0.1
        )
        self._set_entry(self.mass_entry, "1")
        self.mass_entry.place(x=100, y=80)

        tk.Label(self._particle_tab, text="Bounciness:", font=("helvetica", 8)).place(
            x=7, y=110, anchor="nw"
        )
        self.bounciness_entry = tk.Spinbox(
            self._particle_tab, width=7, from_=0, to=1, increment=0.1
        )
        self._set_entry(self.bounciness_entry, "0.7")
        self.bounciness_entry.place(x=100, y=110)

        tk.Label(self._particle_tab, text="Velocity:", font=("helvetica", 8)).place(
            x=7, y=140, anchor="nw"
        )
        tk.Label(self._particle_tab, text="X:", font=("helvetica", 8)).place(
            x=60, y=140, anchor="nw"
        )
        self.velocity_x_entry = tk.Spinbox(
            self._particle_tab, width=7, from_=0, to=1, increment=0.1
        )
        self._set_entry(self.velocity_x_entry, "0")
        self.velocity_x_entry.place(x=100, y=140)
        tk.Label(self._particle_tab, text="Y:", font=("helvetica", 8)).place(
            x=60, y=162, anchor="nw"
        )
        self.velocity_y_entry = tk.Spinbox(
            self._particle_tab, width=7, from_=-5, to=5, increment=0.1
        )
        self._set_entry(self.velocity_y_entry, "0")
        self.velocity_y_entry.place(x=100, y=162)

        self.locked_bool = tk.BooleanVar(self.tk, False)
        self.locked_chk = tk.Checkbutton(
            self._particle_tab, text="Locked", font=("helvetica", 8), variable=self.locked_bool
        )
        self.locked_chk.place(x=7, y=190, anchor="nw")

        self.do_collision_bool = tk.BooleanVar(self.tk, False)
        self.do_collision_chk = tk.Checkbutton(
            self._particle_tab,
            text="Check Collisions",
            font=("helvetica", 8),
            variable=self.do_collision_bool,
        )
        self.do_collision_chk.place(x=7, y=210, anchor="nw")

        tk.Label(self._particle_tab, text="Attraction-radius:", font=("helvetica", 8)).place(
            x=7, y=250, anchor="nw"
        )
        self.attr_r_entry = tk.Spinbox(
            self._particle_tab, width=7, from_=-1, to=500, increment=1
        )
        self._set_entry(self.attr_r_entry, "-1")
        self.attr_r_entry.place(x=100, y=250)

        tk.Label(self._particle_tab, text="Attr-strength:", font=("helvetica", 8)).place(
            x=7, y=273, anchor="nw"
        )
        self.attr_strength_entry = tk.Spinbox(
            self._particle_tab, width=7, from_=0, to=50, increment=0.1
        )
        self._set_entry(self.attr_strength_entry, "0.5")
        self.attr_strength_entry.place(x=100, y=273)

        self.gravity_mode_bool = tk.BooleanVar(self.tk, False)
        self.gravity_mode_chk = tk.Checkbutton(
            self._particle_tab,
            text="Gravity-Mode",
            font=("helvetica", 7),
            variable=self.gravity_mode_bool,
        )
        self.gravity_mode_chk.place(x=7, y=290, anchor="nw")

        tk.Label(self._particle_tab, text="Repulsion-radius:", font=("helvetica", 8)).place(
            x=7, y=313, anchor="nw"
        )
        self.repel_r_entry = tk.Spinbox(
            self._particle_tab, width=7, from_=0, to=500, increment=1
        )
        self._set_entry(self.repel_r_entry, "10")
        self.repel_r_entry.place(x=100, y=323)

        tk.Label(self._particle_tab, text="Repel-strength:", font=("helvetica", 8)).place(
            x=7, y=336, anchor="nw"
        )
        self.repel_strength_entry = tk.Spinbox(
            self._particle_tab, width=7, from_=0, to=50, increment=0.1
        )
        self._set_entry(self.repel_strength_entry, "1")
        self.repel_strength_entry.place(x=100, y=346)

        self.linked_group_bool = tk.BooleanVar(self.tk, True)
        self.linked_group_chk = tk.Checkbutton(
            self._particle_tab,
            text="Linked to group-particles",
            font=("helvetica", 8),
            variable=self.linked_group_bool,
        )
        self.linked_group_chk.place(x=7, y=376, anchor="nw")

        tk.Label(self._particle_tab, text="Link-breaking-force:", font=("helvetica", 8)).place(
            x=7, y=400, anchor="nw"
        )
        tk.Label(self._particle_tab, text="Attr:", font=("helvetica", 8)).place(
            x=7, y=420, anchor="nw"
        )
        self.link_attr_break_entry = tk.Spinbox(
            self._particle_tab, width=5, from_=0, to=5000, increment=0.1
        )
        self._set_entry(self.link_attr_break_entry, "-1")
        self.link_attr_break_entry.place(x=40, y=420)
        tk.Label(self._particle_tab, text="Repel:", font=("helvetica", 8)).place(
            x=100, y=420, anchor="nw"
        )
        self.link_repel_break_entry = tk.Spinbox(
            self._particle_tab, width=5, from_=0, to=5000, increment=0.1
        )
        self._set_entry(self.link_repel_break_entry, "-1")
        self.link_repel_break_entry.place(x=140, y=420)

        tk.Label(self._particle_tab, text="Particle-group:", font=("helvetica", 8)).place(
            x=7, y=450, anchor="nw"
        )
        self.groups_entry = ttk.Combobox(self._particle_tab, width=10)
        self.groups_entry.place(x=10, y=470, anchor="nw")

        self.group_add_btn = tk.Button(
            self._particle_tab,
            text="+",
            font=("Helvetica", 15, "bold"),
            cursor="hand2",
            fg="grey14",
            bg="#F0F0F0",
            activebackground="#F0F0F0",
            relief="flat",
            width=1,
        )
        self.group_add_btn.place(x=105, y=480, anchor="center")

        self._select_img2 = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/select2.gif"), master=self.tk
        ).subsample(54, 54)
        self.group_select_btn = tk.Button(
            self._particle_tab,
            image=self._select_img2,
            cursor="hand2",
            bg="#F0F0F0",
            activebackground="#F0F0F0",
            relief="flat",
        )
        self.group_select_btn.place(x=123, y=480, anchor="center")

        self.separate_group_bool = tk.BooleanVar(self.tk, False)
        self.separate_group_chk = tk.Checkbutton(
            self._particle_tab,
            text="Separate Group",
            font=("helvetica", 8),
            variable=self.separate_group_bool,
        )
        self.separate_group_chk.place(x=10, y=495, anchor="nw")

        self.copy_selected_btn = tk.Button(
            self._particle_tab,
            text="Copy from selected",
            bg="light coral",
        )
        self.copy_selected_btn.place(x=15, y=height - 65)
        self.set_selected_btn = tk.Button(
            self._particle_tab,
            text="Set Selected",
            bg="light green",
        )
        self.set_selected_btn.place(x=15, y=height - 30)
        self.set_all_btn = tk.Button(self._particle_tab, text="Set All", bg="light blue")
        self.set_all_btn.place(x=95, y=height - 30)

    def _reset_mode(self) -> None:
        self._gui_canvas.itemconfig(self._select_rect, state="hidden")
        self._gui_canvas.itemconfig(self._move_rect, state="hidden")
        self._gui_canvas.itemconfig(self._add_rect, state="hidden")

    def _set_select_mode(self) -> None:
        self._reset_mode()
        self._gui_canvas.itemconfig(self._select_rect, state="normal")

    def _set_move_mode(self) -> None:
        self._reset_mode()
        self._gui_canvas.itemconfig(self._move_rect, state="normal")

    def _set_add_mode(self) -> None:
        self._reset_mode()
        self._gui_canvas.itemconfig(self._add_rect, state="normal")

    @staticmethod
    def _set_entry(entry: Union[tk.Entry, tk.Spinbox], text: str) -> None:
        entry.delete(0, tk.END)
        entry.insert(0, text)

    def _ask_color_entry(self, *_event: tk.Event) -> None:
        color, color_hex = colorchooser.askcolor(title="Choose color")
        if color is not None:
            self.color_var.set(str(list(color)))
            self._particle_tab_canvas.itemconfig(self._part_color_rect, fill=color_hex)

    def _set_part_color(self, color: Tuple[int, int, int]) -> None:
        hex_color = color_to_hex(color)
        self._particle_tab_canvas.itemconfig(self._part_color_rect, fill=hex_color)

    def _parse_color(self) -> Optional[Tuple[int, int, int]]:
        color_regex = r"(?P<red>\d+)\s*,\s*(?P<green>\d+)\s*,\s*(?P<blue>\d+)"
        m = re.search(color_regex, self.color_var.get())
        if m is None:
            return None
        return int(m.group("red")), int(m.group("green")), int(m.group("blue"))

    def _change_color_entry(self, *_event: tk.Event) -> None:
        color = self._parse_color()
        if color is None:
            self._set_part_color((255, 255, 255))
        else:
            self._set_part_color(color)

    def get_mouse_pos(self) -> Tuple[int, int]:
        return (
            self.tk.winfo_pointerx() - self.tk.winfo_rootx() - _CANVAS_X,
            self.tk.winfo_pointery() - self.tk.winfo_rooty() - _CANVAS_Y,
        )
