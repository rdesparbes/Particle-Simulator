import os
import re
import tkinter as tk
from tkinter import ttk, colorchooser
from typing import Optional, Tuple

from particle_simulator.color import color_to_hex
from particle_simulator.gui.utils import get_double_var


class ParticleWidget(ttk.Frame):
    def __init__(self, master: tk.Widget, resource_path: str) -> None:
        super().__init__(master, relief="flat")
        self._particle_tab_canvas = tk.Canvas(self)
        self._particle_tab_canvas.pack()

        tk.Label(self, text="Radius:", font=("helvetica", 8)).place(
            x=7, y=20, anchor="nw"
        )
        self._radius_entry = tk.Spinbox(self, width=7, from_=1, to=300, increment=1)
        self.radius_var = get_double_var(self._radius_entry, value=4.0)
        self._radius_entry.place(x=100, y=20)

        tk.Label(self, text="Color:", font=("helvetica", 8)).place(
            x=7, y=50, anchor="nw"
        )
        self.color_var = tk.StringVar(master, "random")
        self._color_entry = tk.Entry(self, width=8, textvariable=self.color_var)
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

        tk.Label(self, text="Mass:", font=("helvetica", 8)).place(
            x=7, y=80, anchor="nw"
        )
        self._mass_entry = tk.Spinbox(self, width=7, from_=0.1, to=100, increment=0.1)
        self.mass_var = get_double_var(self._mass_entry, value=1.0)
        self._mass_entry.place(x=100, y=80)

        tk.Label(self, text="Bounciness:", font=("helvetica", 8)).place(
            x=7, y=110, anchor="nw"
        )
        self._bounciness_entry = tk.Spinbox(self, width=7, from_=0, to=1, increment=0.1)
        self.bounciness_var = get_double_var(self._bounciness_entry, value=0.7)
        self._bounciness_entry.place(x=100, y=110)

        tk.Label(self, text="Velocity:", font=("helvetica", 8)).place(
            x=7, y=140, anchor="nw"
        )
        tk.Label(self, text="X:", font=("helvetica", 8)).place(x=60, y=140, anchor="nw")
        self._velocity_x_entry = tk.Spinbox(self, width=7, from_=0, to=1, increment=0.1)
        self.velocity_x_var = get_double_var(self._velocity_x_entry, 0.0)
        self._velocity_x_entry.place(x=100, y=140)
        tk.Label(self, text="Y:", font=("helvetica", 8)).place(x=60, y=162, anchor="nw")
        self._velocity_y_entry = tk.Spinbox(
            self, width=7, from_=-5, to=5, increment=0.1
        )
        self.velocity_y_var = get_double_var(self._velocity_y_entry, 0.0)
        self._velocity_y_entry.place(x=100, y=162)

        self.locked_bool = tk.BooleanVar(master, False)
        self.locked_chk = tk.Checkbutton(
            self,
            text="Locked",
            font=("helvetica", 8),
            variable=self.locked_bool,
        )
        self.locked_chk.place(x=7, y=190, anchor="nw")

        self.do_collision_bool = tk.BooleanVar(master, False)
        self.do_collision_chk = tk.Checkbutton(
            self,
            text="Check Collisions",
            font=("helvetica", 8),
            variable=self.do_collision_bool,
        )
        self.do_collision_chk.place(x=7, y=210, anchor="nw")

        tk.Label(self, text="Attraction-radius:", font=("helvetica", 8)).place(
            x=7, y=250, anchor="nw"
        )
        self._attr_r_entry = tk.Spinbox(self, width=7, from_=-1, to=500, increment=1)
        self.attr_r_var = get_double_var(self._attr_r_entry, -1.0)
        self._attr_r_entry.place(x=100, y=250)

        tk.Label(self, text="Attr-strength:", font=("helvetica", 8)).place(
            x=7, y=273, anchor="nw"
        )
        self._attr_strength_entry = tk.Spinbox(
            self, width=7, from_=0, to=50, increment=0.1
        )
        self.attr_strength_var = get_double_var(self._attr_strength_entry, 0.5)
        self._attr_strength_entry.place(x=100, y=273)

        self.gravity_mode_bool = tk.BooleanVar(master, False)
        self.gravity_mode_chk = tk.Checkbutton(
            self,
            text="Gravity-Mode",
            font=("helvetica", 7),
            variable=self.gravity_mode_bool,
        )
        self.gravity_mode_chk.place(x=7, y=290, anchor="nw")

        tk.Label(self, text="Repulsion-radius:", font=("helvetica", 8)).place(
            x=7, y=313, anchor="nw"
        )
        self._repel_r_entry = tk.Spinbox(self, width=7, from_=0, to=500, increment=1)
        self.repel_r_var = get_double_var(self._repel_r_entry, 10.0)
        self._repel_r_entry.place(x=100, y=323)

        tk.Label(self, text="Repel-strength:", font=("helvetica", 8)).place(
            x=7, y=336, anchor="nw"
        )
        self._repel_strength_entry = tk.Spinbox(
            self, width=7, from_=0, to=50, increment=0.1
        )
        self.repel_strength_var = get_double_var(self._repel_strength_entry, 1.0)
        self._repel_strength_entry.place(x=100, y=346)

        self.linked_group_bool = tk.BooleanVar(master, True)
        self.linked_group_chk = tk.Checkbutton(
            self,
            text="Linked to group-particles",
            font=("helvetica", 8),
            variable=self.linked_group_bool,
        )
        self.linked_group_chk.place(x=7, y=376, anchor="nw")

        tk.Label(self, text="Link-breaking-force:", font=("helvetica", 8)).place(
            x=7, y=400, anchor="nw"
        )
        tk.Label(self, text="Attr:", font=("helvetica", 8)).place(
            x=7, y=420, anchor="nw"
        )
        self._link_attr_break_entry = tk.Spinbox(
            self, width=5, from_=0, to=5000, increment=0.1
        )
        self.link_attr_break_var = get_double_var(self._link_attr_break_entry, -1.0)
        self._link_attr_break_entry.place(x=40, y=420)
        tk.Label(self, text="Repel:", font=("helvetica", 8)).place(
            x=100, y=420, anchor="nw"
        )
        self._link_repel_break_entry = tk.Spinbox(
            self, width=5, from_=0, to=5000, increment=0.1
        )
        self.link_repel_break_var = get_double_var(self._link_repel_break_entry, -1.0)
        self._link_repel_break_entry.place(x=140, y=420)

        tk.Label(self, text="Particle-group:", font=("helvetica", 8)).place(
            x=7, y=450, anchor="nw"
        )
        self.groups_entry = ttk.Combobox(self, width=10)
        self.groups_entry.place(x=10, y=470, anchor="nw")

        self.group_add_btn = tk.Button(
            self,
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
            file=os.path.join(resource_path, "Assets/select2.gif"), master=master
        ).subsample(54, 54)
        self.group_select_btn = tk.Button(
            self,
            image=self._select_img2,
            cursor="hand2",
            bg="#F0F0F0",
            activebackground="#F0F0F0",
            relief="flat",
        )
        self.group_select_btn.place(x=123, y=480, anchor="center")

        self.separate_group_bool = tk.BooleanVar(master, False)
        self.separate_group_chk = tk.Checkbutton(
            self,
            text="Separate Group",
            font=("helvetica", 8),
            variable=self.separate_group_bool,
        )
        self.separate_group_chk.place(x=10, y=495, anchor="nw")

        self.copy_selected_btn = tk.Button(
            self,
            text="Copy from selected",
            bg="light coral",
        )
        self.copy_selected_btn.place(x=15, y=535)
        self.set_selected_btn = tk.Button(
            self,
            text="Set Selected",
            bg="light green",
        )
        self.set_selected_btn.place(x=15, y=570)
        self.set_all_btn = tk.Button(self, text="Set All", bg="light blue")
        self.set_all_btn.place(x=95, y=570)

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
