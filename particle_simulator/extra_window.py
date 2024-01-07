import os
import time
import tkinter as tk
from tkinter import colorchooser

import numpy as np

from .error import Error
from .simulation_state import SimulationState


class ExtraWindow:
    def __init__(self, sim: SimulationState, path: str) -> None:
        self.sim = sim
        self.path = path
        self.tk = tk.Tk()
        self.tk.title("Extra Options")
        self.tk.resizable(width=False, height=False)

        self.gui_canvas = tk.Canvas(self.tk, width=300, height=300)
        self.gui_canvas.pack()

        tk.Label(self.tk, text="Extra Options:", font=("Helvetica", 9, "bold")).place(
            x=20, y=10
        )

        tk.Label(self.tk, text="Gravity-Angle (°):", font=("helvetica", 8)).place(
            x=25, y=30, anchor="nw"
        )
        self.gravity_dir = tk.IntVar(
            self.tk, value=np.degrees(np.arctan2(*self.sim.g_dir))
        )
        self.gravity_dir_entry = tk.Spinbox(
            self.tk,
            width=7,
            from_=-360,
            to=360,
            increment=5,
            textvariable=self.gravity_dir,
        )
        self.gravity_dir_entry.place(x=118, y=30)
        self.gravity_dir.trace("w", self.update_gravity)
        self.g_dir_line = self.gui_canvas.create_line(
            200, 40, *(self.sim.g_dir * 15 + np.array([200, 40]))
        )

        tk.Label(self.tk, text="Wind-Angle (°):", font=("helvetica", 8)).place(
            x=25, y=60, anchor="nw"
        )
        self.wind_dir = tk.IntVar(
            self.tk, value=np.degrees(np.arctan2(*self.sim.wind_force))
        )
        self.wind_dir_entry = tk.Spinbox(
            self.tk,
            width=7,
            from_=-360,
            to=360,
            increment=5,
            textvariable=self.wind_dir,
        )
        self.wind_dir_entry.place(x=118, y=60)
        self.wind_dir.trace("w", self.update_wind)

        tk.Label(self.tk, text="Wind-Strength:", font=("helvetica", 8)).place(
            x=25, y=85, anchor="nw"
        )
        self.wind_strength = tk.DoubleVar(
            self.tk, value=10.0 * float(np.linalg.norm(self.sim.wind_force))
        )
        self.wind_strength_entry = tk.Spinbox(
            self.tk,
            width=7,
            from_=0,
            to=100,
            increment=0.5,
            textvariable=self.wind_strength,
        )
        self.wind_strength_entry.place(x=118, y=85)
        self.wind_strength.trace("w", self.update_wind)
        self.wind_line = self.gui_canvas.create_line(
            200, 75, *(self.sim.wind_force * 25 + np.array([200, 75]))
        )

        tk.Label(self.tk, text="Background-color:", font=("helvetica", 8)).place(
            x=25, y=115, anchor="nw"
        )
        self.bg_color_rect = self.gui_canvas.create_rectangle(
            130,
            115,
            150,
            135,
            fill=self.sim.bg_color[1],
            activeoutline="red",
            tags="color_rect",
        )
        self.gui_canvas.tag_bind("color_rect", "<Button-1>", self.change_bg_color)

        self.void_edges_bool = tk.BooleanVar(self.tk, self.sim.void_edges)
        self.void_edges_chk = tk.Checkbutton(
            self.tk,
            text="Void edges",
            font=("helvetica", 8),
            variable=self.void_edges_bool,
        )
        self.void_edges_chk.place(x=25, y=135)
        self.void_edges_bool.trace("w", self.void_edges_toggle)

        self.gui_canvas.create_text(
            75, 170, text="Links", font=("helvetica", 9), anchor="center"
        )
        self.gui_canvas.create_line(50, 180, 100, 180, fill="grey50")

        self.fit_link_btn = tk.Button(
            self.tk,
            text="Fit-link Selected",
            font=("helvetica", 7, "bold"),
            bg="light blue",
            command=lambda: self.sim.link_selection(fit_link=True),
        )
        self.fit_link_btn.place(x=30, y=195)

        self.stress_visualization_bool = tk.BooleanVar(
            self.tk, self.sim.stress_visualization
        )
        self.stress_visualization_chk = tk.Checkbutton(
            self.tk,
            text="Stress Visualization",
            font=("helvetica", 8),
            variable=self.stress_visualization_bool,
        )
        self.stress_visualization_chk.place(x=25, y=220, anchor="nw")
        self.stress_visualization_bool.trace("w", self.update_stress)

        self.min_delta_change = 0.25
        self.changing_length_last_time = 0
        self.changing_length_plus = False
        self.changing_length_minus = False
        tk.Label(
            self.tk, text="Change selected fit-link-length:", font=("helvetica", 8)
        ).place(x=25, y=245, anchor="nw")
        self.delta_length_entry = tk.Spinbox(
            self.tk, width=7, from_=0, to=100, increment=0.1
        )
        self.delta_length_entry.place(x=30, y=265)

        self.plus_photo = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/plus.gif"), master=self.tk
        ).subsample(7, 7)
        self.link_longer_button = tk.Button(
            self.tk,
            image=self.plus_photo,
            cursor="hand2",
            bg="#F0F0F0",
            activebackground="#F0F0F0",
            relief="flat",
            command=lambda: self.change_length(1),
        )
        self.link_longer_button.bind(
            "<ButtonPress-1>", lambda x: self.toggle_link_change_plus(True)
        )
        self.link_longer_button.bind(
            "<ButtonRelease-1>", lambda x: self.toggle_link_change_plus(False)
        )
        self.link_longer_button.place(x=100, y=275, anchor="center")

        self.minus_photo = tk.PhotoImage(
            file=os.path.join(self.path, "Assets/minus.gif"), master=self.tk
        ).subsample(10, 10)
        self.link_shorter_button = tk.Button(
            self.tk,
            image=self.minus_photo,
            cursor="hand2",
            bg="#F0F0F0",
            activebackground="#F0F0F0",
            relief="flat",
            command=lambda: self.change_length(-1),
        )
        self.link_shorter_button.bind(
            "<ButtonPress-1>", lambda x: self.toggle_link_change_minus(True)
        )
        self.link_shorter_button.bind(
            "<ButtonRelease-1>", lambda x: self.toggle_link_change_minus(False)
        )
        self.link_shorter_button.place(x=125, y=275, anchor="center")

    def update_gravity(self, *event):
        try:
            rads = np.radians(self.gravity_dir.get())
            self.sim.g_dir = np.array([np.sin(rads), np.cos(rads)])
            self.gui_canvas.delete(self.g_dir_line)
            self.g_dir_line = self.gui_canvas.create_line(
                200, 40, *(self.sim.g_dir * 15 + np.array([200, 40]))
            )
        except:
            pass

    def update_wind(self, *event):
        try:
            rads = np.radians(self.wind_dir.get())
            self.sim.wind_force = (
                np.array([np.sin(rads), np.cos(rads)]) * self.wind_strength.get() / 10
            )
            self.gui_canvas.delete(self.wind_line)
            self.wind_line = self.gui_canvas.create_line(
                200, 75, *(self.sim.wind_force * 100 + np.array([200, 75]))
            )
        except:
            pass

    def update_stress(self, *event):
        self.sim.stress_visualization = self.stress_visualization_bool.get()

    def change_bg_color(self, *event):
        color = colorchooser.askcolor(title="Choose color")
        if color[0] is not None:
            self.sim.bg_color = color
            self.gui_canvas.itemconfig(self.bg_color_rect, fill=color[1])

    def void_edges_toggle(self, *event):
        self.sim.void_edges = self.void_edges_bool.get()

    def change_length(self, sign):
        try:
            self.sim.change_link_lengths(
                self.sim.selection, float(self.delta_length_entry.get()) * sign
            )
        except Exception as e:
            self.sim.error = Error("Input-Error", e)

    def toggle_link_change_plus(self, state):
        self.changing_length_plus = time.time() if state else False

    def toggle_link_change_minus(self, state):
        self.changing_length_minus = time.time() if state else False

    def update(self):
        self.tk.update()
        delta_condition = (
            time.time() - self.changing_length_last_time >= self.min_delta_change
        )
        if (
            self.changing_length_plus
            and time.time() - self.changing_length_plus >= 1
            and delta_condition
        ):
            self.change_length(1)
        if (
            self.changing_length_minus
            and time.time() - self.changing_length_minus >= 1
            and delta_condition
        ):
            self.change_length(-1)
