import os
import tkinter as tk
from tkinter import ttk
from typing import Tuple

from particle_simulator.gui.bar_widget import BarWidget
from particle_simulator.gui.particle_widget import ParticleWidget
from particle_simulator.gui.simulation_widget import SimulationWidget


class GUIWidgets:
    def __init__(self, width: int, height: int, title: str) -> None:
        self._path = os.path.split(os.path.abspath(__file__))[0]

        self.tk = tk.Tk()
        self.tk.title(title)
        self.tk.resizable(False, False)

        self._tab_control = ttk.Notebook(self.tk, width=200)
        self._sim_tab = SimulationWidget(self._tab_control, self._path)
        self._tab_control.add(self._sim_tab, text="Sim-Settings")
        self._particle_tab = ParticleWidget(self._tab_control, self._path)
        self._tab_control.add(self._particle_tab, text="Particle-Settings")
        self._tab_control.pack(side="right", fill="both", expand=True)

        self._bar_canvas = BarWidget(
            width=width, resource_path=self._path, master=self.tk
        )
        self._bar_canvas.pack(side="top", fill="x")

        self.canvas = tk.Canvas(self.tk, width=width, height=height)
        self.canvas.pack(side="top")

    def get_mouse_pos(self) -> Tuple[int, int]:
        return (
            self.tk.winfo_pointerx() - self.canvas.winfo_rootx(),
            self.tk.winfo_pointery() - self.canvas.winfo_rooty(),
        )
