import os
import tkinter as tk
from tkinter import ttk
from typing import Tuple, Union

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

        self._bar_canvas = tk.Canvas(
            self.tk, width=width, height=30, background="#1f3333"
        )
        self._bar_canvas.create_line(80, 0, 80, 30, fill="grey30")
        self._bar_canvas.pack(side="top", fill="x")

        self.canvas = tk.Canvas(self.tk, width=width, height=height)
        self.canvas.pack(side="top")

        self._play_photo = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/play.gif"), master=self._bar_canvas
        ).subsample(8, 8)
        self._pause_photo = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/pause.gif"), master=self._bar_canvas
        ).subsample(7, 7)
        self.pause_button = tk.Button(
            self._bar_canvas,
            image=self._pause_photo,
            cursor="hand2",
            border="0",
            bg="#1f3333",
            activebackground="#1f3333",
        )
        self.pause_button.place(x=40, y=16, anchor="center")

        x = 125
        self._select_img = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/select.gif"), master=self._bar_canvas
        ).subsample(57, 57)
        self.select_btn = tk.Button(
            self._bar_canvas,
            image=self._select_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._set_select_mode,
        )
        self.select_btn.place(x=x, y=16, anchor="center")
        self._select_rect = self._bar_canvas.create_rectangle(
            x - 12, 3, x + 12, 27, outline="blue", state="hidden"  # type: ignore[call-overload]
        )

        x = 165
        self._move_img = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/move.gif"), master=self._bar_canvas
        ).subsample(42, 42)
        self.move_btn = tk.Button(
            self._bar_canvas,
            image=self._move_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._set_move_mode,
        )
        self.move_btn.place(x=x, y=16, anchor="center")
        self._move_rect = self._bar_canvas.create_rectangle(
            x - 12, 3, x + 12, 27, outline="blue"
        )

        x = 205
        self._add_img = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/add.gif"), master=self._bar_canvas
        ).subsample(36, 36)
        self.add_btn = tk.Button(
            self._bar_canvas,
            image=self._add_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._set_add_mode,
        )
        self.add_btn.place(x=x, y=15, anchor="center")
        self._add_rect = self._bar_canvas.create_rectangle(
            x - 13, 3, x + 11, 27, outline="blue", state="hidden"  # type: ignore[call-overload]
        )

        self.link_btn = tk.Button(
            self._bar_canvas,
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
            self._bar_canvas,
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
            file=os.path.join(self._path, "Assets/save.gif"), master=self._bar_canvas
        ).subsample(28, 28)
        self.save_btn = tk.Button(
            self._bar_canvas,
            image=self._save_img,
            cursor="hand2",
            bg="#1f3333",
            activebackground="#1f3333",
            relief="flat",
        )
        self.save_btn.place(x=width - 110, y=16, anchor="center")

        self._load_img = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/load.gif"), master=self._bar_canvas
        ).subsample(32, 32)
        self.load_btn = tk.Button(
            self._bar_canvas,
            image=self._load_img,
            cursor="hand2",
            bg="#1f3333",
            activebackground="#1f3333",
            relief="flat",
        )
        self.load_btn.place(x=width - 75, y=16, anchor="center")

        self._code_img = tk.PhotoImage(
            file=os.path.join(self._path, "Assets/code.gif"), master=self._bar_canvas
        ).subsample(13, 13)
        self.code_btn = tk.Button(
            self._bar_canvas,
            image=self._code_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
        )
        self.code_btn.place(x=width - 25, y=16, anchor="center")

    def _reset_mode(self) -> None:
        self._bar_canvas.itemconfig(self._select_rect, state="hidden")
        self._bar_canvas.itemconfig(self._move_rect, state="hidden")
        self._bar_canvas.itemconfig(self._add_rect, state="hidden")

    def _set_select_mode(self) -> None:
        self._reset_mode()
        self._bar_canvas.itemconfig(self._select_rect, state="normal")

    def _set_move_mode(self) -> None:
        self._reset_mode()
        self._bar_canvas.itemconfig(self._move_rect, state="normal")

    def _set_add_mode(self) -> None:
        self._reset_mode()
        self._bar_canvas.itemconfig(self._add_rect, state="normal")

    @staticmethod
    def _set_entry(entry: Union[tk.Entry, tk.Spinbox], text: str) -> None:
        entry.delete(0, tk.END)
        entry.insert(0, text)

    def get_mouse_pos(self) -> Tuple[int, int]:
        return (
            self.tk.winfo_pointerx() - self.canvas.winfo_rootx(),
            self.tk.winfo_pointery() - self.canvas.winfo_rooty(),
        )
