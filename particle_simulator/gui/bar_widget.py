import os
import tkinter as tk
from typing import Optional

from particle_simulator.engine.event import eventclass, event
from particle_simulator.mouse_mode import Mode


@eventclass
class BarWidget(tk.Canvas):
    def __init__(
        self,
        master: Optional[tk.Misc] = None,
        width: int = 650,
        height: int = 30,
    ) -> None:
        super().__init__(master, width=width, height=height, background="#1f3333")
        resource_path = os.path.split(os.path.abspath(__file__))[0]
        self.create_line(80, 0, 80, height, fill="grey30")

        self._play_photo = tk.PhotoImage(
            file=os.path.join(resource_path, "Assets/play.gif"), master=self
        ).subsample(8, 8)
        self._pause_photo = tk.PhotoImage(
            file=os.path.join(resource_path, "Assets/pause.gif"),
            master=self,
        ).subsample(7, 7)
        self.pause_button = tk.Button(
            self,
            cursor="hand2",
            border="0",
            bg="#1f3333",
            activebackground="#1f3333",
            command=lambda: self.on_pause_button_pressed(),
        )
        self.set_paused(True)
        self.pause_button.place(x=40, y=16, anchor="center")

        x = 125
        self._select_img = tk.PhotoImage(
            file=os.path.join(resource_path, "Assets/select.gif"),
            master=self,
        ).subsample(57, 57)
        self.select_btn = tk.Button(
            self,
            image=self._select_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._set_select_mode,
        )
        self.select_btn.place(x=x, y=16, anchor="center")
        self._select_rect = self.create_rectangle(
            x - 12, 3, x + 12, 27, outline="blue", state="hidden"
        )

        x = 165
        self._move_img = tk.PhotoImage(
            file=os.path.join(resource_path, "Assets/move.gif"), master=self
        ).subsample(42, 42)
        self.move_btn = tk.Button(
            self,
            image=self._move_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._set_move_mode,
        )
        self.move_btn.place(x=x, y=16, anchor="center")
        self._move_rect = self.create_rectangle(x - 12, 3, x + 12, 27, outline="blue")

        x = 205
        self._add_img = tk.PhotoImage(
            file=os.path.join(resource_path, "Assets/add.gif"), master=self
        ).subsample(36, 36)
        self.add_btn = tk.Button(
            self,
            image=self._add_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
            command=self._set_add_mode,
        )
        self.add_btn.place(x=x, y=15, anchor="center")
        self._add_rect = self.create_rectangle(
            x - 13, 3, x + 11, 27, outline="blue", state="hidden"
        )

        self.link_btn = tk.Button(
            self,
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
            self,
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
            file=os.path.join(resource_path, "Assets/save.gif"), master=self
        ).subsample(28, 28)
        self.save_btn = tk.Button(
            self,
            image=self._save_img,
            cursor="hand2",
            bg="#1f3333",
            activebackground="#1f3333",
            relief="flat",
        )
        self.save_btn.place(x=540, y=16, anchor="center")

        self._load_img = tk.PhotoImage(
            file=os.path.join(resource_path, "Assets/load.gif"), master=self
        ).subsample(32, 32)
        self.load_btn = tk.Button(
            self,
            image=self._load_img,
            cursor="hand2",
            bg="#1f3333",
            activebackground="#1f3333",
            relief="flat",
        )
        self.load_btn.place(x=575, y=16, anchor="center")

        self._code_img = tk.PhotoImage(
            file=os.path.join(resource_path, "Assets/code.gif"), master=self
        ).subsample(13, 13)
        self.code_btn = tk.Button(
            self,
            image=self._code_img,
            cursor="hand2",
            relief=tk.FLAT,
            bg="#1f3333",
            activebackground="#1f3333",
        )
        self.code_btn.place(x=625, y=16, anchor="center")

    @event
    def on_pause_button_pressed(self) -> None:
        return None

    def set_paused(self, paused: bool) -> None:
        self.pause_button.config(
            image=self._play_photo if paused else self._pause_photo
        )

    def _reset_mode(self) -> None:
        self.itemconfig(self._select_rect, state="hidden")
        self.itemconfig(self._move_rect, state="hidden")
        self.itemconfig(self._add_rect, state="hidden")

    def _set_select_mode(self) -> None:
        self._reset_mode()
        self.itemconfig(self._select_rect, state="normal")

    def _set_move_mode(self) -> None:
        self._reset_mode()
        self.itemconfig(self._move_rect, state="normal")

    def _set_add_mode(self) -> None:
        self._reset_mode()
        self.itemconfig(self._add_rect, state="normal")

    def set_mode(self, mode: Mode) -> None:
        if mode == "SELECT":
            self._set_select_mode()
        elif mode == "MOVE":
            self._set_move_mode()
        else:
            self._set_add_mode()
