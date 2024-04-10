import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from particle_simulator.engine.event import event, eventclass


@eventclass
class CodeWindow:
    def __init__(self) -> None:
        self.tk = tk.Tk()
        self.tk.title("Code-Window")
        self.tk.geometry("500x500")
        self.tk.resizable(width=False, height=False)
        self.tk.protocol("WM_DELETE_WINDOW", self.destroy)

        tk.Label(self.tk, text="Code:", font=("Helvetica", 10, "bold")).place(
            x=15, y=10
        )

        self.scroll_frame = tk.Frame(self.tk, width=450, height=400)
        self.scroll_frame.place(x=20, y=35)
        self.scroll_frame.grid_propagate(False)
        self.scroll_frame.grid_rowconfigure(0, weight=1)
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        self.code_box = tk.Text(self.scroll_frame, undo=True)
        self.code_box.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ttk.Scrollbar(self.scroll_frame, command=self.code_box.yview)
        self.scrollbar.grid(row=0, column=1, sticky="nsew")
        self.code_box["yscrollcommand"] = self.scrollbar.set

        font = tkfont.Font(font=self.code_box["font"])
        tab = font.measure(" " * 4)
        self.code_box.config(tabs=tab)

        self.exec_btn = tk.Button(
            self.tk, text="Execute Code", background="light green", command=self.execute
        )
        self.exec_btn.place(x=20, y=445)

        self.use_threading = tk.BooleanVar(self.tk, True)
        self.threading_chk = tk.Checkbutton(
            self.tk,
            text="Use Threading",
            font=("helvetica", 8),
            variable=self.use_threading,
        )
        self.threading_chk.place(x=460, y=435, anchor="ne")

    @property
    def code(self) -> str:
        return self.code_box.get("1.0", tk.END)

    def set_code(self, code: str) -> None:
        self.code_box.insert(tk.INSERT, code)

    @event
    def on_save(self, code: str) -> str:
        return code

    @event
    def on_exec(self, code: str) -> str:
        return code

    def execute(self) -> None:
        code = self.code
        self.on_save(code)

        if self.use_threading.get():
            threading.Thread(target=self.on_exec, args=[code]).start()
        else:
            self.on_exec(code)

    def destroy(self) -> None:
        self.on_save(self.code)
        self.tk.destroy()
