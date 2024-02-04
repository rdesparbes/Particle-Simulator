import tkinter as tk


def get_double_var(spinbox: tk.Spinbox, value: float = 0.0) -> tk.DoubleVar:
    double_var = tk.DoubleVar(value=value)
    spinbox.delete(0, tk.END)
    spinbox.insert(0, str(value))
    spinbox.configure(textvariable=double_var, format="%.2f")
    return double_var
