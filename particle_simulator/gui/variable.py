import tkinter as tk
from typing import TypeVar, Union

_Variable = TypeVar("_Variable", tk.IntVar, tk.DoubleVar)
_SupportedWidgets = Union[tk.Spinbox, tk.Scale]


def _set_var(var: _Variable, widget: _SupportedWidgets) -> _Variable:
    if isinstance(widget, tk.Spinbox):
        widget.delete(0, tk.END)
        widget.insert(0, str(var.get()))
        widget.configure(textvariable=var, format="%.2f")
    elif isinstance(widget, tk.Scale):
        widget.set(var.get())
        widget.configure(variable=var)
    else:
        raise NotImplementedError(f"Unsupported widget {widget}")
    return var


def get_double_var(widget: _SupportedWidgets, value: float = 0.0) -> tk.DoubleVar:
    return _set_var(tk.DoubleVar(value=value), widget)


def get_int_var(widget: _SupportedWidgets, value: int = 0) -> tk.IntVar:
    return _set_var(tk.IntVar(value=value), widget)
