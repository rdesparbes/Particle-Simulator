import tkinter as tk
from typing import TypeVar, Union

_NumberVar = TypeVar("_NumberVar", tk.IntVar, tk.DoubleVar)
_Variable = TypeVar("_Variable", tk.IntVar, tk.DoubleVar, tk.StringVar)
_SupportedWidgets = Union[tk.Spinbox, tk.Scale, tk.Entry]


def _set_var_from_entry(
    var: _Variable, widget: Union[tk.Entry, tk.Spinbox]
) -> _Variable:
    widget.delete(0, tk.END)
    widget.insert(0, str(var.get()))
    widget.configure(textvariable=var)
    return var


def _set_var_from_scale(var: _NumberVar, widget: tk.Scale) -> _NumberVar:
    widget.set(var.get())
    widget.configure(variable=var)
    return var


def _set_number_var(var: _NumberVar, widget: _SupportedWidgets) -> _NumberVar:
    if isinstance(widget, (tk.Spinbox, tk.Entry)):
        return _set_var_from_entry(var, widget)
    elif isinstance(widget, tk.Scale):
        return _set_var_from_scale(var, widget)
    else:
        raise NotImplementedError(f"Unsupported widget {widget}")


def get_double_var(widget: _SupportedWidgets, value: float = 0.0) -> tk.DoubleVar:
    return _set_number_var(tk.DoubleVar(value=value), widget)


def get_int_var(widget: _SupportedWidgets, value: int = 0) -> tk.IntVar:
    return _set_number_var(tk.IntVar(value=value), widget)


def get_string_var(
    widget: Union[tk.Spinbox, tk.Entry], value: str = ""
) -> tk.StringVar:
    return _set_var_from_entry(tk.StringVar(value=value), widget)
