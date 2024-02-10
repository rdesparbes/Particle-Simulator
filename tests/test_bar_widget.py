from typing import List

from particle_simulator.gui.bar_widget import BarWidget


def test_on_pause_button_pressed() -> None:
    bar_widget = BarWidget()
    calls: List[None] = []

    def callback(value: None) -> None:
        calls.append(value)

    bar_widget.on_pause_button_pressed.subscribe(callback)

    bar_widget.pause_button.invoke()

    assert calls == [None]
