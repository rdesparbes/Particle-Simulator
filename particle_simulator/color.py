from typing import Tuple


def color_to_hex(color: Tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
