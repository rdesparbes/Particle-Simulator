import random
from typing import Tuple


def color_to_hex(color: Tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def generate_random() -> Tuple[int, int, int]:
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )
