from dataclasses import dataclass


@dataclass
class SimGUISettings:
    show_fps: bool
    show_num: bool
    show_links: bool
    grid_res_x: int
    grid_res_y: int
    delay: float
