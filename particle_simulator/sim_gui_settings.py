from dataclasses import dataclass


@dataclass
class SimGUISettings:
    gravity: float
    air_res: float
    friction: float
    temp: float
    speed: float
    show_fps: bool
    show_num: bool
    show_links: bool
    top: bool
    bottom: bool
    left: bool
    right: bool
    use_grid: bool
    grid_res_x: int
    grid_res_y: int
    delay: float
    calculate_radii_diff: bool
