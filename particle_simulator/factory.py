from typing import Tuple

from particle_simulator.engine.simulation_state import SimulationState
from particle_simulator.gui.gui import GUI
from particle_simulator.simulation import Simulation


def build_simulation(
    width: int = 650,
    height: int = 600,
    title: str = "Simulation",
    gridres: Tuple[int, int] = (50, 50),
    temperature: float = 0,
    g: float = 0.1,
    air_res: float = 0.05,
    ground_friction: float = 0,
    fps_update_delay: float = 0.5,
) -> Simulation:
    state: SimulationState = SimulationState(
        width=width,
        height=height,
        temperature=temperature,
        g=g,
        air_res=air_res,
        ground_friction=ground_friction,
        grid_res_x=gridres[0],
        grid_res_y=gridres[1],
    )
    gui = GUI(state, title)
    simulation = Simulation(state, gui, fps_update_delay)
    gui.register_controller(simulation)
    return simulation
