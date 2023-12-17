from dataclasses import dataclass

from particle_simulator.particle_state import ParticleState
from particle_simulator.sim_gui_settings import SimGUISettings
from particle_simulator.simulation_state import SimulationState


@dataclass
class ControllerState:
    sim_state: SimulationState
    gui_settings: SimGUISettings
    gui_particle_state: ParticleState
