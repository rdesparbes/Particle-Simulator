from dataclasses import dataclass, field
from typing import List

from particle_simulator.particle_data import ParticleData
from particle_simulator.particle_factory import ParticleFactory
from particle_simulator.sim_gui_settings import SimGUISettings
from particle_simulator.simulation_state import SimulationState


@dataclass
class ControllerState:
    sim_state: SimulationState
    gui_settings: SimGUISettings
    gui_particle_state: ParticleFactory
    particles: List[ParticleData] = field(default_factory=list)
