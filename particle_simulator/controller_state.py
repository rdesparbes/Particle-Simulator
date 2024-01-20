from dataclasses import dataclass, field
from typing import Sequence

from particle_simulator.particle_factory import ParticleFactory, ParticleBuilder
from particle_simulator.sim_gui_settings import SimGUISettings
from particle_simulator.simulation_data import SimulationData


@dataclass
class ControllerState:
    sim_data: SimulationData
    gui_settings: SimGUISettings
    gui_particle_state: ParticleFactory
    particles: Sequence[ParticleBuilder] = field(default_factory=list)
