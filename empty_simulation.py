from particle_simulator.factory import build_simulation
from particle_simulator.simulation import Simulation

sim_controller: Simulation = build_simulation()
sim_controller.simulate()
