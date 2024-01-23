from unittest.mock import MagicMock

import numpy as np
import pytest

from particle_simulator.particle import Particle
from particle_simulator.simulation_state import SimulationState


@pytest.fixture(name="sim_state")
def fixture_sim_state() -> SimulationState:
    return SimulationState(
        temperature=0.0,  # Set to 0 to keep the simulation deterministic
        paused=False,
    )


def test_add_group(sim_state: SimulationState) -> None:
    assert sim_state.groups == {"group1": []}
    assert sim_state.add_group() == "group2"
    assert sim_state.groups == {"group1": [], "group2": []}


def test_register_particle_triggers_create_group_callbacks(
    sim_state: SimulationState,
) -> None:
    # Arrange
    create_group_callback = MagicMock()
    sim_state.groups = {}
    sim_state.create_group_callbacks.append(create_group_callback)
    # Act
    sim_state.register_particle(Particle(0.0, 0.0))
    # Assert
    create_group_callback.assert_called_once_with("group1")


def test_simulate_step_when_toggle_to_unpause_clears_selection(
    sim_state: SimulationState,
) -> None:
    # Arrange
    sim_state.toggle_pause = True
    sim_state.paused = True
    sim_state.select_particle(Particle(0.0, 0.0))
    # Act
    sim_state.simulate_step()
    # Assert
    assert sim_state.selection == []


def test_simulate_step_given_no_external_forces_makes_two_identical_particles_travel_the_same_distance(
    sim_state: SimulationState,
) -> None:
    # Arrange
    init_pos = (100.0, 100.0)
    p1 = Particle(*init_pos)
    p2 = Particle(*init_pos)
    sim_state.register_particle(p1)
    sim_state.register_particle(p2)
    sim_state.wind_force[:] = 0.0
    sim_state.g = 0.0
    # Act
    sim_state.simulate_step()
    # Assert
    assert np.isclose(p1.distance(*init_pos), p2.distance(*init_pos))


def test_simulate_step_when_no_edges_removes_particle_about_to_leave_canvas(
    sim_state: SimulationState,
) -> None:
    # Arrange
    p1 = Particle(1.0, 1.0, velocity=np.array([0.0, -10.0]))
    sim_state.register_particle(p1)
    sim_state.void_edges = True
    sim_state.top = False
    # Act
    sim_state.simulate_step()
    # Assert
    assert not sim_state.particles


def test_simulate_step_when_edges_makes_particle_about_to_leave_canvas_bounce(
    sim_state: SimulationState,
) -> None:
    # Arrange
    p1 = Particle(1.0, 1.0, velocity=np.array([0.0, -10.0]))
    sim_state.register_particle(p1)
    # Act
    sim_state.simulate_step()
    # Assert
    assert len(sim_state.particles) == 1
