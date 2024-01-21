from unittest.mock import MagicMock

import pytest

from particle_simulator.particle import Particle
from particle_simulator.simulation_state import SimulationState


@pytest.fixture(name="sim_state")
def fixture_sim_state() -> SimulationState:
    return SimulationState()


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
    sim_state.select_particle(Particle(0., 0.))
    # Act
    sim_state.simulate_step()
    # Assert
    assert sim_state.selection == []
