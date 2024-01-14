import pytest

from particle_simulator.simulation_state import SimulationState


@pytest.fixture(name="sim_state")
def fixture_sim_state() -> SimulationState:
    return SimulationState()


def test_add_group(sim_state: SimulationState) -> None:
    assert sim_state.groups == {"group1": []}
    assert sim_state.add_group() == "group2"
    assert sim_state.groups == {"group1": [], "group2": []}
