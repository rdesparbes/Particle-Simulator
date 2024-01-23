from typing import Tuple
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
    sim_state.paused = True
    sim_state.select_particle(Particle(0.0, 0.0))
    sim_state.toggle_paused()
    # Act
    sim_state.simulate_step()
    # Assert
    assert sim_state.selection == []
    assert not sim_state.paused


@pytest.mark.parametrize(
    "init_pos_1, init_pos_2",
    [((10.0, 10.0), (10.0, 10.0)), ((10.0, 10.0), (11.0, 10.0))],
)
@pytest.mark.parametrize("calculate_radii_diff", [True, False])
def test_simulate_step_given_no_external_forces_makes_two_identical_near_particles_travel_with_opposite_vectors(
    sim_state: SimulationState,
    init_pos_1: Tuple[float, float],
    init_pos_2: Tuple[float, float],
    calculate_radii_diff: bool,
) -> None:
    # Arrange
    p1 = Particle(*init_pos_1)
    p2 = Particle(*init_pos_2)
    sim_state.register_particle(p1)
    sim_state.register_particle(p2)
    sim_state.wind_force[:] = 0.0
    sim_state.g = 0.0
    sim_state.calculate_radii_diff = calculate_radii_diff
    # Act
    sim_state.simulate_step()
    # Assert
    p1_delta_pos = np.subtract((p1.x, p1.y), init_pos_1)
    p2_delta_pos = np.subtract((p2.x, p2.y), init_pos_2)
    assert np.all(p1_delta_pos == -p2_delta_pos)


def test_simulate_step_when_no_edges_removes_particle_about_to_leave_canvas(
    sim_state: SimulationState,
) -> None:
    # Arrange
    p1 = Particle(5.0, 5.0, velocity=np.array([0.0, -10.0]))
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
    p1 = Particle(5.0, 5.0, velocity=np.array([0.0, -10.0]))
    sim_state.register_particle(p1)
    # Act
    sim_state.simulate_step()
    # Assert
    assert len(sim_state.particles) == 1


@pytest.mark.parametrize("paused, expected_to_move", [(True, False), (False, True)])
def test_simulate_step_given_paused_freezes_the_particles(
    sim_state: SimulationState, paused: bool, expected_to_move: bool
) -> None:
    # Arrange
    init_pos = (5.0, 5.0)
    p1 = Particle(*init_pos, velocity=np.array([0.0, -10.0]))
    sim_state.register_particle(p1)
    sim_state.paused = paused
    # Act
    sim_state.simulate_step()
    # Assert
    delta_pos = np.subtract((p1.x, p1.y), init_pos)
    has_moved = np.any(delta_pos)
    assert has_moved == expected_to_move
