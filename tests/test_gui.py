from typing import List, Dict

import pytest

from particle_simulator.engine.particle import Particle
from particle_simulator.engine.particle_properties import ParticleProperties
from particle_simulator.engine.simulation_state import SimulationState
from particle_simulator.gui.gui import GUI


def test_gui_can_display_sorted_groups_of_sim_state() -> None:
    sim_state = SimulationState(groups={"b": [], "a": []})
    gui = GUI(sim_state)
    assert gui._particle_tab.groups_entry["values"] == ("a", "b")


def test_gui_can_display_groups_of_registered_particles() -> None:
    sim_state = SimulationState(groups={})
    gui = GUI(sim_state)
    sim_state.create_group_callbacks.append(gui.create_group)

    sim_state.register_particle(Particle(0.0, 0.0, props=ParticleProperties(group="b")))
    sim_state.register_particle(Particle(0.0, 0.0, props=ParticleProperties(group="a")))

    assert gui._particle_tab.groups_entry.get() == ""
    assert gui._particle_tab.groups_entry["values"] == ("a", "b")


@pytest.mark.parametrize(
    "groups, expected_text",
    [
        ({}, ""),
        ({"b": [], "a": []}, "a"),
    ],
)
def test_gui_displays_expect_group_name(
    groups: Dict[str, List[Particle]], expected_text: str
) -> None:
    sim_state = SimulationState(groups=groups)
    gui = GUI(sim_state)
    assert gui._particle_tab.groups_entry.get() == expected_text


@pytest.mark.parametrize("collisions", [True, False])
def test_copy_selected_btn_invoke_without_selected_particles_changes_nothing(
    collisions: bool,
) -> None:
    # Arrange
    sim_state = SimulationState()
    gui = GUI(sim_state)
    gui._particle_tab.do_collision_bool.set(collisions)
    # Act
    gui._particle_tab.copy_selected_btn.invoke()
    # Assert
    assert gui._particle_tab.do_collision_bool.get() == collisions


@pytest.mark.parametrize("collisions", [True, False])
def test_copy_selected_btn_invoke_with_selected_particles_all_same_param_sets_param(
    collisions: bool,
) -> None:
    """This test verifies that a value is set if all the selected particles have the same
    value for the considered parameter
    """
    # Arrange
    sim_state = SimulationState()
    particles = [
        Particle(0.0, 0.0, props=ParticleProperties(collisions=collisions)),
        Particle(0.0, 0.0, props=ParticleProperties(collisions=collisions)),
    ]
    for particle in particles:
        sim_state.register_particle(particle)
        sim_state.select_particle(particle)
    gui = GUI(sim_state)
    # Act
    gui._particle_tab.copy_selected_btn.invoke()
    # Assert
    assert gui._particle_tab.do_collision_bool.get() == collisions


@pytest.mark.parametrize("collisions", [True, False])
def test_copy_selected_btn_invoke_with_selected_particles_heterogeneous_param_does_not_sets_param(
    collisions: bool,
) -> None:
    """This test verifies that a value is not set if at least one of the
    selected particles have a different value from the others for the
    considered parameter
    """
    # Arrange
    sim_state = SimulationState()
    particles = [
        Particle(0.0, 0.0, props=ParticleProperties(collisions=collisions)),
        Particle(0.0, 0.0, props=ParticleProperties(collisions=not collisions)),
    ]
    for particle in particles:
        sim_state.register_particle(particle)
        sim_state.select_particle(particle)
    gui = GUI(sim_state)
    # Act
    gui._particle_tab.copy_selected_btn.invoke()
    # Assert
    assert not gui._particle_tab.do_collision_bool.get()
