from typing import List, Dict

import pytest

from particle_simulator.gui import GUI
from particle_simulator.particle import Particle
from particle_simulator.particle_properties import ParticleProperties
from particle_simulator.simulation_state import SimulationState


def test_gui_can_display_sorted_groups_of_sim_state() -> None:
    sim_state = SimulationState(groups={"b": [], "a": []})
    gui = GUI(sim_state)
    assert gui.groups_entry["values"] == ("a", "b")


def test_gui_can_display_groups_of_registered_particles() -> None:
    sim_state = SimulationState(groups={})
    gui = GUI(sim_state)
    sim_state.create_group_callbacks.append(gui.create_group)

    sim_state.register_particle(Particle(0.0, 0.0, props=ParticleProperties(group="b")))
    sim_state.register_particle(Particle(0.0, 0.0, props=ParticleProperties(group="a")))

    assert gui.groups_entry.get() == ""
    assert gui.groups_entry["values"] == ("a", "b")


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
    assert gui.groups_entry.get() == expected_text
