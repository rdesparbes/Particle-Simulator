import pickle
from pathlib import Path
from typing import Any

import pytest

from particle_simulator.sim_pickle import SimPickle
from particle_simulator.simulation import Simulation


def diff_subset(obj: Any, reference: Any) -> Any:
    if isinstance(obj, (str, float)):
        if obj != reference:
            return obj, reference
        else:
            return None
    elif hasattr(obj, "items"):
        differences = {}
        for key, value in obj.items():
            ref_value = reference[key]
            difference = diff_subset(value, ref_value)
            if difference:
                differences[key] = difference
        return differences
    elif hasattr(obj, "__iter__"):
        differences = []
        for obj_elem, ref_elem in zip(obj, reference):
            difference = diff_subset(obj_elem, ref_elem)
            if difference:
                differences.append(difference)
        return differences
    else:
        if obj != reference:
            return obj, reference
        else:
            return None


@pytest.fixture(
    name="sim_file_name",
    scope="session",
    params=[
        "building.sim",
        "cloth.sim",
        "elliptical_mirror.sim",
        "fluid.sim",
        "fluid2.sim",
        "hanging_load.sim",
        "rainbow_wave.sim",
        "rope.sim",
        "soft_body.sim",
        "solar_system.sim",
        "wind_simulation.sim",
    ],
)
def fixture_sim_file_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(name="pickle_data", scope="session")
def fixture_pickle_data(sim_file_name: str) -> SimPickle:
    sim_file_path = Path("example_simulations", sim_file_name)
    with open(sim_file_path, "rb") as file_object:
        return pickle.load(file_object)


@pytest.fixture(name="simulation", scope="session")
def fixture_simulation(pickle_data: SimPickle) -> Simulation:
    sim = Simulation()
    sim.from_dict(pickle_data)
    return sim


def test_simulation_load_then_write_generates_subset_file(
    pickle_data: SimPickle, tmp_path: Path
) -> None:
    # Arrange
    sim = Simulation()
    sim.from_dict(pickle_data)

    # Act
    dumped_data = sim.to_dict()

    # Assert
    difference = diff_subset(obj=pickle_data, reference=dumped_data)
    assert not difference


def test_simulation_write_then_load_generates_identical_file(
    simulation: Simulation,
) -> None:
    # Arrange
    dumped_data = simulation.to_dict()

    # Act
    sim = Simulation()
    sim.from_dict(dumped_data)

    # Assert
    assert dumped_data == sim.to_dict()
