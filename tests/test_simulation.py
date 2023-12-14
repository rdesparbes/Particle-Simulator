import pickle
from pathlib import Path
from typing import Any

import pytest

from particle_simulator.simulation import Simulation


def assert_is_subset(obj: Any, reference: Any) -> None:
    if isinstance(obj, (str, float)):
        assert obj == reference
    elif hasattr(obj, "items"):
        for key, value in obj.items():
            ref_value = reference[key]
            assert_is_subset(value, ref_value)
    elif hasattr(obj, "__iter__"):
        for obj_elem, ref_elem in zip(obj, reference):
            assert_is_subset(obj_elem, ref_elem)
    else:
        assert obj == reference


@pytest.fixture(
    name="sim_file_name",
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


def test_simulation_load_then_write_generates_identical_file(
    sim_file_name: str, tmp_path: Path
) -> None:
    # Arrange
    sim_file_path = Path("example_simulations", sim_file_name)
    sim = Simulation()
    with open(sim_file_path, "rb") as file_object:
        loaded_data = pickle.load(file_object)

    # Act
    sim.from_dict(loaded_data)
    dumped_data = sim.to_dict()

    # Assert
    assert_is_subset(loaded_data, dumped_data)
