import pickle
from pathlib import Path
from typing import Any, Dict

import pytest

from particle_simulator.simulation import Simulation


def is_subset(obj: Any, reference: Any) -> bool:
    if not isinstance(obj, type(reference)):
        return False

    if isinstance(obj, Dict):
        for key, value in obj.items():
            try:
                ref_value = reference[key]
            except KeyError:
                return False
            if not is_subset(value, ref_value):
                return False
        return True
    return obj == reference


@pytest.mark.parametrize(
    "sim_file_name",
    [
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
def test_simulation_load_then_write_generates_identical_file(
    sim_file_name: str, tmp_path: Path
) -> None:
    # Arrange
    sim_file_path = Path("example_simulations", sim_file_name)
    sim = Simulation()
    with open(sim_file_path, "rb") as file_object:
        loaded_data = pickle.load(file_object)

    # Act
    sim.save_manager.from_dict(loaded_data)
    dumped_data = sim.save_manager.to_dict()

    # Assert
    assert is_subset(loaded_data, dumped_data)
