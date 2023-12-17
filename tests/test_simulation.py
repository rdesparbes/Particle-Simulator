import pickle
from pathlib import Path
from typing import Iterable, Union, Literal, Tuple

import pytest

from particle_simulator.sim_pickle import SimPickle
from particle_simulator.simulation import Simulation


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


@pytest.mark.parametrize(
    "color_any, expected_color",
    [
        ([0, 1, 2], (0, 1, 2)),
        ([0, 1, 2, 3], (0, 1, 2)),
        ("random", "random"),
        ("[0, 1, 2]", (0, 1, 2)),
        ("(0, 1, 2)", (0, 1, 2)),
        ("(0, 1,   2)", (0, 1, 2)),
    ],
)
def test_parse_color(
    color_any: Union[str, Iterable[float]],
    expected_color: Union[Tuple[int, int, int], Literal["random"]],
) -> None:
    assert Simulation._parse_color(color_any) == expected_color
