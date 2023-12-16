import os
from typing import (
    Any,
    Dict,
    Tuple,
    Literal,
    List,
    TypedDict,
    Union,
)

from particle_simulator.sim_format import SimFormat
from particle_simulator.simulation_state import SimulationState

AttributeType = Literal["set", "entry", "var"]
ParticlesPickle = List[Dict[str, Any]]
ParticleSettings = Dict[str, Tuple[Any, AttributeType]]
SimSettings = Dict[str, Tuple[Any, AttributeType]]
SimPickle = TypedDict(
    "SimPickle",
    {
        "particles": ParticlesPickle,
        "particle-settings": ParticleSettings,
        "sim-settings": SimSettings,
    },
)


class PickleSimFormat(SimFormat):
    def load(self, source: Union[str, bytes, os.PathLike]) -> SimulationState:
        pass

    def save(
        self, destination: Union[str, bytes, os.PathLike], sim: SimulationState
    ) -> None:
        pass
