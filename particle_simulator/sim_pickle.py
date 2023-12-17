import os
from typing import (
    Any,
    Dict,
    Tuple,
    List,
    TypedDict,
    Union,
)

from particle_simulator.sim_format import SimFormat
from particle_simulator.simulation_state import SimulationState

ParticlesPickle = List[Dict[str, Any]]
ParticleSettings = Dict[str, Tuple[Any]]
SimSettings = Dict[str, Tuple[Any]]
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
