import os
from abc import ABC, abstractmethod
from typing import Union

from particle_simulator.simulation_state import SimulationState


class SimFormat(ABC):
    @abstractmethod
    def load(self, source: Union[str, bytes, os.PathLike]) -> SimulationState:
        ...

    @abstractmethod
    def save(
        self, destination: Union[str, bytes, os.PathLike], sim: SimulationState
    ) -> None:
        ...
