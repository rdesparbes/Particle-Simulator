from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass(slots=True, frozen=True)
class ParticleInteraction:
    force: npt.NDArray[np.float_] = field(default_factory=lambda: np.zeros(2))
    link_percentage: Optional[float] = None

    def __neg__(self) -> "ParticleInteraction":
        return ParticleInteraction(-self.force, self.link_percentage)
