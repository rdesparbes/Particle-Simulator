from typing import Optional, NamedTuple

import numpy as np
import numpy.typing as npt


class ParticleInteraction(NamedTuple):
    force: npt.NDArray[np.float_]
    link_percentage: Optional[float] = None

    def __neg__(self) -> "ParticleInteraction":
        return ParticleInteraction(-self.force, self.link_percentage)
