from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from particle_simulator.particle import Particle
from particle_simulator.simulation_state import SimulationState, Link


def _draw_line(
    image: npt.NDArray[np.uint8],
    p1: Particle,
    p2: Particle,
    color: Tuple[int, int, int],
) -> None:
    cv2.line(
        image,
        (int(p1.x), int(p1.y)),
        (int(p2.x), int(p2.y)),
        color,
        1,
    )


def paint_image(
    state: SimulationState, link_colors: Optional[Iterable[Link]] = None
) -> npt.NDArray[np.uint8]:
    image = np.full(
        (state.height, state.width, 3),
        state.bg_color,
        dtype=np.uint8,
    )
    if state.show_links:
        if link_colors is not None:
            for p1, p2, percent in link_colors:
                color = (
                    max(int(255 * percent), 235),
                    int(235 * (1 - percent)),
                    int(235 * (1 - percent)),
                )
                _draw_line(image, p1, p2, color)
        else:
            for p1 in state.particles:
                for p2 in p1.link_lengths:
                    _draw_line(image, p1, p2, (235, 235, 235))
    for particle in state.particles:
        cv2.circle(
            image,
            (int(particle.x), int(particle.y)),
            int(particle.radius),
            particle.color,
            -1,
        )
    for particle in state.selection:
        cv2.circle(
            image,
            (int(particle.x), int(particle.y)),
            int(particle.radius),
            (0, 0, 255),
            2,
        )
    cv2.circle(image, (state.mx, state.my), int(state.mr), (127, 127, 127))
    return image
