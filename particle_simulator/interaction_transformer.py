from typing import Callable, Mapping, NamedTuple, List

from particle_simulator.particle import Particle
from particle_simulator.particle_interaction import ParticleInteraction

InteractionTransformer = Callable[
    [Particle, Mapping[Particle, ParticleInteraction]], None
]


class Link(NamedTuple):
    particle_a: Particle
    particle_b: Particle
    percentage: float


def compute_links(
    particle: Particle,
    interactions: Mapping[Particle, ParticleInteraction],
    links: List[Link],
) -> None:
    for near_particle, interaction in interactions.items():
        if (
            interaction.link_percentage is not None
            and interaction.link_percentage <= 1.0
        ):
            links.append(Link(particle, near_particle, interaction.link_percentage))
