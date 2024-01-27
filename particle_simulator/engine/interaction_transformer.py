from typing import Callable, Mapping, NamedTuple, List

from particle_simulator.engine.particle import Particle, unlink_particles
from particle_simulator.engine.particle_interaction import ParticleInteraction

InteractionTransformer = Callable[
    [Particle, Mapping[Particle, ParticleInteraction]], None
]


class Link(NamedTuple):
    particle_a: Particle
    particle_b: Particle
    percentage: float


def remove_broken_links(
    particle: Particle, interactions: Mapping[Particle, ParticleInteraction]
) -> None:
    for near_particle, interaction in interactions.items():
        if (
            interaction.link_percentage is not None
            and interaction.link_percentage > 1.0
        ):
            unlink_particles([particle, near_particle])


def apply_collisions(
    particle: Particle, interactions: Mapping[Particle, ParticleInteraction]
) -> None:
    for near_particle in interactions:
        particle.compute_collision(near_particle)


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
