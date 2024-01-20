from copy import copy
from typing import Sequence, List

import numpy as np

from particle_simulator.particle import Particle
from particle_simulator.particle_factory import ParticleBuilder


def builders_to_particles(builders: Sequence[ParticleBuilder]) -> List[Particle]:
    particles = [
        Particle(
            p.x,
            p.y,
            radius=p.radius,
            color=p.color,
            props=copy(p.props),
            velocity=np.array(p.velocity),
        )
        for p in builders
    ]
    for particle, builder in zip(particles, builders):
        particle.link_lengths = {
            particles[index]: value
            for index, value in builder.link_indices_lengths.items()
        }
    return particles


def particles_to_builders(particles: Sequence[Particle]) -> List[ParticleBuilder]:
    return [
        ParticleBuilder(
            color=p.color,
            props=copy(p.props),
            velocity=(float(p.velocity[0]), float(p.velocity[1])),
            radius=p.radius,
            x=p.x,
            y=p.y,
            link_indices_lengths={
                particles.index(particle): value
                for particle, value in p.link_lengths.items()
                if particle in particles
            },
        )
        for p in particles
    ]
