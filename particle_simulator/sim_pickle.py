import re
from typing import (
    Any,
    Dict,
    Tuple,
    List,
    TypedDict,
    Optional,
    Sequence,
)

import numpy as np

from particle_simulator.controller_state import ControllerState
from particle_simulator.particle_data import ParticleData
from particle_simulator.particle_factory import ParticleFactory
from particle_simulator.sim_gui_settings import SimGUISettings
from particle_simulator.simulation_data import SimulationData

ParticlesPickle = List[Dict[str, Any]]
PickleSettings = Dict[str, Tuple[Any]]
SimPickle = TypedDict(
    "SimPickle",
    {
        "particles": ParticlesPickle,
        "particle-settings": PickleSettings,
        "sim-settings": PickleSettings,
    },
)


def _sim_settings_to_dict(sim_data: SimulationData) -> PickleSettings:
    s = sim_data
    return {
        "gravity_entry": (s.g,),
        "air_res_entry": (s.air_res,),
        "friction_entry": (s.ground_friction,),
        "temp_sc": (s.temperature,),
        "speed_sc": (s.speed,),
        "top_bool": (s.top,),
        "bottom_bool": (s.bottom,),
        "left_bool": (s.left,),
        "right_bool": (s.right,),
        "grid_bool": (s.use_grid,),
        "calculate_radii_diff_bool": (s.calculate_radii_diff,),
        "g_dir": ((float(s.g_dir[0]), float(s.g_dir[1])),),
        "wind_force": ((float(s.wind_force[0]), float(s.wind_force[1])),),
        "stress_visualization": (s.stress_visualization,),
        "bg_color": ((s.bg_color,),),
        "void_edges": (s.void_edges,),
        "code": (s.code,),
    }


def _gui_settings_to_dict(gui_settings: SimGUISettings) -> PickleSettings:
    g = gui_settings
    return {
        "grid_res_x_value": (g.grid_res_y,),
        "grid_res_y_value": (g.grid_res_y,),
        "delay_entry": (g.delay,),
        "show_fps": (g.show_fps,),
        "show_num": (g.show_num,),
        "show_links": (g.show_links,),
    }


def _particle_settings_to_dict(particle_factory: ParticleFactory) -> PickleSettings:
    p = particle_factory
    return {
        "radius_entry": (p.radius,),
        "color_entry": (p.color or "random",),
        "mass_entry": (p.mass,),
        "velocity_x_entry": (p.velocity[0],),
        "velocity_y_entry": (p.velocity[1],),
        "bounciness_entry": (p.bounciness,),
        "do_collision_bool": (p.collisions,),
        "locked_bool": (p.locked,),
        "linked_group_bool": (p.linked_group_particles,),
        "attr_r_entry": (p.attract_r,),
        "repel_r_entry": (p.repel_r,),
        "attr_strength_entry": (p.attraction_strength,),
        "gravity_mode_bool": (p.gravity_mode,),
        "repel_strength_entry": (p.repulsion_strength,),
        "link_attr_break_entry": (p.link_attr_breaking_force,),
        "link_repel_break_entry": (p.link_repel_breaking_force,),
        "groups_entry": (p.group,),
        "separate_group_bool": (p.separate_group,),
    }


def _particles_to_dict(particles: Sequence[ParticleData]) -> ParticlesPickle:
    return [
        {
            "x": p.x,
            "y": p.y,
            "v": p.velocity,
            "r": p.radius,
            "color": p.color,
            "m": p.mass,
            "bounciness": p.bounciness,
            "locked": p.locked,
            "collision_bool": p.collisions,
            "attr_r": p.attract_r,
            "repel_r": p.repel_r,
            "attr": p.attraction_strength,
            "repel": p.repulsion_strength,
            "linked_group_particles": p.linked_group_particles,
            "link_attr_breaking_force": p.link_attr_breaking_force,
            "link_repel_breaking_force": p.link_repel_breaking_force,
            "group": p.group,
            "separate_group": p.separate_group,
            "gravity_mode": p.gravity_mode,
            "link_lengths": {
                particles.index(particle): value
                for particle, value in p.link_lengths.items()
                if particle in particles
            },
        }
        for p in particles
    ]


def to_dict(controller_state: ControllerState) -> SimPickle:
    sim_settings = _sim_settings_to_dict(controller_state.sim_data)
    gui_settings = _gui_settings_to_dict(controller_state.gui_settings)
    particle_settings = _particle_settings_to_dict(controller_state.gui_particle_state)
    particles = _particles_to_dict(controller_state.particles)

    return {
        "particles": particles,
        "particle-settings": particle_settings,
        "sim-settings": {**sim_settings, **gui_settings},
    }


def _parse_color(color_any: Any) -> Optional[Tuple[int, int, int]]:
    if color_any == "random":
        return None

    if isinstance(color_any, str):
        match = re.search(r"(?P<blue>\d+), *(?P<green>\d+), *(?P<red>\d+)", color_any)
        if match is None:
            raise ValueError(f"Impossible to parse color: {color_any}")
        return (
            int(match.group("blue")),
            int(match.group("green")),
            int(match.group("red")),
        )
    return int(color_any[0]), int(color_any[1]), int(color_any[2])


def _parse_radius(radius_any: Any) -> Optional[float]:
    if radius_any is None:
        return None
    try:
        return float(radius_any)
    except ValueError:
        return None


def _parse_particle_settings(particle_settings: PickleSettings) -> ParticleFactory:
    p = particle_settings
    color = _parse_color(p["color_entry"][0])
    radius = _parse_radius(p["radius_entry"][0])
    return ParticleFactory(
        radius=radius,
        color=color,
        mass=float(p["mass_entry"][0]),
        velocity=(float(p["velocity_x_entry"][0]), float(p["velocity_y_entry"][0])),
        bounciness=float(p["bounciness_entry"][0]),
        collisions=bool(p["do_collision_bool"][0]),
        locked=bool(p["locked_bool"][0]),
        linked_group_particles=bool(p["linked_group_bool"][0]),
        attract_r=float(p["attr_r_entry"][0]),
        repel_r=float(p["repel_r_entry"][0]),
        attraction_strength=float(p["attr_strength_entry"][0]),
        gravity_mode=bool(p["gravity_mode_bool"][0]),
        repulsion_strength=float(p["repel_strength_entry"][0]),
        link_attr_breaking_force=float(p["link_attr_break_entry"][0]),
        link_repel_breaking_force=float(p["link_repel_break_entry"][0]),
        group=str(p["groups_entry"][0]),
        separate_group=bool(p["separate_group_bool"][0]),
    )


def _extract_sim_gui_settings(sim_settings: PickleSettings) -> SimGUISettings:
    s = sim_settings
    return SimGUISettings(
        delay=float(s["delay_entry"][0]),
        grid_res_x=int(s["grid_res_x_value"][0]),
        grid_res_y=int(s["grid_res_y_value"][0]),
        show_fps=bool(s["show_fps"][0]),
        show_num=bool(s["show_num"][0]),
        show_links=bool(s["show_links"][0]),
    )


def _extract_sim_settings(sim_pickle: PickleSettings) -> SimulationData:
    s = sim_pickle
    g_dir_x, g_dir_y = s["g_dir"][0]
    wind_dir_x, wind_dir_y = s["wind_force"][0]
    bg_color = s["bg_color"][0][0]
    return SimulationData(
        g=float(s["gravity_entry"][0]),
        air_res=float(s["air_res_entry"][0]),
        ground_friction=float(s["friction_entry"][0]),
        temperature=float(s["temp_sc"][0]),
        speed=float(s["speed_sc"][0]),
        top=bool(s["top_bool"][0]),
        bottom=bool(s["bottom_bool"][0]),
        left=bool(s["left_bool"][0]),
        right=bool(s["right_bool"][0]),
        use_grid=bool(s["grid_bool"][0]),
        calculate_radii_diff=bool(s["calculate_radii_diff_bool"][0]),
        g_dir=np.array([float(g_dir_x), float(g_dir_y)]),
        wind_force=np.array([float(wind_dir_x), float(wind_dir_y)]),
        stress_visualization=bool(s["stress_visualization"][0]),
        bg_color=(int(bg_color[0]), int(bg_color[1]), int(bg_color[2])),
        void_edges=bool(s["void_edges"][0]),
        code=str(s.get("code", [""])[0]),
    )


def _parse_repel_r(value: Any) -> Optional[float]:
    if isinstance(value, (float, type(None))):
        return value
    elif value == "repel":
        return None
    return float(value)


def _parse_particles(particles_pickle: ParticlesPickle) -> List[ParticleData]:
    particles: List[ParticleData] = []
    for d in particles_pickle:
        v_x, v_y = d["v"]
        particle = ParticleData(
            x=float(d["x"]),
            y=float(d["y"]),
            color=d["color"],
            mass=float(d["m"]),
            velocity=np.array([float(v_x), float(v_y)]),
            bounciness=float(d["bounciness"]),
            locked=bool(d["locked"]),
            collisions=bool(d["collision_bool"]),
            attract_r=float(d["attr_r"]),
            repel_r=float(d["repel_r"]),
            attraction_strength=float(d["attr"]),
            repulsion_strength=float(d["repel"]),
            linked_group_particles=bool(d["linked_group_particles"]),
            link_attr_breaking_force=float(d["link_attr_breaking_force"]),
            link_repel_breaking_force=float(d["link_repel_breaking_force"]),
            group=str(d["group"]),
            separate_group=bool(d["separate_group"]),
            gravity_mode=bool(d["gravity_mode"]),
            link_indices_lengths={
                int(p_index): _parse_repel_r(length)
                for p_index, length in d["link_lengths"].items()
            },
        )
        particles.append(particle)
    return particles


def from_dict(controller_pickle: SimPickle) -> ControllerState:
    sim_settings = _extract_sim_settings(controller_pickle["sim-settings"])
    sim_gui_settings = _extract_sim_gui_settings(controller_pickle["sim-settings"])
    particle_settings = _parse_particle_settings(controller_pickle["particle-settings"])
    particles = _parse_particles(controller_pickle["particles"])

    return ControllerState(
        sim_data=sim_settings,
        gui_settings=sim_gui_settings,
        gui_particle_state=particle_settings,
        particles=particles,
    )
