import pickle
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

from particle_simulator.color import generate_random
from particle_simulator.controller_state import ControllerState
from particle_simulator.engine.particle_factory import ParticleFactory, ParticleBuilder
from particle_simulator.engine.particle_properties import ParticleProperties
from particle_simulator.engine.simulation_data import SimulationData
from particle_simulator.sim_gui_settings import SimGUISettings

ParticlesPickle = List[Dict[str, Any]]
PickleSettings = Dict[str, Sequence[Any]]
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
        "color_entry": (p.color,),
        "mass_entry": (p.props.mass,),
        "velocity_x_entry": (p.velocity[0],),
        "velocity_y_entry": (p.velocity[1],),
        "bounciness_entry": (p.props.bounciness,),
        "do_collision_bool": (p.props.collisions,),
        "locked_bool": (p.props.locked,),
        "linked_group_bool": (p.props.linked_group_particles,),
        "attr_r_entry": (p.props.attract_r,),
        "repel_r_entry": (p.props.repel_r,),
        "attr_strength_entry": (p.props.attraction_strength,),
        "gravity_mode_bool": (p.props.gravity_mode,),
        "repel_strength_entry": (p.props.repulsion_strength,),
        "link_attr_break_entry": (p.props.link_attr_breaking_force,),
        "link_repel_break_entry": (p.props.link_repel_breaking_force,),
        "groups_entry": (p.props.group,),
        "separate_group_bool": (p.props.separate_group,),
    }


def _particles_to_dict(particles: Sequence[ParticleBuilder]) -> ParticlesPickle:
    return [
        {
            "x": p.x,
            "y": p.y,
            "v": p.velocity,
            "r": p.radius,
            "color": p.color,
            "m": p.props.mass,
            "bounciness": p.props.bounciness,
            "locked": p.props.locked,
            "collision_bool": p.props.collisions,
            "attr_r": p.props.attract_r,
            "repel_r": p.props.repel_r,
            "attr": p.props.attraction_strength,
            "repel": p.props.repulsion_strength,
            "linked_group_particles": p.props.linked_group_particles,
            "link_attr_breaking_force": p.props.link_attr_breaking_force,
            "link_repel_breaking_force": p.props.link_repel_breaking_force,
            "group": p.props.group,
            "separate_group": p.props.separate_group,
            "gravity_mode": p.props.gravity_mode,
            "link_lengths": p.link_indices_lengths,
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
    props = ParticleProperties(
        mass=float(p["mass_entry"][0]),
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
    if color is None:
        color = generate_random()
    factory = ParticleFactory(
        color=color,
        props=props,
        velocity=(float(p["velocity_x_entry"][0]), float(p["velocity_y_entry"][0])),
    )
    if radius is not None:
        factory.radius = radius
    return factory


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
    if value == "repel":
        return None
    return float(value)


def _parse_particles(particles_pickle: ParticlesPickle) -> List[ParticleBuilder]:
    particles: List[ParticleBuilder] = []
    for d in particles_pickle:
        props = ParticleProperties(
            mass=float(d["m"]),
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
        )
        v_x, v_y = d["v"]
        color = _parse_color(d["color"])
        if color is None:
            color = generate_random()
        particle = ParticleBuilder(
            x=float(d["x"]),
            y=float(d["y"]),
            color=color,
            radius=float(d["r"]),
            props=props,
            velocity=(float(v_x), float(v_y)),
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


def dump(controller_state: ControllerState, filename: str) -> None:
    data = to_dict(controller_state)
    with open(filename, "wb") as file_object:
        pickle.dump(data, file_object)


def load(filename: str) -> ControllerState:
    with open(filename, "rb") as file_object:
        data = pickle.load(file_object)
    return from_dict(data)
