from typing import (
    Any,
    Dict,
    Tuple,
    List,
    TypedDict,
)

from particle_simulator.controller_state import ControllerState

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


def to_dict(controller_state: ControllerState) -> SimPickle:
    g = controller_state.gui_settings
    s = controller_state.sim_state
    sim_settings: SimSettings = {
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
        "bg_color": (s.bg_color,),
        "void_edges": (s.void_edges,),
        "code": (s.code,),
        "grid_res_x_value": (g.grid_res_y,),
        "grid_res_y_value": (g.grid_res_y,),
        "delay_entry": (g.delay,),
        "show_fps": (g.show_fps,),
        "show_num": (g.show_num,),
        "show_links": (g.show_links,),
    }
    p = controller_state.gui_particle_state
    particle_settings: ParticleSettings = {
        "radius_entry": (p.radius,),
        "color_entry": (p.color,),
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

    return {
        "particles": [
            particle.return_dict(index_source=s.particles) for particle in s.particles
        ],
        "particle-settings": particle_settings,
        "sim-settings": sim_settings,
    }
