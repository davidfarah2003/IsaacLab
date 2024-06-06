from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def cam_rgb(env: ManagerBasedRLEnv) -> torch.Tensor():
    return env.scene["camera"].data.output["rgb"]


def cam_depth(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["camera"].data.output["distance_to_image_plane"]


def l2_distance(env: ManagerBasedRLEnv,
                cfg1: SceneEntityCfg = SceneEntityCfg("cube"),
                cfg2: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    obj1: RigidObject = env.scene[cfg1.name]
    obj2: RigidObject = env.scene[cfg2.name]

    sub = obj1.data.root_pos_w[:] - obj2.data.root_pos_w[:]
    sub[:, 2] = 0  # set all z to 0 as it's not needed
    return sub.norm(dim=1)  # Should be tensor of dim (nb_envs)


def is_close_to(env: ManagerBasedRLEnv,
                cfg1: SceneEntityCfg = SceneEntityCfg("cube"),
                cfg2: SceneEntityCfg = SceneEntityCfg("ball"),
                threshold=0.2):
    return l2_distance(env, cfg1, cfg2) < threshold  # Should be bool tensor of dim (nb_envs)


def is_close_once(env: ManagerBasedRLEnv, func=is_close_to):
    is_close = func(env)
    ret = torch.logical_and(is_close, ~env.close_reward_given)
    env.close_reward_given = torch.logical_or(env.close_reward_given, is_close)
    return ret


def angle_diff(env: ManagerBasedRLEnv,
               cfg1: SceneEntityCfg = SceneEntityCfg("cube"),
               cfg2: SceneEntityCfg = SceneEntityCfg("ball"),
               epsilon=1e-8):
    obj1: RigidObject = env.scene[cfg1.name]
    obj2: RigidObject = env.scene[cfg2.name]

    vect_roots = obj2.data.root_pos_w[:] - obj1.data.root_pos_w[:]
    vect_roots[:, 2] = 0  # set all z to 0 as it's not needed

    # Compute the norm of vect_roots and add epsilon to avoid division by zero if norm is 0
    vect_roots_norm = vect_roots / (torch.norm(vect_roots, dim=1, keepdim=True).expand_as(vect_roots) + epsilon)

    # Define the x-axis in the same frame
    x_axis = torch.tensor((1, 0, 0), device=env.device).expand_as(vect_roots_norm)

    # Compute cosine similarity with x axis (in robot frame)
    cosine_sim = F.cosine_similarity(vect_roots_norm, x_axis)
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)  # clip to [-1, 1] to avoid NaNs in acos
    angle_in_radians = torch.acos(cosine_sim)  # Compute the angle in radians
    return torch.rad2deg(angle_in_radians)  # Compute angle in degrees


def is_angle_close(env: ManagerBasedRLEnv,
                   cfg1: SceneEntityCfg = SceneEntityCfg("cube"),
                   cfg2: SceneEntityCfg = SceneEntityCfg("ball"),
                   threshold=10):
    return torch.abs(angle_diff(env, cfg1, cfg2)) < threshold


def reached_target(env: ManagerBasedRLEnv, dist_threshold=0.2, angle_threshold=10):
    reached = torch.logical_and(is_close_to(env, threshold=dist_threshold),
                                is_angle_close(env, threshold=angle_threshold))
    # Apply mask to only give reward the first time
    ret = torch.logical_and(reached, ~env.target_reward_given)
    # Update the target reward given flags
    env.target_reward_given = torch.logical_or(env.target_reward_given, reached)
    return ret


def reset_env_params(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    env.close_reward_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.target_reward_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def got_illegal_contacts(env: ManagerBasedRLEnv, threshold=0.01):
    forces = contact_forces(env)
    forces[:, :, 2] = 0     # set z components to 0 (gravity)
    print(forces)
    return forces.norm(dim=1) > threshold


def contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")):
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    return sensor.data.net_forces_w