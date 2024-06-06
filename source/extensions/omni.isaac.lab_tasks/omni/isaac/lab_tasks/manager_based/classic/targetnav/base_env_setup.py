import torch

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg, AssetBase
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, ContactSensor, Camera
from omni.isaac.lab.sim import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
import torch.nn.functional as F


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=GroundPlaneCfg())

    room = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room",
                        spawn=UsdFileCfg(usd_path="omniverse://localhost/Library/assets/simple_room/simple_room.usd"),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0.8)))

    # add cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.70, 0.31, 0.40),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0,
                                                         max_linear_velocity=3,
                                                         max_angular_velocity=80,
                                                         disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0)),
            activate_contact_sensors=True
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6875), lin_vel=(0, 0, 0)),
    )

    ball = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/ball",
                          spawn=sim_utils.SphereCfg(radius=0.1,
                                                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                                    mass_props=sim_utils.MassPropertiesCfg(mass=5),
                                                    collision_props=sim_utils.CollisionPropertiesCfg(),
                                                    visual_material=sim_utils.PreviewSurfaceCfg(
                                                        diffuse_color=(1.0, 0.0, 0.0), metallic=0),
                                                    ),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=(1, 1, 0.67)))

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/robot", update_period=0.0, history_length=6, debug_vis=False
    )

    # sensors
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/robot/front_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.4, 0.0, 0.23), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )
    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/front_cam",
        update_period=0.1,
        height=128,
        width=128,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.4, 0.0, 0.23), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0)
    )


class CubeActionTerm(ActionTerm):
    """Simple action term that implements a PD controller to track a target position.

    The action term is applied to the cube asset. It involves two steps:

    1. **Process the raw actions**: Typically, this includes any transformations of the raw actions
       that are required to map them to the desired space. This is called once per environment step.
    2. **Apply the processed actions**: This step applies the processed actions to the asset.
       It is called once per simulation step.

    In this case, the action term simply applies the raw actions to the cube asset. The raw actions
    are the desired target positions of the cube in the environment frame. The pre-processing step
    simply copies the raw actions to the processed actions as no additional processing is required.
    The processed actions are then applied to the cube asset by implementing a PD controller to
    track the target position.
    """

    _asset: RigidObject
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: 'CubeActionTermCfg', env: ManagerBasedEnv):
        # call super constructor
        super().__init__(cfg, env)
        # create buffers
        self._raw_actions = torch.zeros(env.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 6, device=self.device)
        self._vel_command = torch.zeros(self.num_envs, 6, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions

        # set z, pitch and roll velocities to 0
        self._processed_actions = self._raw_actions
        self._processed_actions[:, 2] = self._asset.data.root_lin_vel_w[:, 2]
        self._processed_actions[:, 3:5] = 0

    def apply_actions(self):
        self._asset.write_root_velocity_to_sim(self._processed_actions)


@configclass
class CubeActionTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = CubeActionTerm
    """The class corresponding to the action term."""


@configclass
class ActionsCfg:
    vel: CubeActionTermCfg = CubeActionTermCfg(asset_name="cube")


##
# Custom observation terms
##
def cam_data(env: ManagerBasedRLEnv) -> torch.Tensor():
    cam : Camera = env.scene["camera"]

    # Extract the first three channels of the RGB tensor
    rgb_trimmed: torch.Tensor = cam.data.output["rgb"][:, :, :, :3] / 255.0   # Normalize to 0,1

    print(rgb_trimmed.max())
    # Expand the distance tensor to have a singleton fourth dimension
    distance_expanded = cam.data.output["distance_to_image_plane"].unsqueeze(-1)
    print(cam.data.output["distance_to_image_plane"].min())

    # Concatenate along the last dimension
    combined_tensor = torch.cat((rgb_trimmed, distance_expanded), dim=-1)

    # Flatten all dimensions after the 0th dimension into a single vector
    num_envs = combined_tensor.size(0)
    combined_tensor = combined_tensor.view(num_envs, -1)

    # return combined_tensor.view(env.num_envs, combined_tensor.size(1), combined_tensor.size(2), 4)
    return combined_tensor.view(env.num_envs, combined_tensor.size(1))


def l2_distance(env: ManagerBasedRLEnv,
                cfg1: SceneEntityCfg = SceneEntityCfg("cube"),
                cfg2: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    obj1: RigidObject = env.scene[cfg1.name]
    obj2: RigidObject = env.scene[cfg2.name]

    sub = obj1.data.root_pos_w[:] - obj2.data.root_pos_w[:]
    sub[:, 2] = 0  # set all z to 0 as it's not needed
    return torch.norm(sub, dim=1).view(env.num_envs, -1)  # Should be tensor of dim (nb_envs)


def is_close_to(env: ManagerBasedRLEnv,
                cfg1: SceneEntityCfg = SceneEntityCfg("cube"),
                cfg2: SceneEntityCfg = SceneEntityCfg("ball"),
                threshold=0.2):
    return (l2_distance(env, cfg1, cfg2) < threshold).view(env.num_envs).float()  # Should be bool tensor of dim (nb_envs)


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
    return (torch.abs(angle_diff(env, cfg1, cfg2)) < threshold).float()


def reached_target(env: ManagerBasedRLEnv, dist_threshold=0.2, angle_threshold=10):
    reached = torch.logical_and(is_close_to(env, threshold=dist_threshold),
                                is_angle_close(env, threshold=angle_threshold))
    # Apply mask to only give reward the first time
    ret = torch.logical_and(reached, ~env.target_reward_given)
    # Update the target reward given flags
    env.target_reward_given = torch.logical_or(env.target_reward_given, reached)
    return ret.view(env.num_envs).float()


def reset_env_params(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    env.close_reward_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.target_reward_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def got_illegal_contacts(env: ManagerBasedRLEnv, threshold=0.01):
    forces = contact_forces(env)
    forces[:, 2] = 0  # set z components to 0 (gravity)
    return (forces.norm(dim=1) > threshold).view(env.num_envs).float()


def contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")):
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    return sensor.data.net_forces_w.view(env.num_envs, -1)


@configclass
class ObservationsCfg:
    @configclass
    class SimCfg(ObsGroup):
        """Observations for simulation and rewards"""
        forces = ObsTerm(func=contact_forces)
        l2_dist = ObsTerm(func=l2_distance)
        #vel = ObsTerm(func=lambda env: env.scene["cube"].data.root_lin_vel_w)

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group"""
        camera = ObsTerm(func=cam_data)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    sim: SimCfg = SimCfg()


@configclass
class EventCfg:
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0, 0)},
            "velocity_range": {"x": (0, 0), "y": (0, 0), "z": (0, 0)},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )
    reset_vars = EventTerm(func=reset_env_params, mode="reset")


# Custom terminations mdp
def joint_pos_out_of_manual_limit(
        env: ManagerBasedRLEnv,
        room_cfg: SceneEntityCfg = SceneEntityCfg("room"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
        bounds: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0)
) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the configured bounds.

    Note:
        This function is similar to :func:`joint_pos_out_of_limit` but allows the user to specify the bounds manually.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    room: AssetBase = env.scene[room_cfg.name]

    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)

    # compute any violations
    out_of_x_lower = asset.data.root_pos_w[:, 0] < env.scene.env_origins[:, 0] + bounds[0]
    out_of_x_higher = asset.data.root_pos_w[:, 0] > env.scene.env_origins[:, 0] + bounds[1]
    out_of_y_lower = asset.data.root_pos_w[:, 1] < env.scene.env_origins[:, 1] + bounds[2]
    out_of_y_higher = asset.data.root_pos_w[:, 1] > env.scene.env_origins[:, 1] + bounds[3]

    out_x = torch.logical_or(out_of_x_higher, out_of_x_lower)
    out_y = torch.logical_or(out_of_y_higher, out_of_y_lower)

    return torch.logical_or(out_x, out_y)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=-1)

    # (2) Failure (out of bounds , timeout)
    terminating = RewTerm(func=joint_pos_out_of_manual_limit, weight=-50.0)
    time_out = RewTerm(func=mdp.time_out, weight=-1.0)

    # (3) Rewards
    is_close = RewTerm(func=is_close_once, weight=40)  # Intermediary reward if close to the target
    target_reached = RewTerm(func=reached_target, weight=100)  # Reward when target is reached

    # (4) Negative rewards
    illegal_contacts = RewTerm(func=got_illegal_contacts, weight=-10)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    dog_out_of_bounds = DoneTerm(
        func=joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("cube"),
                "room_cfg": SceneEntityCfg("room"),
                "bounds": (-3.0, 3.0, -3.0, 3.0)},
    )


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""
    pass


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class CubeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=2, env_spacing=10)
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    # RL settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 20 Hz control (calls to the model for actions)
        self.episode_length_s = 20

        # simulation settings
        self.sim.dt = 0.01  # run Physics at 100Hz
        # update sensor update periods

        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.camera is not None:
            self.scene.camera.update_period = self.sim.dt

        self.close_reward_given = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.sim.device)
        self.target_reward_given = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.sim.device)
