import argparse
import os

from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.envs.mdp import JointVelocityActionCfg

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.sim import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

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
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6152), lin_vel=(0, 0, 0)),
    )

    ball = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/ball",
                          spawn=sim_utils.SphereCfg(radius=0.1,
                                                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                                    collision_props=sim_utils.CollisionPropertiesCfg(),
                                                    visual_material=sim_utils.PreviewSurfaceCfg(
                                                        diffuse_color=(1.0, 0.0, 0.0), metallic=0),
                                                    ),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=(1, 1, 0.8)))

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0),
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
        # no-processing of actions
        self._processed_actions = self._raw_actions

    def apply_actions(self):
        self._asset.write_root_velocity_to_sim(self._processed_actions)


@configclass
class CubeActionTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = CubeActionTerm
    """The class corresponding to the action term."""


@configclass
class ActionsCfg:
    vel = JointVelocityActionCfg(asset_name="robot")


##
# Custom observation terms
##
def cam_rgb(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["camera"].data.output["rgb"]


def cam_depth(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["camera"].data.output["distance_to_image_plane"]


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # cube velocity
        camera_rgb = ObsTerm(func=cam_rgb)
        camera_depth = ObsTerm(func=cam_depth)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0, 0)},
            "velocity_range": {"x": (0, 0), "y": (0, 0), "z": (0, 0)},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


# Custom terminations mdp
def joint_pos_out_of_manual_limit(
        env: ManagerBasedRLEnv, room_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the configured bounds.

    Note:
        This function is similar to :func:`joint_pos_out_of_limit` but allows the user to specify the bounds manually.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    room: AssetBaseCfg = env.scene[room_cfg.name]


    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > bounds[1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < bounds[0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    dog_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot"), "room_cfg": SceneEntityCfg("room")},
    )


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


@configclass
class CubeEnvCfg(ManagerBasedRLEnv):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=10)
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    # RL settings



    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics


def main():
    """Main function."""
    # setup base environment
    env_cfg = CubeEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene[SceneEntityCfg("cube")]
    # Setup target position commands with random x and y, and fixed z
    target_position = robot  # Randomize x and y coordinates
    fixed_z_column = torch.full((env.num_envs, 1), 0.6152, device=env.device)  # fixed z of 1m

    # offset all targets so that they move to the world origin
    #target_position -= env.scene.env_origins
    action = torch.zeros((env.num_envs, 1), device=env.device)

    # simulate physics
    count = 0
    obs, _ = env.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            action[:, 0] = torch.ones_like(action[:, 0], device=env.device)

            # # step env
            # if (count < 100):
            #     action[:, 0] = torch.zeros_like(action[:, 0], device=env.device)
            # elif (count < 150):
            #     action[:, 0] = torch.ones_like(action[:, 0], device=env.device)
            # elif (count < 200):
            #     action[:, 0] = torch.full_like(action[:, 0], -1, device=env.device)
            # else:
            #     action[:, 0] = torch.ones_like(action[:, 0], device=env.device)

            obs, _ = env.step(action)
            # print mean squared position error between target and current position
            #error = torch.norm(obs["policy"][0] - action).mean().item()
            #print(f"[Step: {count:04d}]: Mean position error: {error:.4f}")

            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
