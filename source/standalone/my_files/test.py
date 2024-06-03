"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="My implem for Go2")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import os
import torch

import omni.isaac.orbit.envs.mdp as mdp
from omni.isaac.orbit.envs import BaseEnv, BaseEnvCfg
from omni.isaac.orbit.managers import EventTermCfg as EventTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.scene import InteractiveSceneCfg
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg, ArticulationCfg
from omni.isaac.orbit.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.markers.config import FRAME_MARKER_CFG
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.utils.math import subtract_frame_transforms


@configclass
class MySceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # #create a 2 wheeled turtlebot
    # robot: ArticulationCfg = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Turtlebot/turtlebot.usd",
    #     ),
    # )

# environment configuration contains scene, observations, actions and events configs
@configclass
class MyEnvCfg(BaseEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = MySceneCfg(num_envs=10, env_spacing=2.5)

    # Modify the envs.BaseEnvCfg.sim which defines the simulation parameters
    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz

def main():
    """Main function."""
    # parse the arguments
    env_cfg = MyEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = BaseEnv(cfg=env_cfg)

    # simulate physics
    while simulation_app.is_running():
        pass

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
