import argparse
from omni.isaac.orbit.app import AppLauncher

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

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.sim import SimulationContext, UsdFileCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.envs import BaseEnvCfg, BaseEnv, mdp

##
# Pre-defined configs
##
from omni.isaac.orbit_assets import CARTPOLE_CFG, UNITREE_GO2_CFG  # isort:skip
from omni.isaac.orbit.sensors import RayCasterCfg, ContactSensorCfg, CameraCfg
from omni.isaac.orbit.sensors.ray_caster import patterns
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.envs import RLTaskEnvCfg
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit_assets.unitree import UNITREE_GO2_CFG, GO1_ACTUATOR_CFG
from omni.isaac.orbit.managers import EventTermCfg as EventTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise
import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    room = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room",
                        spawn=UsdFileCfg(usd_path="omniverse://localhost/Library/assets/simple_room/simple_room.usd"),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0.8)))

    # robots
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                                     init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.5)))

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0,
                                     texture_file="omniverse://localhost/NVIDIA/Assets/Skies/Cloudy/abandoned_parking_4k.hdr"),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        # if count % 500 == 0:
        #     # reset counter
        #     count = 0
        #     # reset the scene entities
        #     # root state
        #     # we offset the root state by the origin since the states are written in simulation world frame
        #     # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
        #     root_state = robot.data.default_root_state.clone()
        #     root_state[:, :3] += scene.env_origins
        #     robot.write_root_state_to_sim(root_state)
        #     # set joint positions with some noise
        #     joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        #     joint_pos += torch.rand_like(joint_pos) * 0.1
        #     robot.write_joint_state_to_sim(joint_pos, joint_vel)
        #     # clear internal buffers
        #     scene.reset()
        #     print("[INFO]: Resetting robot state...")

        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 10], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=20.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
