import genesis as gs
import torch
import math
import os

import cable as cable

class DronePayloadEnv:
    def __init__(self, num_envs: int, env_cfg: dict, obs_cfg: dict, reward_cfg: dict, target_cfg: dict, show_viever: bool=False, device: str="cuda"):
        self.device = torch.device(device)
        self.num_envs = num_envs

        # self.dim_obs = obs_cfg["dim_obs"] # dimension of observer vector (drone/payload/world observed state)
        # self.dim_actions = env_cfg["dim_actions"] # dimension of action vector (output dimension of policy - fed to FC)
        # self.dim_targets = target_cfg["dim_targets"] # dimension for payload target (pos, velocity, attitude - what error is based on)

        self.dt = 0.01 # 100 Hz target
        # self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt) # in number of timesteps

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.target_cfg = target_cfg

        # scales for obs/rewards todo

        # Genesis-World Scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt),
            viewer_options=gs.options.ViewerOptions(
                # max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(0.0, 7.0, 3.0),
                camera_lookat=(0.0, 0.0, 2.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1)),
                                              background_color=(0.9, 0.9, 0.9)),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=False,
            ),
            show_viewer=show_viever,
        )

        # World ground plane surface
        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="urdf/plane/checker_blue.png"
                ),
            ),
        )

        # Drone
        self.drone = self.scene.add_entity(
            gs.morphs.Drone(
                file="urdf/drones/racer.urdf",
                model="RACE",  # "CF2X", "CF2P", or "RACE"
                pos=(0.0, 0.0, 0.5),  # meters, Z-up
                euler=(0.0, 0.0, 0.0),  # scipy extrinsic x-y-z, degrees
                propellers_link_name=("prop0_link", "prop1_link", "prop2_link", "prop3_link"),
                propellers_spin=(-1, 1, -1, 1),  # per propeller: -1 = CW, +1 = CCW
            ),
        )

        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Payload
        self.payload = self.scene.add_entity(
            gs.morphs.URDF(
                file=script_dir + "/payload/payload.urdf",
                pos=(1,0,0.5),
                euler=(0,0,0),
                scale=(1,1,1),
                links_to_keep=["attach_0", "attach_1", "attach_2"],
            ),
        )

        self.scene.build(n_envs=num_envs)

        # Drone base_link + Payload attachment links
        self.drone_link = self.drone.get_link("base_link")
        self.payload_attach_links = [self.payload.get_link(f"attach_{i}") for i in range(3)]
        self.payload_attach_links_idx = [link.idx_local for link in self.payload_attach_links]

        drone_attach_radius = 0.1  # m, within the base_link's 0.06 m collision radius
        drone_attach_z_offset = -0.02  # m, below the body center
        num_cables = len(self.payload_attach_links)
        drone_attach_offsets = [
            [
                drone_attach_radius * math.cos(math.radians(120) * i),
                drone_attach_radius * math.sin(math.radians(120) * i),
                drone_attach_z_offset,
            ]
            for i in range(num_cables)
        ]
        self.drone_attach = torch.tensor(
            drone_attach_offsets, dtype=gs.tc_float, device=self.device
        ).unsqueeze(0)

        cable_cfg = {
            "cables": [
                {
                    "stiffness": 2000.0,
                    "damping": 100.0,
                    "rest_length": 3,
                    "slack_transition_width": 0.01,
                    "cable_diameter": 0.01,
                }
                for _ in self.payload_attach_links
            ],
        }
        self.cable = cable.TetherModel(cable_cfg, num_envs=num_envs, device=self.device, dtype=gs.tc_float)
        self.num_cables = self.cable.num_cables
        self.drone_link_idx_repeated = [self.drone_link.idx_local] * self.num_cables

        drone_mass = self.drone.get_mass()
        hover_thrust_per_rotor = drone_mass * 9.81 / self.drone.n_propellers
        self.hover_rpm = torch.sqrt(hover_thrust_per_rotor / self.drone.KF) * 1.05

        hover_actions = torch.zeros(num_envs, self.drone.n_propellers, dtype=gs.tc_float, device=self.device)
        for i in range(2000):
            self.step(hover_actions)

    def step(self, actions):
        actions = torch.clamp(actions, -1.0, 1.0)
        propellers_rpm = (1.0 + actions * 0.8) * self.hover_rpm.unsqueeze(-1)
        self.drone.set_propellers_rpm(propellers_rpm)
        
        drone_link_pos = self.drone_link.get_pos(relative=False)
        drone_link_quat = self.drone_link.get_quat(relative=False)
        drone_link_vel = self.drone_link.get_vel()
        drone_link_ang = self.drone_link.get_ang()

        drone_pos, drone_vel = cable.attach_point_kinematics(drone_link_pos, drone_link_quat, drone_link_vel, drone_link_ang, self.drone_attach)

        payload_pos = self.payload.get_links_pos(self.payload_attach_links_idx, relative=False)
        payload_vel = self.payload.get_links_vel(self.payload_attach_links_idx)

        wrench = self.cable.compute(drone_pos, drone_vel, payload_pos, payload_vel)
        self.cable_wrench = wrench  # stashed for reward/obs (e.g. _reward_swing, tension penalty)

        self.drone.apply_links_external_wrench(wrench.drone_force, links_idx_local=self.drone_link_idx_repeated, pos=drone_pos)
        self.payload.apply_links_external_wrench(wrench.payload_force, links_idx_local=self.payload_attach_links_idx)

        self.scene.step()

    def reset_idx(self, idx):
        pass

    def reset(self):
        pass

    def _get_payload_state(self):
        pass

    def _get_drone_state(self):
        pass

    # Reward functions (TODO FOR RL)
    def _reward_target_trajectory(self):
        # Reward payload being close to trajectory target (sampled)
        pass

    def _reward_target_velocity(self):
        # Reward payload maintaining constant velocity
        pass

    def _reward_target_attitude(self):
        # Reward payload horizontal attitude
        pass

    def _reward_swing(self):
        # Reward (negatively) for payload swing
        pass

    def _reward_crash(self):
        # Reward (negatively) for drone/payload collision with environment
        pass