import os
from typing import Dict, Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


#WARNING WARNING WARNING
#Below only works if sphere has no degrees of freedom
#things will break if it does
class Pentapede(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.camera_link = self.cfg["env"]["camera_link"]
        
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["camera_pos"] = self.cfg["env"]["learn"]["cameraPosRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang
        self.base_init_state = state

        # sphere init state
        sphere_pos = self.cfg["env"]["sphereInitState"]["pos"]
        sphere_rot = self.cfg["env"]["sphereInitState"]["rot"]
        sphere_v_lin = self.cfg["env"]["sphereInitState"]["vLinear"]
        sphere_v_ang = self.cfg["env"]["sphereInitState"]["vAngular"]
        sphere_state = sphere_pos + sphere_rot + sphere_v_lin + sphere_v_ang
        self.sphere_init_state = sphere_state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 66
        self.cfg["env"]["numActions"] = 18

        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[self.pentapede_indices] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.initial_root_states[self.sphere_indices] = to_torch(self.sphere_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.pentapede_indices = to_torch(self.pentapede_indices, dtype=torch.long, device=self.device)
        self.sphere_indices = to_torch(self.sphere_indices, dtype=torch.long, device=self.device)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/pentapede/urdf/pentapede.urdf" #TODO or do we want pentapede_minimal?

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        asset_options.use_mesh_materials = True


        pentapede_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(pentapede_asset)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(pentapede_asset)
        self.dof_names = self.gym.get_asset_dof_names(pentapede_asset)
        knee_names = [s for s in body_names if "THIGH" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)

        dof_props = self.gym.get_asset_dof_properties(pentapede_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.Kp
            dof_props['damping'][i] = self.Kd

        sphere_radius = self.cfg["env"]["sphereInitState"]["radius"]
        sphere_color = self.cfg["env"]["sphereInitState"]["color"]
        sphere_asset_options = gymapi.AssetOptions()
        sphere_asset_options.slices_per_cylinder = 40
        sphere_asset_options.fix_base_link = True
        sphere_asset = self.gym.create_sphere(self.sim, sphere_radius, sphere_asset_options)
        sphere_start_pose = gymapi.Transform()
        sphere_start_pose.p = gymapi.Vec3(*self.sphere_init_state[:3])

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.pentapede_handles = []
        self.sphere_handles = []
        self.pentapede_indices = []
        self.sphere_indices = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env_ptr)
            
            pentapede_handle = self.gym.create_actor(env_ptr, pentapede_asset, start_pose, "pentapede", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, pentapede_handle, dof_props)
            self.pentapede_handles.append(pentapede_handle)
            object_idx = self.gym.get_actor_index(env_ptr, pentapede_handle, gymapi.DOMAIN_SIM)
            self.pentapede_indices.append(object_idx)

            sphere_handle = self.gym.create_actor(env_ptr, sphere_asset, sphere_start_pose, "sphere", i, 1, 0)
            self.sphere_handles.append(sphere_handle)
            box_idx = self.gym.get_actor_index(env_ptr, sphere_handle, gymapi.DOMAIN_SIM)
            self.sphere_indices.append(box_idx)
            self.gym.set_rigid_body_color(env_ptr, sphere_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*sphere_color))

        for i in range(len(knee_names)):
            #TODO handle or index?
            self.knee_indices[i] = self.gym.find_asset_rigid_body_index(pentapede_asset, knee_names[i])

        #TODO handle or index?
        self.camera_link_index = self.gym.find_asset_rigid_body_index(pentapede_asset, self.camera_link)
        #TODO handle or index?
        self.base_index = self.gym.find_asset_rigid_body_index(pentapede_asset, "base")

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        #TODO looks like set position and not delta, confirm
        targets = self.action_scale * self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_pentapede_reward(
            self.root_states,
            self.camera_link_states,
            self.torques,
            self.contact_forces,
            self.sphere_indices,
            self.knee_indices,
            self.progress_buf,
            self.rew_scales,
            self.base_index,
            self.max_episode_length,
        )
    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.camera_link_states = self.rigid_body_states[:, self.camera_link_index][:, 0:13]

        self.obs_buf[:] = compute_pentapede_observations(  # tensors
                                                        self.root_states,
                                                        self.pentapede_indices,
                                                        self.camera_link_states,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.actions,
                                                        # scales
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale
        )

    def reset_idx(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = 0
        
        actor_ids = torch.cat((self.pentapede_indices[env_ids], self.sphere_indices[env_ids])).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        

        pentapede_indices = self.pentapede_indices[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(pentapede_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_pentapede_reward(
    # tensors
    root_states,
    camera_link_states,
    torques,
    contact_forces,
    sphere_indices,
    knee_indices,
    episode_lengths,
    # Dict
    rew_scales,
    # other
    base_index,
    max_episode_length
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int) -> Tuple[Tensor, Tensor]

    sphere_pos = root_states[sphere_indices, 0:3]
    camera_pos = camera_link_states[:, 0:3]
    
    # camera position reward
    camera_error = torch.sum(torch.square(sphere_pos - camera_pos), dim=1)
    rew_cam_pos = torch.exp(-camera_error/0.25) * rew_scales["camera_pos"]

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    total_reward = rew_cam_pos# + rew_torque
    total_reward = torch.clip(total_reward, 0., None)

    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out

    #TODO do we detach reward? Why does this work with example code?
    return total_reward.detach(), reset


@torch.jit.script
def compute_pentapede_observations(root_states,
                                pentapede_indices,
                                camera_link_states,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale
                                ):
    
    base_quat = root_states[pentapede_indices, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[pentapede_indices, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[pentapede_indices, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    
    cam_pos = camera_link_states[:, 0:3]
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale
    dof_vel_scaled = dof_vel * dof_vel_scale

    obs = torch.cat((cam_pos,
                     base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     dof_pos_scaled,
                     dof_vel_scaled,
                     actions
                     ), dim=-1)

    return obs
