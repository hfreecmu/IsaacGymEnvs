import os
from typing import Dict, Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

from PIL import Image as im

#WARNING WARNING WARNING
#Below only works if sphere and wall have no degrees of freedom
#things will break if it does

#WARNING WARNING WARNING
#Becareful of observation space of first frame
#as not everything has rendered and values are 0

#TODO arm isn't going all the way up because of clipActions set to 1
#need to retune clipActions, clipObservations, actionScale, and dofPositionScale
#possibly separate between arm and legs

#Notes:
#quat_rotate is object frame to world frame
#quat_rotate_inverse is world frame to object frame

NO_GPU = False

#Change cmaera enabled here and in computer obserations
CAMERA_ENABLED = False
USE_RANDOM_SPHERE = True
MOVE_SPHERE_STRAIGHT = False
MOVE_SPHERE_RANDOM = False
NUM_LINES = 4
FREEZE_PENTAPEDE = False
DEBUG_IM = False

assert not (MOVE_SPHERE_RANDOM and MOVE_SPHERE_STRAIGHT)
MOVE_SPHERE = (MOVE_SPHERE_RANDOM or MOVE_SPHERE_STRAIGHT)

class Pentapede(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.debug_count = 0

        self.cfg = cfg

        self.camera_link = self.cfg["env"]["camera_link"]
        self.position_link = self.cfg["env"]["position_link"]
        self.sphere_seg_id = self.cfg["env"]["sphere_seg_id"]
        
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale= self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.sphere_cam_dir_scale = self.cfg["env"]["learn"]["sphereCamDirScale"]
        self.sphere_actor_dir_scale = self.cfg["env"]["learn"]["sphereActorDirScale"]
        self.action_scale_leg = self.cfg["env"]["control"]["actionScaleLeg"]
        #self.action_scale_arm = self.cfg["env"]["control"]["actionScaleArm"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["camera_pos"] = self.cfg["env"]["learn"]["cameraPosRewardScale"]
        self.rew_scales["camera_quat"] = self.cfg["env"]["learn"]["cameraQuatRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["linear_vel"] = self.cfg["env"]["learn"]["linearVelRewardScale"]

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
        sphere_rot = self.cfg["env"]["sphereInitState"]["rot"]
        sphere_v_lin = self.cfg["env"]["sphereInitState"]["vLinear"]
        sphere_v_ang = self.cfg["env"]["sphereInitState"]["vAngular"]

        if USE_RANDOM_SPHERE:
            self.sphere_pos_lower = np.array(self.cfg["env"]["sphereInitState"]["pos_lower"])
            self.sphere_pos_upper = np.array(self.cfg["env"]["sphereInitState"]["pos_upper"])
            self.max_sphere_dist = self.cfg["env"]["sphereInitState"]["max_sphere_dist"]
            sphere_state = [0.0, 0.0, 0.0] + sphere_rot + sphere_v_lin + sphere_v_ang
        else:
            self.sphere_pos_fixed = self.cfg["env"]["sphereInitState"]["pos_fixed"]
            sphere_state = self.cfg["env"]["sphereInitState"]["pos_fixed"] + sphere_rot + sphere_v_lin + sphere_v_ang

        self.sphere_init_state = sphere_state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # arm scales
        self.named_action_arm_scales = self.cfg["env"]["control"]["actionScaleArms"]

        width = self.cfg["env"]["camera"]["image_width"]
        height = self.cfg["env"]["camera"]["image_height"]
        if CAMERA_ENABLED and not NO_GPU:
            self.cfg["env"]["numObservations"] = 68 + width*height
        else:
            self.cfg["env"]["numObservations"] = 70
        self.cfg["env"]["numActions"] = 18

        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        # for key in self.rew_scales.keys():
        #     self.rew_scales[key] *= self.dt

        if MOVE_SPHERE:
            self.sphere_speed = self.cfg["env"]["sphereInitState"]["speed"]# * self.dt

        if MOVE_SPHERE_RANDOM:
            self.sphere_target_pos = torch.zeros((self.num_envs, 3)).float().to(self.device)
            self.sphere_source_pos = torch.zeros((self.num_envs, 3)).float().to(self.device)

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

        self.seg_image_states = None

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        #WARNING WARNING WARNING if other actor has dof, would need to view to -1 (like above) and find correct indices/handles
        #for each one
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        self.action_arm_scales = torch.zeros(len(self.cfg["env"]["arm_joints"]), dtype=torch.float, device=self.device, requires_grad=False)
        index = 0
        debug_arm_names = []
        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            if not (name in self.cfg["env"]["arm_joints"]):
                continue
            scale = self.named_action_arm_scales[name]
            self.action_arm_scales[index] = scale
            debug_arm_names.append(name)
            assert debug_arm_names[index] == self.cfg["env"]["arm_joints"][index]
            index += 1

        assert index == len(self.cfg["env"]["control"]["actionScaleArms"])
            
        # initialize some data used later on
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[self.pentapede_indices] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.initial_root_states[self.sphere_indices] = to_torch(self.sphere_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.pentapede_indices = to_torch(self.pentapede_indices, dtype=torch.long, device=self.device)
        self.sphere_indices = to_torch(self.sphere_indices, dtype=torch.long, device=self.device)

  
        self.camera_base_dir = torch.zeros(self.num_envs, 3).to(self.device)

        #WARNING WARNING WARNING
        #only works if defining initial pose is in this direction
        #if initial pose is randomized not sure how this may affect this
        self.camera_base_dir[:, 0] = 1

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
        asset_file = "urdf/pentapede/urdf/pentapede.urdf"

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

        arm_names = self.cfg["env"]["arm_joints"]
        self.arm_indices = torch.zeros(len(arm_names), dtype=torch.long, device=self.device, requires_grad=False)

        dof_props = self.gym.get_asset_dof_properties(pentapede_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.Kp
            dof_props['damping'][i] = self.Kd

        sphere_radius = self.cfg["env"]["sphereInitState"]["radius"]
        sphere_color = self.cfg["env"]["sphereInitState"]["color"]
        sphere_asset_options = gymapi.AssetOptions()
        sphere_asset_options.slices_per_cylinder = 40
        sphere_asset_options.fix_base_link = False
        sphere_asset_options.disable_gravity = True
        sphere_asset = self.gym.create_sphere(self.sim, sphere_radius, sphere_asset_options)
        sphere_start_pose = gymapi.Transform()
        if not USE_RANDOM_SPHERE:
            sphere_start_pose.p = gymapi.Vec3(*self.sphere_pos_fixed)

        #TODO tune fov
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg["env"]["camera"]["image_width"]
        camera_props.height = self.cfg["env"]["camera"]["image_height"]

        horizontal_fov = self.cfg["env"]["camera"]["horizontal_fov"]
        camera_props.horizontal_fov = horizontal_fov

        dot_product_thresh = np.cos(np.deg2rad(horizontal_fov) / 2)
        self.dot_product_thresh = torch.as_tensor(dot_product_thresh, dtype=torch.float32, device=self.device)
        
        #TODO might be  able to access tensor buffer directly even if cpu
        #maybe not because get_camera_image does not return buffer, but could check
        if NO_GPU:
            camera_props.enable_tensors = False
        else:
            camera_props.enable_tensors = True

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.pentapede_handles = []
        self.sphere_handles = []
        self.pentapede_indices = []
        self.sphere_indices = []

        if CAMERA_ENABLED:
            self.camera_handles = []
            self.seg_images = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env_ptr)
            
            pentapede_handle = self.gym.create_actor(env_ptr, pentapede_asset, start_pose, "pentapede", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, pentapede_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, pentapede_handle)
            self.pentapede_handles.append(pentapede_handle)
            object_idx = self.gym.get_actor_index(env_ptr, pentapede_handle, gymapi.DOMAIN_SIM)
            self.pentapede_indices.append(object_idx)

            sphere_handle = self.gym.create_actor(env_ptr, sphere_asset, sphere_start_pose, "sphere", i, 1, self.sphere_seg_id)
            self.sphere_handles.append(sphere_handle)
            sphere_idx = self.gym.get_actor_index(env_ptr, sphere_handle, gymapi.DOMAIN_SIM)
            self.sphere_indices.append(sphere_idx)
            self.gym.set_rigid_body_color(env_ptr, sphere_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*sphere_color))

            if CAMERA_ENABLED:
                camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
                body_handle = self.gym.find_actor_rigid_body_handle(env_ptr, pentapede_handle, self.camera_link)
                self.gym.attach_camera_to_body(camera_handle, env_ptr, body_handle, gymapi.Transform(), gymapi.FOLLOW_TRANSFORM)
                self.camera_handles.append(camera_handle)

                if NO_GPU:
                    ### WARNING WARNING WARNING won't be able to read camera in non gpu mode using this
                    seg_image = torch.from_numpy(np.array(self.gym.get_camera_image(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION), dtype=np.float32))
                else:
                    seg_image = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION))
                self.seg_images.append(seg_image)

        ###WARNING WARNING WARNING 
        #We use find_asset_{TYPE}_index, but anymal uses find_actor_{TYPE}_body_handle
        #I think indices makes more sense and I don't trust Nvidia to do anything right
        #but wanted to make a note of it

        for i in range(len(knee_names)):
            #TODO handle or index?
            self.knee_indices[i] = self.gym.find_asset_rigid_body_index(pentapede_asset, knee_names[i])

        for i in range(len(arm_names)):
            self.arm_indices[i] = self.gym.find_asset_dof_index(pentapede_asset, arm_names[i])

        self.camera_link_index = self.gym.find_asset_rigid_body_index(pentapede_asset, self.camera_link)
        #self.position_link_index = self.gym.find_asset_rigid_body_index(pentapede_asset, self.position_link)
        
        self.base_index = self.gym.find_asset_rigid_body_index(pentapede_asset, "base")

        if CAMERA_ENABLED:
            self.seg_images_stack = torch.stack(self.seg_images)

    def pre_physics_step(self, actions):

        self.actions = actions.clone().to(self.device)
        
        #WARNING WARNING WARNING looks like set joint value directly, not delta
        #TODO make it so don't need both operations on arm indices
        targets = self.action_scale_leg * self.actions + self.default_dof_pos
        #targets[:, self.arm_indices] = self.action_scale_arm * self.actions[:, self.arm_indices] + self.default_dof_pos[:, self.arm_indices]

        #WARNING WARNING WARNING
        #TODO
        #action_arm_scales order could be different. Is not, but could be, so be careful
        targets[:, self.arm_indices] = self.action_arm_scales * self.actions[:, self.arm_indices] + self.default_dof_pos[:, self.arm_indices]

        if FREEZE_PENTAPEDE:
            targets = self.default_dof_pos

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        #leaving this in as this is where it was before
        #but there was in issue with ball when other envs would reset
        #moving it after fixed it, not sure why
        #almost as if set set_actor_root_state_tensor_indexed needs to be called
        #right before sim and not anytime else
        # env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(env_ids) > 0:
        #     self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if MOVE_SPHERE_RANDOM:
            #TODO I know technically wrong if random target position is close to current position
            #but such a small and rare case I do not care that much
            
            sphere_vel = self.sphere_target_pos - self.sphere_source_pos
            des_sphere_vel = self.sphere_target_pos - self.root_states[self.sphere_indices, 0:3]
            dot_prod = torch.sum(sphere_vel*des_sphere_vel, dim=-1)
            sphere_speed_ids = torch.where(dot_prod <= 0.0, 1.0, 0.0).nonzero(as_tuple=False).squeeze(-1)
            is_not_in_env_id = [False if i.item() in env_ids else True for i in sphere_speed_ids]
            sphere_speed_ids = sphere_speed_ids[is_not_in_env_id]
 
            sphere_speed_indices = self.sphere_indices[sphere_speed_ids].to(dtype=torch.int32) 
            self.sphere_target_pos[sphere_speed_ids] = torch.as_tensor(np.random.rand(len(sphere_speed_ids), 3) * (self.sphere_pos_upper-self.sphere_pos_lower) + \
                                                        self.sphere_pos_lower, dtype=torch.float, device=self.device)
            self.sphere_source_pos[sphere_speed_ids] = self.root_states[sphere_speed_indices.long(), 0:3]
            sphere_dist = self.sphere_target_pos[sphere_speed_ids] - self.sphere_source_pos[sphere_speed_ids]
            sphere_vel = torch.nn.functional.normalize(sphere_dist)
            self.sphere_target_pos[sphere_speed_ids] = torch.where((torch.norm(sphere_dist, dim=1) <= self.max_sphere_dist).reshape(-1, 1), self.sphere_target_pos[sphere_speed_ids], self.sphere_source_pos[sphere_speed_ids] + self.max_sphere_dist*sphere_vel)
            
            self.initial_root_states[sphere_speed_indices.long(), 7:10] = sphere_vel
            self.initial_root_states[sphere_speed_indices.long(), 0:3] = self.sphere_source_pos[sphere_speed_ids]
        else:
            sphere_speed_ids = []
            sphere_speed_indices = []

        if (len(env_ids) > 0) or (len(sphere_speed_ids) > 0):
            self.reset_idx(env_ids, sphere_speed_indices, sphere_speed_ids)

    def compute_reward(self, actions):
        self.rew_buf1[:], self.rew_buf2[:], self.reset_buf[:] = compute_pentapede_reward(
            self.root_states,
            self.camera_link_states,
            self.position_link_states,
            self.torques,
            self.contact_forces,
            self.pentapede_indices,
            self.sphere_indices,
            self.knee_indices,
            self.progress_buf,
            self.rew_scales,
            self.base_index,
            self.camera_base_dir,
            self.dot_product_thresh,
            self.max_episode_length,
            self.debug_count,
        )
        # self.debug_count += 1
        # print (self.debug_count)
    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.camera_link_states = self.rigid_body_states[:, self.camera_link_index][:, 0:13]
        #self.position_link_states = self.rigid_body_states[:, self.position_link_index][:, 0:13]
        self.position_link_states = self.root_states[self.pentapede_indices, :]

        if CAMERA_ENABLED:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            self.seg_image_states = torch.stack(self.seg_images).float()
            self.gym.end_access_image_tensors(self.sim)
        else:
            self.seg_image_states = torch.as_tensor(0)

        if CAMERA_ENABLED and DEBUG_IM:
            debug_im = self.seg_image_states[0].detach().cpu().numpy()
            im.fromarray(255*debug_im.astype(np.uint8), mode="L").save("graphics_images/seg_env%d.jpg" % (0))

        if NUM_LINES > 0:
            self.gym.clear_lines(self.viewer)

        for i in range(NUM_LINES):
            line_colors = [0.0, 0.0, 1.0]
            camera_pos_debug = self.camera_link_states[i:i+1, 0:3].detach().cpu().numpy()

            camera_quat = self.camera_link_states[i:i+1, 3:7]
            camera_dir_world = quat_rotate(camera_quat, self.camera_base_dir[i:i+1]).detach().cpu().numpy()
        
            camera_end_point = (camera_pos_debug + 5*camera_dir_world)
            line_vertices = np.concatenate((camera_pos_debug, camera_end_point), axis=1).flatten().tolist()
        
            self.gym.add_lines(self.viewer, self.envs[i], 1, line_vertices, line_colors)

            line_colors_2 = [0.0, 1.0, 0.0]
            sphere_pos_worlds = self.root_states[self.sphere_indices, 0:3].detach().cpu().numpy()
            sphere_pos_world = sphere_pos_worlds[i:i+1]
            line_vertices_2 = np.concatenate((camera_pos_debug, sphere_pos_world), axis=1).flatten().tolist()
            self.gym.add_lines(self.viewer, self.envs[i], 1, line_vertices_2, line_colors_2)

            if MOVE_SPHERE_RANDOM:
                line_colors_3 = [1.0, 0.0, 0.0]
                sphere_pos_origin = self.sphere_source_pos[i:i+1, 0:3].detach().cpu().numpy()
                sphere_pos_target = self.sphere_target_pos[i:i+1, 0:3].detach().cpu().numpy()
                line_vertices_3 = np.concatenate((sphere_pos_origin, sphere_pos_target), axis=1).flatten().tolist()
                self.gym.add_lines(self.viewer, self.envs[i], 1, line_vertices_3, line_colors_3)
            grid_line_colors = [1.0, 0.65, 0.0]
            #TODO can make this called once
            spacing = self.cfg["env"]['envSpacing']
            corner_0 = np.array([[-spacing, -spacing, 0]])
            corner_1 = np.array([[-spacing, spacing, 0]])
            corner_2 = np.array([[spacing, spacing, 0]])
            corner_3 = np.array([[spacing, -spacing, 0]])
            line_corner_0 = np.concatenate((corner_0, corner_1), axis=1).flatten().tolist()
            line_corner_1 = np.concatenate((corner_1, corner_2), axis=1).flatten().tolist()
            line_corner_2 = np.concatenate((corner_2, corner_3), axis=1).flatten().tolist()
            line_corner_3 = np.concatenate((corner_3, corner_0), axis=1).flatten().tolist()
            self.gym.add_lines(self.viewer, self.envs[i], 1, line_corner_0, grid_line_colors)
            self.gym.add_lines(self.viewer, self.envs[i], 1, line_corner_1, grid_line_colors)
            self.gym.add_lines(self.viewer, self.envs[i], 1, line_corner_2, grid_line_colors)
            self.gym.add_lines(self.viewer, self.envs[i], 1, line_corner_3, grid_line_colors)

            pos_line_colors = [1.0, 0.65, 0.0]
            pos_start = self.position_link_states[i:i+1, 0:3].detach().cpu().numpy()
            pos_end = pos_start + [0, 0, 1]
            pos_line_vertices = np.concatenate((pos_start, pos_end), axis=1).flatten().tolist()
            self.gym.add_lines(self.viewer, self.envs[i], 1, pos_line_vertices, pos_line_colors)
        
        self.obs_buf[:] = compute_pentapede_observations(  # tensors
                                                        self.root_states,
                                                        self.pentapede_indices,
                                                        self.camera_link_states,
                                                        self.position_link_states,
                                                        self.seg_image_states,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.actions,
                                                        self.camera_base_dir,
                                                        # scales
                                                        self.sphere_cam_dir_scale,
                                                        self.sphere_actor_dir_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale,
                                                        self.sphere_indices,
                                                        self.dot_product_thresh,
                                                        CAMERA_ENABLED,
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.gravity_vec,
                                                        self.debug_count
        )

    #TODO call set_actor_root_state_tensor_indexed ONLY ONCE with combined indices
    #could be reason for bug of moving this somewhere else and can possibly put it back
    #it is not turns out but leaving this comment here
    #7db807a55a81b7979f46bbf86430088cd57e56de
    def reset_idx(self, env_ids, sphere_speed_indices=[], sphere_speed_ids=[]):
        if (len(env_ids) <= 0) and (len(sphere_speed_indices) <= 0):
            return

        if len(env_ids) > 0 and len(sphere_speed_indices) > 0:
            pass
        
        for i in env_ids:
            if i.item() in sphere_speed_ids:
                raise RuntimeError('here')

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = 0
        
        pentapede_indices = self.pentapede_indices[env_ids].to(dtype=torch.int32)
        #leaving below in as that is what it used to be but documentaiton says call once
        #even though it worked
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.initial_root_states),
        #                                              gymtorch.unwrap_tensor(pentapede_indices), len(env_ids))

        #TODO combine these two set actor calls into one if it affects speed         
        sphere_indices = self.sphere_indices[env_ids].to(dtype=torch.int32)                                   
        if USE_RANDOM_SPHERE: 
            sphere_pos = torch.as_tensor(np.random.rand(len(env_ids), 3) * (self.sphere_pos_upper-self.sphere_pos_lower) + self.sphere_pos_lower, 
                                         dtype=torch.float, device=self.device)
            self.initial_root_states[sphere_indices.long(), 0:3] = sphere_pos


        if MOVE_SPHERE_STRAIGHT:
            random_vel = np.random.rand(len(env_ids), 3)*2 - 1
            random_vel[:, 2] = 0
            random_vel = (self.sphere_speed * random_vel / np.expand_dims(np.linalg.norm(random_vel, axis=1), axis=1)).tolist()
            random_vel = to_torch(random_vel, device=self.device, requires_grad=False)

            self.initial_root_states[sphere_indices.long(), 7:10] = random_vel
        elif MOVE_SPHERE_RANDOM:
            self.sphere_target_pos[env_ids] = torch.as_tensor(np.random.rand(len(env_ids), 3) * (self.sphere_pos_upper-self.sphere_pos_lower) + \
                                              self.sphere_pos_lower, dtype=torch.float, device=self.device)
            
            self.sphere_source_pos[env_ids] = self.initial_root_states[sphere_indices.long(), 0:3]
                
            sphere_dist = self.sphere_target_pos[env_ids] - self.sphere_source_pos[env_ids]
            sphere_vel = torch.nn.functional.normalize(sphere_dist)
            self.sphere_target_pos[env_ids] = torch.where((torch.norm(sphere_dist, dim=1) <= self.max_sphere_dist).reshape(-1, 1), self.sphere_target_pos[env_ids], self.sphere_source_pos[env_ids] + self.max_sphere_dist*sphere_vel)
            self.initial_root_states[sphere_indices.long(), 7:10] = sphere_vel

        #leaving below in as that is what it used to be but documentaiton says call once
        #even though it worked
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.initial_root_states),
        #                                              gymtorch.unwrap_tensor(sphere_indices), len(env_ids))

        full_indices = torch.cat((pentapede_indices, sphere_indices), dim=0)
        num_actors = 2*len(env_ids)
        
        if len(sphere_speed_indices) > 0:
            full_indices = torch.cat((full_indices, sphere_speed_indices.to(dtype=torch.int32) ), dim=0)
            num_actors = full_indices.shape[0]
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(full_indices), num_actors)

        if len(env_ids) > 0:
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
    position_link_states,
    torques,
    contact_forces,
    pentapede_indices,
    sphere_indices,
    knee_indices,
    episode_lengths,
    # Dict
    rew_scales,
    # other
    base_index,
    camera_base_dir,
    dot_product_thresh,
    max_episode_length,
    debug_count
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, Tensor, Tensor, int, int) -> Tuple[Tensor, Tensor, Tensor]

    #new error consists of camera quat, root state velocity, root position, staying alive, torque penalty
    #instead of position, could do change in position?
    #could also add upright reward like in ant.py?

    #start with camera error and reward
    sphere_pos_world = root_states[sphere_indices, 0:3]
    camera_pos_world = camera_link_states[:, 0:3]
    camera_quat = camera_link_states[:, 3:7]
    
    sphere_dir_world = normalize(sphere_pos_world - camera_pos_world)
    sphere_dir_camera = quat_rotate_inverse(camera_quat, sphere_dir_world)

    camera_dir_world = quat_rotate(camera_quat, camera_base_dir)
    in_front_cam = is_in_camera_view(sphere_dir_world, camera_dir_world, dot_product_thresh)

    #TODO square or abs? For all below not just this one
    #camera_quat_error = torch.sum(torch.square(sphere_dir_cam - camera_base_dir), dim=1)
    camera_quat_error = torch.sum(torch.square(sphere_dir_camera - camera_base_dir), dim=1)
    #rew_cam_quat = torch.exp(-camera_quat_error/0.5) * rew_scales["camera_quat"]
    #rew_cam_quat = -camera_quat_error * rew_scales["camera_quat"]
    #rew_cam_quat = torch.exp(-camera_quat_error/0.25) * rew_scales["camera_quat"]
    rew_cam_quat = (1.0 / (1.0 + camera_quat_error)) * rew_scales['camera_quat'] * in_front_cam[:, 0]

    #now root state velocity
    #linear_velocity_world = root_states[pentapede_indices, 7:9]
    actor_pos_world = position_link_states[:, 0:3]
    desired_linear_velocity = sphere_pos_world[:, 0:2] - actor_pos_world[:, 0:2]
    #linear_velocity_error = torch.sum(torch.abs(desired_linear_velocity - linear_velocity_world), dim=1)
    linear_velocity_error = torch.sum(torch.square(desired_linear_velocity), dim=1)
    #rew_lin_vel = -linear_velocity_error * rew_scales["linear_vel"]
    #rew_lin_vel = torch.exp(-linear_velocity_error) * rew_scales["linear_vel"]
    rew_lin_vel = (1.0 / (1.0 + linear_velocity_error)) * rew_scales['linear_vel'] * in_front_cam[:, 0]
    
    #now actor position
    #camera_pos_error = torch.sum(torch.square(sphere_pos_world[:, 0:2] - pos_world[:, 0:2]), dim=1) / torch.sum(torch.square(sphere_pos_world[:, 0:2]), dim=1)
    #position_error = torch.sum(torch.abs(sphere_pos_world[:, 0:2] - actor_pos_world[:, 0:2]), dim=1)
    #rew_cam_pos = 0.01 + (-camera_pos_error) * rew_scales["camera_pos"]
    #rew_position = -position_error * rew_scales["camera_pos"]

    #now stay alive
    # reward for duration of staying alive
    #alive_reward = torch.ones_like(rew_cam_quat) * 0.5

    #torque penalty
    #torque_penalty = torch.sum(torch.square(torques), dim=1) 
    #rew_torque = torque_penalty* rew_scales["torque"]

    #print(camera_quat_error[0].item(), rew_cam_quat[0].item(), linear_velocity_error[0].item(), rew_lin_vel[0].item())

    #total_reward = rew_cam_quat + rew_lin_vel #+ alive_reward + rew_torque #+rew_position
    #total_reward = rew_lin_vel

    # if debug_count % 12 == 0:
    #     breakpoint()

    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)

    #adjust reward for fallen agents
    #TODO change -2.0 to configurable param
    #total_reward = torch.where(reset, torch.ones_like(total_reward) * -2.0, total_reward)

    #total_reward = torch.clip(total_reward, 0., None)

    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out

    rew_cam_pos = rew_lin_vel
    return rew_cam_pos.detach(), rew_cam_quat.detach(), reset   # (leg reward, arm reward, reset)
    #return total_reward.detach(), torch.zeros_like(total_reward.detach()), reset   # (leg reward, arm reward, reset)
    # return total_reward.detach(), reset


@torch.jit.script
def compute_pentapede_observations(root_states,
                                pentapede_indices,
                                camera_link_states,
                                position_link_states,
                                seg_image_states,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                actions,
                                camera_base_dir,
                                sphere_cam_dir_scale,
                                sphere_actor_dir_scale,
                                dof_pos_scale,
                                dof_vel_scale,
                                sphere_indices,
                                dot_product_thresh,
                                camera_enabled,
                                lin_vel_scale,
                                ang_vel_scale,
                                gravity_vec,
                                debug_count
                                ):

    #get infront of cam
    sphere_pos_world = root_states[sphere_indices, 0:3]
    camera_pos_world = camera_link_states[:, 0:3]
    camera_quat = camera_link_states[:, 3:7]

    camera_dir_world = quat_rotate(camera_quat, camera_base_dir)
    sphere_dir_world = normalize(sphere_pos_world - camera_pos_world)

    #TODO ones and zeros or ones and minus ones?
    in_front_cam = is_in_camera_view(sphere_dir_world, camera_dir_world, dot_product_thresh)

    #get direction of sphere in camera frame
    #only know it if it is in front of cam
    sphere_dir_cam = quat_rotate_inverse(camera_quat, sphere_dir_world) * in_front_cam * sphere_cam_dir_scale

    #get normalized sphere direction in actor frame
    #only know it if it is in front of cam
    base_quat = root_states[pentapede_indices, 3:7]
    actor_pos_world = position_link_states[:, 0:3]
    actor_sphere_direction_world = sphere_pos_world - actor_pos_world
    actor_sphere_direction_world[:, 2] = 0
    actor_sphere_direction_actor = quat_rotate_inverse(base_quat, actor_sphere_direction_world)
    actor_sphere_direction_actor = actor_sphere_direction_actor * in_front_cam * sphere_actor_dir_scale

    #linear and angular vel
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[pentapede_indices, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[pentapede_indices, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)

    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    dof_vel_scaled = dof_vel * dof_vel_scale

    # if debug_count % 12 == 0:
    #     breakpoint()

    if camera_enabled:
        raise RuntimeError('camera enabled not supported right now')
        seg_image_states_flat = seg_image_states.view(pentapede_indices.shape[0], -1)
        obs = torch.cat((base_lin_vel,
                         base_ang_vel,
                         projected_gravity,
                         dof_pos_scaled,
                         dof_vel_scaled,
                         actions,
                         seg_image_states_flat,
                         in_front_cam*sphere_dir_cam,
                         in_front_cam*pos_dist,
                         in_front_cam,
                         ), dim=-1)
    else:
        obs = torch.cat((base_lin_vel,              # 3
                         base_ang_vel,              # 3
                         projected_gravity,         # 3
                         dof_pos_scaled,                # 18
                         dof_vel_scaled,                # 18
                         actions,                       # 18
                         sphere_dir_cam,   # 3
                         actor_sphere_direction_actor, # 3
                         in_front_cam,                  # 1
                         ), dim=-1)

    return obs
