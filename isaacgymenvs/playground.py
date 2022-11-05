from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import math
from PIL import Image as im

camera_link = "wx250s/gripper_link"
box_seg_id = 1
image_width = 360
image_height = 240

# default_join_angles = {
#     "LF_HAA": 0.03,    # [rad]
#     "LH_HAA": 0.03,    # [rad]
#     "RF_HAA": -0.03,   # [rad]
#     "RH_HAA": -0.03,   # [rad]

#     "LF_HFE": 0.4,     # [rad]
#     "LH_HFE": -0.4,    # [rad]
#     "RF_HFE": 0.4,     # [rad]
#     "RH_HFE": -0.4,    # [rad]

#     "LF_KFE": -0.8,    # [rad]
#     "LH_KFE": 0.8,     # [rad]
#     "RF_KFE": -0.8,    # [rad]
#     "RH_KFE": 0.8,     # [rad]

#     "waist": 0.00,
#     "shoulder": 0.00,
#     "elbow": 0.00,
#     "forearm_roll": 0.00,
#     "wrist_angle": 0.00,
#     "wrist_rotate": 0.00
# }

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="Franka Attractor Example")

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
else:
    raise RuntimeError('Only supporting physx')

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
gym.add_ground(sim, plane_params)

# load asset
asset_root = "../assets"
asset_file = "urdf/pentapede/urdf/pentapede.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
pentapede_asset = gym.load_asset(
    sim, asset_root, asset_file, asset_options)

# Set up the env grid
num_envs = 1
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Some common handles for later use
envs = []
pentapede_handles = []
box_handles = []
camera_handles = []

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.62)

# box_asset = gym.create_box(sim, 1, 0.3, 2, asset_options)
# box_pose = gymapi.Transform()
# box_pose.p = gymapi.Vec3(3, -0.2, 1)
# box_color = gymapi.Vec3(1., 0., 0.)
box_asset = gym.create_box(sim, 0.05, 0.05, 0.05, asset_options)
box_pose = gymapi.Transform()
box_pose.p = gymapi.Vec3(3, -0.2, 2)
box_color = gymapi.Vec3(1., 0., 0.)

camera_props = gymapi.CameraProperties()
camera_props.width = image_width
camera_props.height = image_height

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add pentapede
    pentapede_handle = gym.create_actor(env, pentapede_asset, pose, "pentapede", i, 2)
    body_dict = gym.get_actor_rigid_body_dict(env, pentapede_handle)

    pentapede_handles.append(pentapede_handle)

    box_handle = gym.create_actor(env, box_asset, box_pose, 'box', i, 0, segmentationId=box_seg_id)
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL, box_color)
    box_handles.append(box_handle)

    camera_handle = gym.create_camera_sensor(env, camera_props)

    body_handle = gym.find_actor_rigid_body_handle(env, pentapede_handle, camera_link)
    gym.attach_camera_to_body(camera_handle, env, body_handle, gymapi.Transform(), gymapi.FOLLOW_TRANSFORM)
    camera_handles.append(camera_handle)

num_dof = gym.get_asset_dof_count(pentapede_asset)
dof_props = gym.get_asset_dof_properties(pentapede_asset)
for i in range(num_dof):
    dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    dof_props['stiffness'][i] = 85.0 #self.Kp
    dof_props['damping'][i] = 2.0 #self.Kd


for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], pentapede_handles[i], dof_props)

# Point camera at environments
cam_pos = gymapi.Vec3(2.308732, 2.887052, 1.502522)
cam_target = gymapi.Vec3(0.0, 0.0, 0.62)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

while not gym.query_viewer_has_closed(viewer):
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # render the camera sensors
    gym.render_all_camera_sensors(sim)

    for i in range(num_envs):
        rgb_filename = "graphics_images/rgb_env%d.png" % (i)
        gym.write_camera_image_to_file(sim, envs[i], camera_handles[i], gymapi.IMAGE_COLOR, rgb_filename)
        #rgb_image = gym.get_camera_image(sim, envs[i], camera_handles[i], gymapi.IMAGE_COLOR)
        #rgb_image = np.reshape(rgb_image, (image_height, image_width, 4))[:, :, 0:3]

        depth_image = gym.get_camera_image(sim, envs[i], camera_handles[i], gymapi.IMAGE_DEPTH)
        print(depth_image.min(), depth_image.max())
        #depth_image[depth_image == -np.inf] = 0
        #normalized_depth = -255.0*(depth_image/np.min(depth_image + 1e-4))
        #normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
        #normalized_depth_image.save("graphics_images/depth_env%d.jpg" % (i))

        seg_image = gym.get_camera_image(sim, envs[i], camera_handles[i], gymapi.IMAGE_SEGMENTATION)
        #seg_image[seg_image != box_seg_id] = 0
        #seg_image[seg_image == box_seg_id] = 255
        #im.fromarray(seg_image.astype(np.uint8), mode="L").save("graphics_images/seg_env%d.jpg" % (i))

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)