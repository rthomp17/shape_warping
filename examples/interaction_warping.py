import numpy as np
import trimesh
import pickle
import os.path as osp
from shape_warping import utils, viz_utils
from shape_warping.skill_transfer import *
from shape_warping.shape_reconstruction import ObjectWarpingSE3Batch, warp_to_pcd_se3, ObjectWarpingSE2Batch, warp_to_pcd_se2

'''


EXTRACTING INFO FROM DEMONSTRATION
Load demonstration - consists of point clouds for (parent,child) objects + initial and final poses of child object 
Child object is in the end effector. Parent object is on the table. 
These are copied from RelNDF.
'''

demo_path = osp.join('example_data', 'mug_on_rack_demos', 'place_demo_0.npz')
demo = np.load(demo_path, mmap_mode="r", allow_pickle=True)

demo_start_pcds = {
   'parent': demo["multi_obj_start_pcd"].item()['parent'],
   'child': demo["multi_obj_start_pcd"].item()['child']
}

demo_start_poses = {
   'parent': demo["multi_obj_start_obj_pose"].item()['parent'],
   'child': demo["multi_obj_start_obj_pose"].item()['child']
}

demo_final_pcds = {
   'parent': demo["multi_obj_final_pcd"].item()['parent'],
   'child': demo["multi_obj_final_pcd"].item()['child']
}

demo_final_poses = {
   'parent': demo["multi_obj_final_obj_pose"].item()['parent'],
   'child': demo["multi_obj_final_obj_pose"].item()['child']
}

# Subsamples points for faster inference/to prevent CUDA from running out of memory
for object_type in ('parent', 'child'):
    demo_start_pcds[object_type], _ = utils.farthest_point_sample(
                        demo_start_pcds[object_type], 2000
                    )
    demo_final_pcds[object_type], _ = utils.farthest_point_sample(
                        demo_final_pcds[object_type], 2000
                    )

# We actually care about the relative transform between objects
demo_transforms = {
   'parent': np.matmul(
                utils.pos_quat_to_transform(demo_final_poses['parent'][:3], demo_final_poses['parent'][3:]),
                np.linalg.inv(utils.pos_quat_to_transform(demo_start_poses['parent'][:3], demo_start_poses['parent'][3:]))),
   'child': np.matmul(
                 utils.pos_quat_to_transform(demo_final_poses['child'][:3], demo_final_poses['child'][3:]),
                np.linalg.inv(utils.pos_quat_to_transform(demo_start_poses['child'][:3], demo_start_poses['child'][3:]))),
}

# Just to make visualization easier
good_demo_camera_view = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=.5, y=.5, z=.5),
    eye=dict(x=-1, y=-1, z=.5)
)

# Visualize the skill demonstration
demo_fig = viz_utils.show_pcds_plotly({
                             "Parent Object Initial Pose": demo_start_pcds['parent'],
                             "Child Object Initial Pose": demo_start_pcds['child'],
                             "Parent Object End Pose": demo_final_pcds['parent'],
                             "Child Object End Pose": demo_final_pcds['child']}, 
                             camera=good_demo_camera_view)

demo_fig.show()

# Load shape warping models. These have been pretrained from ~5 examples.
warp_models = {
    'parent': utils.CanonShape.from_pickle("example_data/rack_models/example_pretrained_whole_rack"),
    'child': utils.CanonShape.from_pickle("example_data/mug_models/example_pretrained_whole_mug"),
}

'''

Reconstruct the demo object geometries from the demo pointclouds using shape warping
'''

reconstructed_demo_pcds = {}
reconstruction_params = {}
for object_type in ("parent", "child",):
    warp_model = warp_models[object_type]

    # Inferring over all possible parameters - adjust this for ablations or debugging
    inference_kwargs = {
        "train_latents": True,
        "train_poses": True,
        "train_centers": True,
        "train_scales": True,
    }

    warp_reconstruction = ObjectWarpingSE2Batch(
        warp_model,
        demo_start_pcds[object_type],
        device="cuda",
        lr=1e-2,
        n_steps=100,
    )

    print(f"Fitting the demo {object_type} object shape and pose")

    reconstructed_demo_pcds[object_type], _, reconstruction_params[object_type] = warp_to_pcd_se2(
        warp_reconstruction, n_angles=5, n_batches=1, inference_kwargs=inference_kwargs
    )

# Transforming the reconstruction to the 'placed' position
# Just to visualize and verify that it matches the demonstrated placement
reconstructed_demo_pcds_final = {
   'parent': utils.transform_pcd(reconstructed_demo_pcds['parent'], demo_transforms['parent']),
   'child': utils.transform_pcd(reconstructed_demo_pcds['child'], demo_transforms['child'])
   }

# This updates the estimated child object pose to the 'placed' position
# Which is important for obtaining the interaction points from the demo
reconstruction_params['child'] = update_reconstruction_params_with_transform(reconstruction_params['child'], demo_transforms['child'])

# Visualize demo reconstruction
# The original pointclouds and reconstructions should match up pretty well
demo_reconstructed_fig = viz_utils.show_pcds_plotly({
                             "Parent Object Target": demo_start_pcds['parent'],
                             "Child Object Target": demo_start_pcds['child'],
                             "Parent Object Reconstruction": reconstructed_demo_pcds['parent'],
                             "Child Object Reconstruction": reconstructed_demo_pcds['child'],

                             "Parent Object Final": demo_final_pcds['parent'],
                             "Child Object Final": demo_final_pcds['child'],
                             "Parent Reconstruction Final": reconstructed_demo_pcds_final['parent'],
                             "Child Reconstruction Final": reconstructed_demo_pcds_final['child'],
                             },
                             camera=good_demo_camera_view)

demo_reconstructed_fig.show()

'''



Identify demo interaction points - keypoints representing the relative transform between the demonstration objects. 
Details in Biza et al "Interaction Warping".
'''

# Find and save interaction points 
nearby_points_delta = 0.055 # Empirically picked

(   
    knns,
    deltas,
    target_indices,
) = get_interaction_points(
    demo_start_pcds['child'],
    demo_start_pcds['parent'],
    warp_models['child'],
    reconstruction_params['child'],
    warp_models['parent'],
    reconstruction_params['parent'],
    nearby_points_delta,
)

interaction_points = {'knns': knns, 'deltas': deltas, "target_indices": target_indices}

warped_interaction_points = warp_interaction_points(interaction_points, 
                                                    warp_models['child'], 
                                                    warp_models['parent'], 
                                                    reconstruction_params['child'], 
                                                    reconstruction_params['parent'])

# Visualize demo with the interaction points
demo_with_keypoints_fig = viz_utils.show_pcds_plotly({
                             "Parent Object Final": demo_final_pcds['parent'],
                             "Child Object Final": demo_final_pcds['child'],
                             "Parent Reconstruction Final": reconstructed_demo_pcds_final['parent'],
                             "Child Reconstruction Final": reconstructed_demo_pcds_final['child'],
                             "Parent Interaction Points": warped_interaction_points['parent'],
                             "Child Interaction Points": warped_interaction_points['child'],
                             })
demo_with_keypoints_fig.show()

'''



AT TEST TIME

# Load test objects (a few example objects copied from Shapenet)
'''
test_idx = 0
all_test_mugs = load_all_shapenet_files('mug')
all_test_racks= load_all_shapenet_files('syn_rack_easy')
test_parent_mesh = get_rack_mesh(all_test_racks[test_idx])
test_child_mesh = get_mug_mesh(all_test_mugs[test_idx])

utils.trimesh_transform(test_child_mesh, rotation=utils.euler_to_matrix([np.pi/2, 0, 0]))

# # You can try changing the orientation of the test object by uncommenting these lines
# rotation = utils.random_quat()
# utils.trimesh_transform(test_child_mesh, rotation=rotation)


# Get the test pointclouds from the meshes
test_pcds = {
    "parent": utils.trimesh_create_verts_surface(test_parent_mesh, 1000),
    "child": utils.trimesh_create_verts_surface(test_child_mesh, 1000)
}


# Downscale the pointclouds (hack because they come from different sources)

test_pcds = {
    "parent": utils.scale_points_circle([test_pcds['parent']], base_scale=.2)[0],
    "child": utils.scale_points_circle([test_pcds['child']], base_scale=.2)[0],
}

'''
Reconstruct the test objects using the warp model
'''
reconstructed_test_pcds = {}
reconstructed_test_params = {}
for object_type in ("parent", "child"):
    warp_model = warp_models[object_type]

    inference_kwargs = {
        "train_latents": True,
        "train_poses": True,
        "train_centers": True,
        "train_scales": True,
    }

    warp_reconstruction = ObjectWarpingSE2Batch(
        warp_model,
        test_pcds[object_type],
        device="cpu",
        lr=1e-2,
        n_steps=100,
    )

    print(f"Fitting the demo {object_type} object shape and pose")
    reconstructed_test_pcds[object_type], _, reconstructed_test_params[object_type] = warp_to_pcd_se2(
        warp_reconstruction, n_angles=5, n_batches=1, inference_kwargs=inference_kwargs
    )


# Visualize target reconstruction
test_reconstructed_fig = viz_utils.show_pcds_plotly({
                             "Test Parent Object": test_pcds['parent'],
                             "Test Child Object": test_pcds['child'],
                             "Test Parent Object Reconstruction": reconstructed_test_pcds['parent'],
                             "Test Child Object Reconstruction": reconstructed_test_pcds['child'],
                             })
test_reconstructed_fig.show()

'''

Skill transfer using the reconstruction from shape warping
'''

# Transfer interaction points
test_interaction_points = warp_interaction_points(interaction_points, 
                                                    warp_models['child'], 
                                                    warp_models['parent'], 
                                                    reconstructed_test_params['child'], 
                                                    reconstructed_test_params['parent'])

# Visualize transferred interaction points
test_transferred_points_fig = viz_utils.show_pcds_plotly({"Test Parent Object": test_pcds['parent'],
                             "Test Child Object": test_pcds['child'],
                             "Test Parent Object Reconstruction": reconstructed_test_pcds['parent'],
                             "Test Child Object Reconstruction": reconstructed_test_pcds['child'],
                             "Test Parent Interaction Points": test_interaction_points['parent'],
                             "Test Child Interaction Points": test_interaction_points['child'],

                             })
test_transferred_points_fig.show()


# Transfer the skill (based on interaction point alignment)
test_transform = infer_relpose(
        interaction_points,
        warp_models['child'], 
        warp_models['parent'],
        reconstructed_test_params['child'], 
        reconstructed_test_params['parent'],
    )


# Visualize the transferred skill
final_transfer = viz_utils.show_pcds_plotly({
    "Test Parent Object": test_pcds['parent'],
    "Test Child Object": test_pcds['child'],
    "Test Child Place": utils.transform_pcd(test_pcds['child'], test_transform)
    })
final_transfer.show()