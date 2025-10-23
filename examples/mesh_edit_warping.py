import numpy as np
import trimesh
import pickle
import copy as cp
import os.path as osp
from shape_warping import utils, viz_utils
from shape_warping.skill_transfer import *
from shape_warping.shape_reconstruction import (
    ObjectWarpingSE3Batch,
    warp_to_pcd_se3,
    ObjectWarpingSE2Batch,
    warp_to_pcd_se2,
)

"""


EXTRACTING INFO FROM DEMONSTRATION
Load demonstration - consists of point clouds for (parent,child) objects + initial and final poses of child object 
Child object is in the end effector. Parent object is on the table. 
These are copied from RelNDF.
"""

demo_path = osp.join("example_data", "mug_on_rack_demos", "place_demo_0.npz")
demo = np.load(demo_path, mmap_mode="r", allow_pickle=True)

demo_start_pcds = {
    "parent": demo["multi_obj_start_pcd"].item()["parent"],
    "child": demo["multi_obj_start_pcd"].item()["child"],
}

demo_start_poses = {
    "parent": demo["multi_obj_start_obj_pose"].item()["parent"],
    "child": demo["multi_obj_start_obj_pose"].item()["child"],
}

demo_final_pcds = {
    "parent": demo["multi_obj_final_pcd"].item()["parent"],
    "child": demo["multi_obj_final_pcd"].item()["child"],
}

demo_final_poses = {
    "parent": demo["multi_obj_final_obj_pose"].item()["parent"],
    "child": demo["multi_obj_final_obj_pose"].item()["child"],
}

# Subsamples points for faster inference/to prevent CUDA from running out of memory
for object_type in ("parent", "child"):
    demo_start_pcds[object_type], _ = utils.farthest_point_sample(
        demo_start_pcds[object_type], 2000
    )
    demo_final_pcds[object_type], _ = utils.farthest_point_sample(
        demo_final_pcds[object_type], 2000
    )

# We actually care about the relative transform between objects
demo_transforms = {
    "parent": np.matmul(
        utils.pos_quat_to_transform(
            demo_final_poses["parent"][:3], demo_final_poses["parent"][3:]
        ),
        np.linalg.inv(
            utils.pos_quat_to_transform(
                demo_start_poses["parent"][:3], demo_start_poses["parent"][3:]
            )
        ),
    ),
    "child": np.matmul(
        utils.pos_quat_to_transform(
            demo_final_poses["child"][:3], demo_final_poses["child"][3:]
        ),
        np.linalg.inv(
            utils.pos_quat_to_transform(
                demo_start_poses["child"][:3], demo_start_poses["child"][3:]
            )
        ),
    ),
}

# Just to make visualization easier
good_demo_camera_view = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0.5, y=0.5, z=0.5),
    eye=dict(x=-1, y=-1, z=0.5),
)

# Visualize the skill demonstration
demo_fig = viz_utils.show_pcds_plotly(
    {
        "Parent Object Initial Pose": demo_start_pcds["parent"],
        "Child Object Initial Pose": demo_start_pcds["child"],
        "Parent Object End Pose": demo_final_pcds["parent"],
        "Child Object End Pose": demo_final_pcds["child"],
    },
    camera=good_demo_camera_view,
)

# demo_fig.show()

# Load shape warping models. These have been pretrained from ~5 examples.
warp_models = {
    "parent": utils.CanonShape.from_pickle(
        "example_data/rack_models/example_pretrained_whole_rack"
    ),
    "child": utils.CanonShape.from_pickle(
        "example_data/mug_models/example_pretrained_whole_mug"
    ),
}

"""

Reconstruct the demo object geometries from the demo pointclouds using shape warping
"""

reconstructed_demo_pcds = {}
reconstruction_params = {}
for object_type in (
    "parent",
    "child",
):
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

    reconstructed_demo_pcds[object_type], _, reconstruction_params[object_type] = (
        warp_to_pcd_se2(
            warp_reconstruction,
            n_angles=5,
            n_batches=1,
            inference_kwargs=inference_kwargs,
        )
    )

# Transforming the reconstruction to the 'placed' position
# Just to visualize and verify that it matches the demonstrated placement
reconstructed_demo_pcds_final = {
    "parent": utils.transform_pcd(
        reconstructed_demo_pcds["parent"], demo_transforms["parent"]
    ),
    "child": utils.transform_pcd(
        reconstructed_demo_pcds["child"], demo_transforms["child"]
    ),
}

# This updates the estimated child object pose to the 'placed' position
# Which is important for obtaining the interaction points from the demo
reconstruction_params["child"] = update_reconstruction_params_with_transform(
    reconstruction_params["child"], demo_transforms["child"]
)

# Visualize demo reconstruction
# The original pointclouds and reconstructions should match up pretty well
demo_reconstructed_fig = viz_utils.show_pcds_plotly(
    {
        "Parent Object Target": demo_start_pcds["parent"],
        "Child Object Target": demo_start_pcds["child"],
        "Parent Object Reconstruction": reconstructed_demo_pcds["parent"],
        "Child Object Reconstruction": reconstructed_demo_pcds["child"],
        "Parent Object Final": demo_final_pcds["parent"],
        "Child Object Final": demo_final_pcds["child"],
        "Parent Reconstruction Final": reconstructed_demo_pcds_final["parent"],
        "Child Reconstruction Final": reconstructed_demo_pcds_final["child"],
    },
    camera=good_demo_camera_view,
)

##demo_reconstructed_fig.show()

"""


Identify demo interaction points - keypoints representing the relative transform between the demonstration objects. 
Details in Biza et al "Interaction Warping".
"""

# Find and save interaction points
nearby_points_delta = 0.055  # Empirically picked

(
    knns,
    deltas,
    target_indices,
) = get_interaction_points(
    demo_start_pcds["child"],
    demo_start_pcds["parent"],
    warp_models["child"],
    reconstruction_params["child"],
    warp_models["parent"],
    reconstruction_params["parent"],
    nearby_points_delta,
    mesh_vertices_only=True,
)

interaction_points = {"knns": knns, "deltas": deltas, "target_indices": target_indices}

warped_interaction_points = warp_interaction_points(
    interaction_points,
    warp_models["child"],
    warp_models["parent"],
    reconstruction_params["child"],
    reconstruction_params["parent"],
)

# Visualize demo with the interaction points
demo_with_keypoints_fig = viz_utils.show_pcds_plotly(
    {
        "Parent Object Final": demo_final_pcds["parent"],
        "Child Object Final": demo_final_pcds["child"],
        "Parent Reconstruction Final": reconstructed_demo_pcds_final["parent"],
        "Child Reconstruction Final": reconstructed_demo_pcds_final["child"],
        "Parent Interaction Points": warped_interaction_points["parent"],
        "Child Interaction Points": warped_interaction_points["child"],
    }
)
demo_with_keypoints_fig.show()


"""
'Test' Object is the transformed rack mesh.


"""
rack_mesh = warp_models["parent"].to_mesh(reconstruction_params["parent"])
mug_mesh = warp_models["child"].to_mesh(reconstruction_params["child"])

# Just being careful because the trimesh transform functions do in-place mutation
transformed_mug_mesh = cp.deepcopy(mug_mesh)
transformed_rack_mesh = cp.deepcopy(rack_mesh)

mug_scaling_matrix = np.eye(4)
mug_scaling_matrix[2, 2] *= 0.85
transformed_mug_mesh.apply_transform(mug_scaling_matrix)

rack_scaling_matrix = np.eye(4)
rack_scaling_matrix[2, 2] *= 1.15
transformed_rack_mesh.apply_transform(rack_scaling_matrix)


# Visualize demo reconstruction
# The original pointclouds and reconstructions should match up pretty well
mesh_edit_fig = viz_utils.show_pcds_plotly(
    {
        "Parent Object Original": demo_start_pcds["parent"],
        "Child Object Original": demo_start_pcds["child"],
        "Parent Object Edited": transformed_rack_mesh.vertices,
        "Child Object Edited": transformed_mug_mesh.vertices,
    },
    camera=good_demo_camera_view,
)

# mesh_edit_fig.show()
edit_transformed_interaction_points = mesh_edit_interaction_points(
    interaction_points,
    transformed_mug_mesh,
    transformed_rack_mesh,
    np.eye(4),
    np.eye(4),
)

# Visualize demo with the interaction points
mesh_edit_with_keypoints_fig = viz_utils.show_pcds_plotly(
    {
        "Parent Object Original": demo_start_pcds["parent"],
        "Child Object Original": demo_start_pcds["child"],
        "Parent Object Edited": transformed_rack_mesh.vertices,
        "Child Object Edited": transformed_mug_mesh.vertices,
        "Parent Interaction Points": edit_transformed_interaction_points["parent"],
        "Child Interaction Points": edit_transformed_interaction_points["child"],
    }
)
# mesh_edit_with_keypoints_fig.show()

test_transform = mesh_edit_infer_relpose(
    interaction_points,
    transformed_mug_mesh,
    transformed_rack_mesh,
    np.eye(4),
    np.eye(4),
)

# Visualize the transferred skill
final_transfer = viz_utils.show_pcds_plotly(
    {
        "Test Parent Object": transformed_rack_mesh.vertices,
        "Test Child Object": transformed_mug_mesh.vertices,
        "Test Child Place": utils.transform_pcd(
            transformed_mug_mesh.vertices, test_transform
        ),
    }
)
final_transfer.show()
