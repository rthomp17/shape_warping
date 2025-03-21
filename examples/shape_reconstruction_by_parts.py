from shape_warping import utils, viz_utils
import pickle
import trimesh
from shape_warping.shape_reconstruction import ObjectWarpingSE3Batch, warp_to_pcd_se3, mask_and_cost_batch_pt 

""" Example of reconstructing an object using a set of warp models for its component parts """

object_name = "teapot"
object_parts = ["body", "handle", "lid", "spout"]

part_warp_models = {
    "body": pickle.load(
        open("example_data/teapot_models/example_pretrained_body", "rb")
    ),
    "handle": pickle.load(
        open("example_data/teapot_models/example_pretrained_handle", "rb")
    ),
    "lid": pickle.load(open("example_data/teapot_models/example_pretrained_lid", "rb")),
    "spout": pickle.load(
        open("example_data/teapot_models/example_pretrained_spout", "rb")
    ),
}

test_object_parts = {
    "body": trimesh.load("example_data/teapot_meshes/kettle_5/kettle_5_body.obj"),
    "handle": trimesh.load("example_data/teapot_meshes/kettle_5/kettle_5_handle.obj"),
    "lid": trimesh.load("example_data/teapot_meshes/kettle_5/kettle_5_lid.obj"),
    "spout": trimesh.load("example_data/teapot_meshes/kettle_5/kettle_5_spout.obj"),
}

# Target Object
viz_utils.show_meshes_plotly(
    {name: test_object_parts[name].vertices for name in test_object_parts.keys()},
    {name: test_object_parts[name].faces for name in test_object_parts.keys()},
).show()

# # You can try changing the orientation of the test object by uncommenting these lines
# rotation = utils.random_quat()
# utils.trimesh_transform(test_object, rotation=rotation)

# Get the pointcloud from the mesh so we can warp to it
test_object_pcls = {
    part_name: utils.trimesh_create_verts_surface(test_object_parts[part_name], 1000)
    for part_name in test_object_parts.keys()
}


# Scaling to match
(
    test_object_pcls["body"],
    test_object_pcls["spout"],
    test_object_pcls["handle"],
    test_object_pcls["lid"],
) = utils.scale_points_circle(
    [
        test_object_pcls["body"],
        test_object_pcls["spout"],
        test_object_pcls["handle"],
        test_object_pcls["lid"],
    ],
    base_scale=0.2,
)


viz_utils.show_pcds_plotly(
    {
        f"Source (Canonical) {part} Pointcloud": utils.transform_pcd(
            part_warp_models[part].canonical_pcl,
            utils.pos_quat_to_transform(
                part_warp_models[part].center_transform, (0, 0, 0, 1)
            ),
        )
        for part in object_parts
    }
    | {f"Target {part} Pointcloud": test_object_pcls[part] for part in object_parts},
   title= "Canon and Test Object Pointclouds"
).show()


# Used in constructing the relational heuristic for part alignment
canon_adjacent_parts = [
                         {"body": part_warp_models["body"],
                         "handle": part_warp_models["handle"]},
                         {"body": part_warp_models["body"],
                         "spout": part_warp_models["spout"]},
                         {"body": part_warp_models["body"],
                         "lid": part_warp_models["lid"]},
                         {"lid": part_warp_models["lid"],
                         "handle": part_warp_models["handle"]},
                         {"lid": part_warp_models["lid"],
                         "spout": part_warp_models["spout"]},
                        ]


test_object_adjacent_parts = [
                         {"body": test_object_pcls["body"],
                         "handle": test_object_pcls["handle"]},
                         {"body": test_object_pcls["body"],
                         "spout": test_object_pcls["spout"]},
                         {"body": test_object_pcls["body"],
                         "lid": test_object_pcls["lid"]},
                         {"lid": test_object_pcls["lid"],
                         "handle": test_object_pcls["handle"]},
                         {"lid": test_object_pcls["lid"],
                         "spout": test_object_pcls["spout"]},
                        ]

# Getting descriptors for better fitting heuristic
canon_part_labels = utils.get_canon_labels(
                canon_adjacent_parts, 
                part_warp_models, 
                object_parts, 
            )

viz_utils.visualize_teapot_relational_descriptors(canon_adjacent_parts,
                                                  {part: part_warp_models[part].canonical_pcl for part in object_parts},
                                                  canon_part_labels, title="Canon Relational Descriptors")

test_object_part_labels = utils.get_part_labels(
                test_object_adjacent_parts, 
            )

viz_utils.visualize_teapot_relational_descriptors(test_object_adjacent_parts,
                                                  test_object_pcls,
                                                  test_object_part_labels, title="Test Object Relational Descriptors")
# This tells the warp model to do gradient descent on position, orientation, and shape
# To get the best possible reconstruction
inference_kwargs = {
    "train_latents": True,
    "train_poses": True,
    "train_centers": True,
    "train_scales": True,
}

part_warps = {}
part_reconstructed = {}
part_params = {}
viz_pointclouds = {}
viz_mesh_vertices = {}
viz_mesh_faces = {}

for part_name in object_parts:
    # Constructing the cost function with the relational heuristics
    cost_function = (
            lambda source, test_object, canon_part_labels, latent, scale, initial: mask_and_cost_batch_pt(
                test_object,
                test_object_part_labels[part_name],
                source,
                canon_part_labels,
            )
        )
    
    part_warps[part_name] = ObjectWarpingSE3Batch(
        part_warp_models[part_name],
        test_object_pcls[part_name],
        canon_labels=canon_part_labels[part_name],
        cost_function = cost_function,
        device="cuda",
        lr=1e-2,
        n_steps=100,
    )

    print(f"Fitting the test {part_name} shape and pose")
    part_reconstructed[part_name], _, part_params[part_name] = warp_to_pcd_se3(
        part_warps[part_name],
        n_angles=5,
        n_batches=12,
        inference_kwargs=inference_kwargs,
    )

    viz_pointclouds = viz_pointclouds | {
        f"Source (Canonical) {part_name} Pointcloud": part_warp_models[
            part_name
        ].canonical_pcl,
        f"Target {part_name} Pointcloud": test_object_pcls[part_name],
        f"{part_name} Reconstruction": part_reconstructed[part_name],
    }

    viz_mesh_vertices = viz_mesh_vertices | {
        # f"Source (Canonical) {part_name} Mesh": part_warp_models[part_name].mesh_vertices,
        # f"Target {part_name} Mesh": test_object_parts[part_name].vertices,
        f"{part_name} Reconstruction": part_warp_models[part_name]
        .to_transformed_mesh(part_params[part_name])
        .vertices
    }

    viz_mesh_faces = viz_mesh_faces | {
        # f"Source (Canonical) {part_name} Mesh": part_warp_models[part_name].mesh_faces,
        # f"Target {part_name} Mesh": test_object_parts[part_name].faces,
        f"{part_name} Reconstruction": part_warp_models[part_name]
        .to_transformed_mesh(part_params[part_name])
        .faces,
    }

# See the pointclouds
viz_utils.show_pcds_plotly(viz_pointclouds, title="Reconstructed Pointclouds").show()

# See the meshes
viz_utils.show_meshes_plotly(viz_mesh_vertices, viz_mesh_faces).show()
