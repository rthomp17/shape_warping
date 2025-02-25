from shape_warping import utils, viz_utils
import pickle
import trimesh
from shape_warping.shape_reconstruction import ObjectWarpingSE3Batch, warp_to_pcd_se3

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
    | {f"Target {part} Pointcloud": test_object_pcls[part] for part in object_parts}
).show()


# Getting point descriptors for better fitting heuristics

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
    part_warps[part_name] = ObjectWarpingSE3Batch(
        part_warp_models[part_name],
        test_object_pcls[part_name],
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
viz_utils.show_pcds_plotly(viz_pointclouds).show()

# See the meshes
viz_utils.show_meshes_plotly(viz_mesh_vertices, viz_mesh_faces).show()
