from shape_warping import utils, viz_utils
import pickle
import trimesh
from shape_warping.shape_reconstruction import ObjectWarpingSE3Batch, warp_to_pcd_se3


""" Example of reconstructing an object using a warp model """

warp_model = pickle.load(open("example_pretrained", "rb"))
test_object = trimesh.load("./example_data/spatula_meshes/spatula_7.obj")

# Scaling to match the environment
utils.trimesh_transform(test_object, scale=0.075)

# # You can try changing the orientation of the test object by uncommenting these lines
# rotation = utils.random_quat()
# utils.trimesh_transform(test_object, rotation=rotation)

# Get the pointcloud from the mesh so we can warp to it
test_pcl = utils.trimesh_create_verts_surface(test_object, 1000)

viz_utils.show_pcds_plotly(
    {
        "Source (Canonical) Pointcloud": warp_model.canonical_pcl,
        "Target Pointcloud": test_pcl,
    }
)

# This tells the warp model to do gradient descent on position, orientation, and shape
# To get the best possible reconstruction
inference_kwargs = {
    "train_latents": True,
    "train_poses": True,
    "train_centers": True,
    "train_scales": True,
}

warp_reconstruction = ObjectWarpingSE3Batch(
    warp_model,
    test_pcl,
    device="cuda",
    lr=1e-2,
    n_steps=100,
)

print("Fitting the test object shape and pose")
pcl_reconstr, _, reconstr_params = warp_to_pcd_se3(
    warp_reconstruction, n_angles=5, n_batches=1, inference_kwargs=inference_kwargs
)

# See the transform matrix
print(utils.pos_quat_to_transform(reconstr_params.position, reconstr_params.quat))

# See the latent shape params
print(reconstr_params.latent)

# See the pointclouds
viz_utils.show_pcds_plotly(
    {
        "Source (Canonical) Pointcloud": warp_model.canonical_pcl,
        "Target Pointcloud": test_pcl,
        "Reconstruction": pcl_reconstr,
    }
).show()

# See the meshes
viz_utils.show_meshes_plotly(
    {
        # "Source (Canonical) Mesh": warp_model.mesh_vertices,
        # "Target Mesh": test_object.vertices,
        "Reconstruction": warp_model.to_transformed_mesh(reconstr_params).vertices,
    },
    {
        # "Source (Canonical) Mesh": warp_model.mesh_vertices,
        # "Target Mesh": test_object.vertices,
        "Reconstruction": warp_model.to_transformed_mesh(reconstr_params).faces,
    },
).show()
