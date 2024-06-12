import utils
import trimesh
import time
import numpy as np
import copy as cp
import pickle
from warp_model_learning import learn_warps

# Example of generating the warp model using the spatulas

warp_training_object_ids = [
    "./example_data/spatula_1.obj",
    "./example_data/spatula_2.obj",
    "./example_data/spatula_3.obj",
    "./example_data/spatula_4.obj",
    "./example_data/spatula_5.obj",
    "./example_data/spatula_6.obj",
]
warp_training_meshes = [trimesh.load(f) for f in warp_training_object_ids]

n_dimensions = 4  # Number of dimensions of the latent space
timestr = time.strftime("%Y%m%d-%H%M%S")
model_name = f"spatula_{timestr}_dim_{n_dimensions}"
model_tag = "None"  # This is extraneous unless we're doing the part-based warping

model_warp_data = learn_warps(warp_training_meshes, n_dimensions=n_dimensions)

# Metadata keeps track of what objects were used to train the warp model
training_object_ids = (
    warp_training_object_ids[: model_warp_data["canonical_idx"]]
    + warp_training_object_ids[model_warp_data["canonical_idx"] + 1 :]
)

model_metadata = utils.CanonShapeMetadata(
    model_tag,
    warp_training_object_ids[model_warp_data["canonical_idx"]],
    training_object_ids,
    part_label=None,
)

model_warps = utils.CanonShape(
    model_warp_data["canonical_pcl"],
    model_warp_data["canonical_mesh_points"],
    model_warp_data["canonical_mesh_faces"],
    model_warp_data["canonical_center_transform"],
    contact_points=None,
    metadata=model_metadata,
    pca=model_warp_data["pca"],
)

pickle.dump(model_warps, open(model_name, "wb"))
