from shape_warping import utils
import trimesh
import time
import numpy as np
import copy as cp
import pickle
from shape_warping.warp_model_learning import learn_warps

# Example of generating the warp model using the spatulas


warp_training_part_ids = {'body': [
    "./example_data/teapot_meshes/kettle_1/kettle_1_body.obj",
    "./example_data/teapot_meshes/kettle_2/kettle_2_body.obj",
    "./example_data/teapot_meshes/kettle_3/kettle_3_body.obj",
    "./example_data/teapot_meshes/kettle_4/kettle_4_body.obj",
    ],
    'spout': [
    "./example_data/teapot_meshes/kettle_1/kettle_1_spout.obj",
    "./example_data/teapot_meshes/kettle_2/kettle_2_spout.obj",
    "./example_data/teapot_meshes/kettle_3/kettle_3_spout.obj",
    "./example_data/teapot_meshes/kettle_4/kettle_4_spout.obj",
    ],
    'handle': [
    "./example_data/teapot_meshes/kettle_1/kettle_1_handle.obj",
    "./example_data/teapot_meshes/kettle_2/kettle_2_handle.obj",
    "./example_data/teapot_meshes/kettle_3/kettle_3_handle.obj",
    "./example_data/teapot_meshes/kettle_4/kettle_4_handle.obj",
    ],
    'lid': [
    "./example_data/teapot_meshes/kettle_1/kettle_1_lid.obj",
    "./example_data/teapot_meshes/kettle_2/kettle_2_lid.obj",
    "./example_data/teapot_meshes/kettle_3/kettle_3_lid.obj",
    "./example_data/teapot_meshes/kettle_4/kettle_4_lid.obj",
    ],
}

for part in warp_training_part_ids.keys():
    warp_training_meshes = [trimesh.load(f) for f in warp_training_part_ids[part]]

    n_dimensions = 3  # Number of dimensions of the latent space
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{part}_{timestr}_dim_{n_dimensions}"
    model_tag = "None"  # This is extraneous unless we're doing the part-based warping

    model_warp_data = learn_warps(warp_training_meshes, n_dimensions=n_dimensions)

    # Metadata keeps track of what objects were used to train the warp model
    training_object_ids = (
        warp_training_part_ids[part][: model_warp_data["canonical_idx"]]
        + warp_training_part_ids[part][model_warp_data["canonical_idx"] + 1 :]
    )

    model_metadata = utils.CanonShapeMetadata(
        model_tag,
        warp_training_part_ids[part][model_warp_data["canonical_idx"]],
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
