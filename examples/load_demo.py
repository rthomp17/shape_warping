import numpy as np
import os.path as osp
import pickle
from scipy.spatial.transform import Rotation
import trimesh
import json
from shape_warping import viz_utils

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def to_json_friendly(obj):
    """
    Convert a dictionary (or other object) to a JSON-friendly format.

    Handles conversion of:
    - PyTorch tensors -> lists
    - NumPy arrays -> lists
    - NumPy scalars -> Python native types
    - Nested dictionaries and lists (recursive)
    - Sets and tuples -> lists
    - Other common types

    Args:
        obj: The object to convert (dict, list, tensor, array, etc.)

    Returns:
        A JSON-serializable version of the input object
    """
    # Handle None
    if obj is None:
        return None

    # Handle PyTorch tensors
    if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()

    # Handle NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle NumPy scalar types
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # Handle NumPy bool
    if isinstance(obj, np.bool_):
        return bool(obj)

    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {key: to_json_friendly(value) for key, value in obj.items()}

    # Handle lists and tuples recursively
    if isinstance(obj, (list, tuple)):
        return [to_json_friendly(item) for item in obj]

    # Handle sets
    if isinstance(obj, set):
        return [to_json_friendly(item) for item in obj]

    # Handle basic JSON-serializable types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # For objects with custom attributes, try to convert to string
    # This handles objects that might not be directly serializable
    try:
        # Try to see if it's already JSON serializable
        import json
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # If not serializable, convert to string representation
        return str(obj)


def transform_pcd(pcd, trans, is_position):
    n = pcd.shape[0]
    cloud = pcd.T
    augment = np.ones((1, n)) if is_position else np.zeros((1, n))
    cloud = np.concatenate((cloud, augment), axis=0)
    cloud = np.dot(trans.astype(np.float32), cloud)
    cloud = cloud[0:3, :].T
    return cloud


def quat_to_rotm(quat):
    return Rotation.from_quat(quat).as_matrix()


def rotm_to_quat(rotm):
    return Rotation.from_matrix(rotm).as_quat()


def pos_quat_to_transform(
    pos,
    quat,
):
    trans = np.eye(4).astype(np.float64)
    trans[:3, 3] = pos
    trans[:3, :3] = quat_to_rotm(np.array(quat))
    return trans


def transform_to_pos_quat(trans):
    pos = trans[:3, 3]
    quat = rotm_to_quat(trans[:3, :3])
    # Just making sure.
    return pos.astype(np.float64), quat.astype(np.float64)


def update_reconstruction_params_with_transform(demo_child_params, demo_transform):
    child_transform = pos_quat_to_transform(
        demo_child_params.position, demo_child_params.quat
    )
    new_child_transform = np.matmul(demo_transform, child_transform)
    demo_child_params.position, demo_child_params.quat = transform_to_pos_quat(
        new_child_transform
    )
    return demo_child_params


def best_fit_transform(A, B):
    """
    https://github.com/ClayFlannigan/icp/blob/master/icp.py
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1).astype(np.float64)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R.astype(np.float64), t.astype(np.float64)


def get_relative_transform_interaction_points(demo, part_pairs=[["branch", "handle"]]):
    interaction_points = demo["interaction_points"]

    all_parent_targets = []
    all_child_targets = []
    # Concatenating rather than do the full part alignment step for the sake of simplicity.
    for pair in part_pairs:
        parent_part, child_part = pair
        knns = interaction_points[parent_part][child_part]["knns"]
        deltas = interaction_points[parent_part][child_part]["deltas"]
        target_indices = interaction_points[parent_part][child_part]["target_indices"]

        anchors = demo["child_warp_model"][child_part].to_pcd(
            demo["child_reconstruction_params"][child_part]
        )[knns]
        targets_child = np.mean(anchors + deltas, axis=1)
        targets_parent = demo["parent_warp_model"][parent_part].to_pcd(
            demo["parent_reconstruction_params"][parent_part]
        )[target_indices]
        all_parent_targets.append(targets_parent)
        all_child_targets.append(targets_child)

    all_parent_targets = np.concatenate(all_parent_targets, axis=0)
    all_child_targets = np.concatenate(all_child_targets, axis=0)

    # Canonical source obj to canonical target obj.
    trans_cs_to_ct, _, _ = best_fit_transform(all_child_targets, all_parent_targets)

    # hacky thing because of how the warps are saved
    final_transform = pos_quat_to_transform(
        final_poses["child"][:3], final_poses["child"][3:]
    ) @ np.linalg.inv(
        pos_quat_to_transform(start_poses["child"][:3], start_poses["child"][3:])
    )

    demo["child_reconstruction_params"][child_part] = (
        update_reconstruction_params_with_transform(
            demo["child_reconstruction_params"][child_part], final_transform
        )
    )

    trans_s_to_b = pos_quat_to_transform(
        demo["child_reconstruction_params"][child_part].position,
        demo["child_reconstruction_params"][child_part].quat,
    )

    trans_t_to_b = pos_quat_to_transform(
        demo["parent_reconstruction_params"][parent_part].position,
        demo["parent_reconstruction_params"][parent_part].quat,
    )

    # Compute relative transform.
    trans_s_to_t = trans_t_to_b @ trans_cs_to_ct @ np.linalg.inv(trans_s_to_b)
    return np.linalg.inv(trans_s_to_t)


def get_object_meshes(demo):
    child_meshes = [
        demo["child_warp_model"][part].to_transformed_mesh(
            demo["child_reconstruction_params"][part]
        )
        for part in demo["child_part_names"]
    ]
    parent_meshes = [
        demo["parent_warp_model"][part].to_transformed_mesh(
            demo["parent_reconstruction_params"][part]
        )
        for part in demo["parent_part_names"]
    ]
    return trimesh.util.concatenate(child_meshes), trimesh.util.concatenate(
        parent_meshes
    )






if __name__ == "__main__":
    # Load from the pickle file
    demo_path = osp.join(
        "example_data", "mug_on_rack_demos", "mug_on_tree_demonstration_np_back_compatible.pkl"
    )
    demo = pickle.load(open(demo_path, "rb"))


    demo["child_part_names"] = ["cup", "handle"]
    demo["parent_part_names"] = ["trunk", "branch"]

    # Pointclouds and world frame poses before placement
    initial_pcds = {
        "parent": demo["start_pcds"]["parent"],
        "child": demo["start_pcds"]["child"],
    }

    start_poses = {
        "parent": demo["start_object_poses"]["parent"],
        "child": demo["start_object_poses"]["child"],
    }

    # Pointclouds and world frameposes after placement
    final_pcds = {
        "parent": demo["final_pcds"]["parent"],
        "child": demo["final_pcds"]["child"],
    }

    final_poses = {
        "parent": demo["final_object_poses"]["parent"],
        "child": demo["final_object_poses"]["child"],
    }

    # Trimesh meshes
    child_mesh, parent_mesh = get_object_meshes(demo)

    # Reconstructed transform from initial mug placement to the final mug placement in the world_frame
    relative_transform_rack_to_mug = get_relative_transform_interaction_points(demo)

    print(relative_transform_rack_to_mug)
    # Ground truth transform from initial mug placement to the final mug placement in the world_frame
    final_transform = pos_quat_to_transform(
        final_poses["child"][:3], final_poses["child"][3:]
    ) @ np.linalg.inv(
        pos_quat_to_transform(start_poses["child"][:3], start_poses["child"][3:])
    )

    print(final_transform)
    # trans_cs_to_ct is the mug's position in the estimated rack frame
