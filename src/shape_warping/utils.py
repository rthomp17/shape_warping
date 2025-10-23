# Code by Ondrej Biza and Skye Thompson

import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import trimesh
import trimesh.voxel.creation as vcreate
from cycpd import deformable_registration
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from shape_warping import viz_utils

inference_kwargs = {
    "train_latents": True,
    "train_scales": True,
    "train_poses": True,
}

NPF32 = NDArray[np.float32]
NPF64 = NDArray[np.float64]


@dataclass
class ObjParam:
    """Object shape and pose parameters."""

    position: NPF64 = np.array([0.0, 0.0, 0.0])
    quat: NPF64 = np.array([0.0, 0.0, 0.0, 1.0])
    latent: Optional[NPF32] = None
    scale: NPF32 = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def get_transform(self) -> NPF64:
        return pos_quat_to_transform(self.position, self.quat)


# Information about the model and the data it was trained on
@dataclass
class CanonShapeMetadata:
    experiment_tag: str
    canonical_id: str
    training_ids: List[str]
    part_label: Optional[int]  # For components of objects with multiple labeled parts


@dataclass
class CanonShape:
    """Canonical object with shape warping."""

    canonical_pcl: NPF32
    mesh_vertices: NPF32
    mesh_faces: NDArray[np.int32]
    center_transform: NPF32  # For saving the relative transform of parts
    metadata: CanonShapeMetadata
    contact_points: Optional[List[int]]
    pca: Optional[PCA] = None

    def __post_init__(self):
        if self.pca is not None:
            self.n_components = self.pca.n_components

    def to_pcd(self, obj_param: ObjParam) -> NPF32:
        if self.pca is not None and obj_param.latent is not None:
            pcd = self.canonical_pcl + self.pca.inverse_transform(
                obj_param.latent
            ).reshape(-1, 3)
        else:
            if self.pca is not None:
                print(
                    "WARNING: Skipping warping because we do not have a latent vector. We however have PCA."
                )
            pcd = np.copy(self.canonical_pcl)
        return pcd * obj_param.scale[None]

    def to_transformed_pcd(self, obj_param: ObjParam) -> NPF32:
        pcd = self.to_pcd(obj_param)
        trans = pos_quat_to_transform(obj_param.position, obj_param.quat)
        return transform_pcd(pcd, trans)

    def to_mesh(self, obj_param: ObjParam) -> trimesh.Trimesh:
        pcd = self.to_pcd(obj_param)
        # The mesh vertices are assumed to be at the start of the canonical_pcd.
        vertices = pcd[: len(self.mesh_vertices)]
        return trimesh.Trimesh(vertices, self.mesh_faces)

    def to_transformed_mesh(self, obj_param: ObjParam) -> trimesh.Trimesh:
        pcd = self.to_transformed_pcd(obj_param)
        # The mesh vertices are assumed to be at the start of the canonical_pcd.
        vertices = pcd[: len(self.mesh_vertices)]
        return trimesh.Trimesh(vertices, self.mesh_faces)

    @staticmethod
    def from_pickle(load_path: str) -> "CanonShape":
        with open(load_path, "rb") as f:
            data = pickle.load(f)

        if "canonical_obj" in data.keys():
            pcd = data["canonical_obj"]
        else:
            pcd = data["canonical_obj_pcl"]

        mesh_vertices = data["canonical_mesh_points"]
        mesh_faces = data["canonical_mesh_faces"]
        contact_points = None
        pca = None
        metadata = CanonShapeMetadata("", "", None, None)
        center_transform = np.eye(4)

        if "center_transform" in data:
            center_transform = data["center_transform"]
        if "pca" in data:
            pca = data["pca"]
        if "metadata" in data:
            metadata = data["metadata"]
        if "contact_points" in data:
            contact_points = data["contact_points"]

        return CanonShape(
            pcd,
            mesh_vertices,
            mesh_faces,
            center_transform,
            metadata,
            contact_points,
            pca,
        )


@dataclass
class ConstraintShape:
    """Does not warp. Only for use with the final alignment during part-based skill transfer."""

    canonical_pcl: NPF32
    center_transform: NPF32  # For saving the relative transform of parts
    metadata: CanonShapeMetadata
    pca: None

    def to_pcd(self, obj_param: ObjParam) -> NPF32:
        return self.constraint_pcd

    def to_transformed_pcd(self, obj_param: ObjParam) -> NPF32:
        pcd = self.to_pcd(obj_param)
        trans = pos_quat_to_transform(obj_param.position, obj_param.quat)
        return transform_pcd(pcd, trans)

    def to_mesh(self, obj_param: ObjParam) -> trimesh.Trimesh:
        raise NotImplementedError()

    def to_transformed_mesh(self, obj_param: ObjParam) -> trimesh.Trimesh:
        raise NotImplementedError()

    @staticmethod
    def from_part_reconstructions(
        part_reconstructions: dict[str, NDArray[np.float32]],
        part_transforms: List[NDArray[np.float32]],
    ):
        constraint_component_pcds = []
        for part in part_reconstructions.keys():
            constraint_component_pcds.append(
                transform_pcd(part_reconstructions[part], part_transforms[part])
            )

        combined_pcl = np.concatenate(constraint_component_pcds, axis=0)
        center_transform = pos_quat_to_transform(
            np.mean(np.unique(trunc(combined_pcl), axis=0), axis=0),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
        combined_pcl = transform_pcd(combined_pcl, np.linalg.inv(center_transform))

        metadata = CanonShapeMetadata("none", "none", ["none"], None)
        constraint_part = ConstraintShape(
            combined_pcl,
            center_transform,
            metadata,
            None,
        )
        return constraint_part


def trunc(values, decs=2):
    return np.trunc(values * 10**decs) / (10**decs)


def quat_to_rotm(quat: NPF64) -> NPF64:
    return Rotation.from_quat(quat).as_matrix()


def rotm_to_quat(rotm: NPF64) -> NPF64:
    return Rotation.from_matrix(rotm).as_quat()


def pos_quat_to_transform(
    pos: Union[Tuple[float, float, float], NPF64],
    quat: Union[Tuple[float, float, float, float], NPF64],
) -> NPF64:
    trans = np.eye(4).astype(np.float64)
    trans[:3, 3] = pos
    trans[:3, :3] = quat_to_rotm(np.array(quat))
    return trans


def transform_to_pos_quat(trans: NPF64) -> Tuple[NPF64, NPF64]:
    pos = trans[:3, 3]
    quat = rotm_to_quat(trans[:3, :3])
    # Just making sure.
    return pos.astype(np.float64), quat.astype(np.float64)


def transform_to_pos_rot(trans: NPF64) -> Tuple[NPF64, NPF64]:
    pos = trans[:3, 3]
    rot = trans[:3, :3]
    # Just making sure.
    return pos.astype(np.float64), rot.astype(np.float64)


def euler_to_quat(euler_rot):
    from scipy.spatial.transform import Rotation

    # Create a rotation object from Euler angles specifying axes of rotation
    rot = Rotation.from_euler("xyz", euler_rot)

    # Convert to quaternions and print
    rot_quat = rot.as_quat()
    return rot_quat


def euler_to_matrix(euler_rot):
    from scipy.spatial.transform import Rotation

    # Create a rotation object from Euler angles specifying axes of rotation
    rot = Rotation.from_euler("xyz", euler_rot)

    # Convert to quaternions and print
    rot_mat = rot.as_matrix()
    return rot_mat


def random_quat():
    return Rotation.random().as_quat()


def transform_pcd(pcd: NPF32, trans: NPF64, is_position: bool = True) -> NPF32:
    n = pcd.shape[0]
    cloud = pcd.T
    augment = np.ones((1, n)) if is_position else np.zeros((1, n))
    cloud = np.concatenate((cloud, augment), axis=0)
    cloud = np.dot(trans.astype(np.float32), cloud)
    cloud = cloud[0:3, :].T
    return cloud


def update_reconstruction_params_with_transform(demo_child_params, demo_transform):
    child_transform = pos_quat_to_transform(
        demo_child_params.position, demo_child_params.quat
    )
    new_child_transform = np.matmul(demo_transform, child_transform)
    demo_child_params.position, demo_child_params.quat = transform_to_pos_quat(
        new_child_transform
    )
    return demo_child_params


def best_fit_transform(A: NPF32, B: NPF32) -> Tuple[NPF64, NPF64, NPF64]:
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


def convex_decomposition(
    mesh: trimesh.base.Trimesh, save_path: Optional[str] = None
) -> List[trimesh.base.Trimesh]:
    """Convex decomposition of a mesh using testVHCAD through trimesh."""
    convex_meshes = trimesh.decomposition.convex_decomposition(
        mesh,
        resolution=1000000,
        depth=20,
        concavity=0.0025,
        planeDownsampling=4,
        convexhullDownsampling=4,
        alpha=0.05,
        beta=0.05,
        gamma=0.00125,
        pca=0,
        mode=0,
        maxNumVerticesPerCH=256,
        minVolumePerCH=0.0001,
        convexhullApproximation=1,
        oclDeviceID=0,
        debug=True,
    )
    # Bug / undocumented feature in Trimesh :<
    if not isinstance(convex_meshes, list):
        convex_meshes = [convex_meshes]

    if save_path is not None:
        decomposed_scene = trimesh.scene.Scene()
        for i, convex_mesh in enumerate(convex_meshes):
            decomposed_scene.add_geometry(convex_mesh, node_name=f"hull_{i}")
        print(save_path)
        decomposed_scene.export(save_path, file_type="obj")

    return convex_meshes


def farthest_point_sample(point: NPF32, npoint: int) -> Tuple[NPF32, NDArray[np.int32]]:
    # https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    if npoint > len(point):
        raise ValueError("Cannot sample more point then we have in the point cloud.")

    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    indices = centroids.astype(np.int32)
    point = point[indices]
    return point, indices


def trimesh_load_object(obj_path: str) -> trimesh.Trimesh:
    return trimesh.load(obj_path)


def trimesh_transform(
    mesh: trimesh.Trimesh,
    center: bool = True,
    scale: Optional[float] = None,
    rotation: Optional[NDArray] = None,
):
    # Automatically center. Also possibly rotate and scale.
    translation_matrix = np.eye(4)
    scaling_matrix = np.eye(4)
    rotation_matrix = np.eye(4)

    if center:
        t = mesh.centroid
        translation_matrix[:3, 3] = -t

    if scale is not None:
        scaling_matrix[0, 0] *= scale
        scaling_matrix[1, 1] *= scale
        scaling_matrix[2, 2] *= scale

    if rotation is not None:
        rotation_matrix[:3, :3] = rotation

    transform = np.matmul(
        scaling_matrix, np.matmul(rotation_matrix, translation_matrix)
    )
    mesh.apply_transform(transform)


def trimesh_create_verts_volume(
    mesh: trimesh.Trimesh, voxel_size: float = 0.015
) -> NDArray[np.float32]:
    voxels = vcreate.voxelize(mesh, voxel_size)
    return np.array(voxels.points, dtype=np.float32)


def trimesh_create_verts_surface(
    mesh: trimesh.Trimesh, num_surface_samples: Optional[int] = 1500
) -> NDArray[np.float32]:
    surf_points, _ = trimesh.sample.sample_surface_even(mesh, num_surface_samples)
    return np.array(surf_points, dtype=np.float32)


def trimesh_get_vertices_and_faces(
    mesh: trimesh.Trimesh,
) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    return vertices, faces


def scale_points_circle(
    points: List[NDArray[np.float32]], base_scale: float = 1.0
) -> List[NDArray[np.float32]]:
    points_cat = np.concatenate(points)
    assert len(points_cat.shape) == 2

    length = np.sqrt(np.sum(np.square(points_cat), axis=1))
    max_length = np.max(length, axis=0)

    new_points = []
    for p in points:
        new_points.append(base_scale * (p / max_length))

    return new_points


def cpd_transform(source, target, alpha: float = 2.0) -> Tuple[NDArray, NDArray]:
    source, target = source.astype(np.float64), target.astype(np.float64)
    reg = deformable_registration(
        **{"X": source, "Y": target, "tolerance": 0.00001}, alpha=alpha
    )
    reg.register()
    # Returns the gaussian means and their weights - WG is the warp of source to target
    return reg.W, reg.G


def sst_cost_batch(source: NDArray, target: NDArray) -> float:
    idx = np.sum(np.abs(source[None, :] - target[:, None]), axis=2).argmin(axis=0)
    return np.mean(
        np.linalg.norm(source - target[idx], axis=1)
    )  # TODO: test averaging instead of sum


def sst_cost_batch_pt(source, target):
    # for each vertex in source, find the closest vertex in target
    # we don't need to propagate the gradient here
    source_d, target_d = source.detach(), target.detach()
    indices = (
        (source_d[:, :, None] - target_d[:, None, :]).square().sum(dim=3).argmin(dim=2)
    )

    # go from [B, indices_in_target, 3] to [B, indices_in_source, 3] using target[batch_indices, indices]
    batch_indices = torch.arange(0, indices.size(0), device=indices.device)[
        :, None
    ].repeat(1, indices.size(1))
    c = torch.sqrt(
        torch.sum(torch.square(source - target[batch_indices, indices]), dim=2)
    )
    return torch.mean(c, dim=1)

    # simple version, about 2x slower
    # bigtensor = source[:, :, None] - target[:, None, :]
    # diff = torch.sqrt(torch.sum(torch.square(bigtensor), dim=3))
    # c = torch.min(diff, dim=2)[0]
    # return torch.mean(c, dim=1)


def warp_gen(
    canonical_index, objects, scale_factor=1.0, alpha: float = 2.0, visualize=False
):
    source = objects[canonical_index] * scale_factor
    targets = []

    for obj_idx, obj in enumerate(objects):
        if obj_idx != canonical_index:
            targets.append(obj * scale_factor)

    warps = []
    costs = []

    for target_idx, target in enumerate(targets):
        print(f"target {target_idx:d}")

        w, g = cpd_transform(target, source, alpha=alpha)

        warp = np.dot(g, w)
        warp = np.hstack(warp)

        tmp = source + warp.reshape(-1, 3)
        costs.append(sst_cost_batch(tmp, target))

        warps.append(warp)

        if visualize:
            viz_utils.show_pcds_plotly(
                {
                    "target": target,
                    "warp": source + warp.reshape(-1, 3),
                },
                center=True,
            )

    return warps, costs


def sst_pick_canonical(known_pts: List[NDArray[np.float32]]) -> int:
    # GPU acceleration makes this at least 100 times faster.
    known_pts = [torch.tensor(x, device="cpu", dtype=torch.float32) for x in known_pts]

    overall_costs = []
    for i in range(len(known_pts)):
        print(i)
        cost_per_target = []
        for j in range(len(known_pts)):
            if i != j:
                with torch.no_grad():
                    cost = sst_cost_batch_pt(
                        known_pts[i][None], known_pts[j][None]
                    ).cpu()

                cost_per_target.append(cost.item())

        overall_costs.append(np.mean(cost_per_target))
    print(f"overall costs: {str(overall_costs):s}")
    return np.argmin(overall_costs)


def pca_transform(distances, n_dimensions=4):
    pca = PCA(n_components=n_dimensions)
    p_components = pca.fit_transform(np.array(distances).squeeze())
    return p_components, pca


def rotation_distance(A, B):
    # Compute the relative rotation matrix
    R = np.dot(A, B.T)

    # Calculate the trace of the matrix
    trace = np.trace(R)

    # Clamp the trace value to the valid range [-1, 3] to avoid numerical errors
    trace = np.clip(trace, -1, 3)

    # Calculate the angle of rotation in radians
    angle = np.arccos((trace - 1) / 2)

    # Return the distance as a float
    return float(angle)


def pose_distance(trans1, trans2):
    pos1, rot1 = transform_to_pos_rot(trans1)
    pos2, rot2 = transform_to_pos_rot(trans2)

    # Compute the position (translation) distance
    position_distance = np.linalg.norm(pos1 - pos2)

    # Compute the orientation (rotation) distance using the rotation_distance function
    orientation_distance = rotation_distance(rot1, rot2)

    # Combine the position and orientation distances into a single metric (optional)
    # You can use different weights depending on the importance of position and orientation in your application
    position_weight = 1
    orientation_weight = 1
    total_distance = (
        position_weight * position_distance + orientation_weight * orientation_distance
    )

    return total_distance


# Returns the unit vector for vector
def unit_vector(vector):
    return vector / np.linalg.norm(vector)


# Returns the angle between vectors
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (
        (vec1 / np.linalg.norm(vec1)).reshape(3),
        (vec2 / np.linalg.norm(vec2)).reshape(3),
    )
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def k_largest_index_argpartition(a, k):
    idx = np.argpartition(a.ravel(), k)[:k]
    return np.column_stack(np.unravel_index(idx, a.shape))


def threshold_index_argwhere(a, thresh):
    idx = np.argwhere(a.ravel() < thresh)
    return np.column_stack(np.unravel_index(idx, a.shape))


def get_closest_point_pairs(pcl_1: NDArray, pcl_2: NDArray, k: int = 20):
    dists = np.sum(np.square(pcl_1[None] - pcl_2[:, None]), axis=-1)
    return k_largest_index_argpartition(dists, k)


def get_closest_point_pairs_thresh(
    pcl_1: NDArray, pcl_2: NDArray, thresh: float = 0.005
):
    dists = np.sum(np.square(pcl_1[None] - pcl_2[:, None]), axis=-1)
    return threshold_index_argwhere(dists, thresh)


def nearest_neighbor_distances(pcl_1: NDArray, pcl_2: NDArray) -> NDArray:
    """
    Compute the distance from each point in pcl_1 to its nearest neighbor in pcl_2.

    Args:
        pcl_1: First point cloud of shape (N, 3)
        pcl_2: Second point cloud of shape (M, 3)

    Returns:
        distances: Array of shape (N,) containing the distance from each point in pcl_1
                   to its closest point in pcl_2
    """
    # Compute pairwise squared distances: shape (M, N)
    # pcl_1[None] has shape (1, N, 3)
    # pcl_2[:, None] has shape (M, 1, 3)
    # Broadcasting gives (M, N, 3), then sum over last axis gives (M, N)
    dists_squared = np.sum(np.square(pcl_1[None] - pcl_2[:, None]), axis=-1)

    # Find minimum distance for each point in pcl_1 (axis 0 corresponds to pcl_2)
    min_dists_squared = np.min(dists_squared, axis=0)

    # Return Euclidean distances (take square root)
    return np.sqrt(min_dists_squared)


def center_pcl(pcl, return_centroid=False):
    pcl.shape[0]
    centroid = np.mean(pcl, axis=0)
    pcl = pcl - centroid
    if return_centroid:
        return pcl, centroid
    else:
        return pcl


# Constructs relational descriptors between parts
def get_part_labels(part_pairs, include_z=False):
    part_labels = {}
    for part_pair in part_pairs:
        ordered_part_names = list(part_pair.keys())
        ordered_part_names.sort()

        dists = np.sum(
            np.square(
                part_pair[ordered_part_names[0]][None]
                - part_pair[ordered_part_names[1]][:, None]
            ),
            axis=-1,
        )

        part_dists = {
            ordered_part_names[i]: np.min(dists, axis=i)
            for i in range(len(ordered_part_names))
        }

        # 0 where parts are near each other, 1 otherwise
        for part in ordered_part_names:
            if part not in part_labels.keys():
                part_labels[part] = []
            min_dist = np.min(part_dists[part])
            part_labels[part].append(
                np.where(
                    part_dists[part] - min_dist
                    < np.mean(part_dists[part] - min_dist) * 0.6,
                    np.zeros_like(part_dists[part]),
                    np.ones_like(part_dists[part]),
                )
            )
    if include_z:
        for part in part_labels.keys():
            mean_z = np.mean(part_pairs[0][part][:, 2])
            part_labels[part].append(
                np.where(
                    part_pairs[0][part][:, 2] - mean_z
                    < np.mean(part_pairs[0][part][:, 2] - mean_z) * 0.6,
                    np.zeros_like(part_pairs[0][part][:, 2]),
                    np.ones_like(part_pairs[0][part][:, 2]),
                )
            )

    return part_labels


# Processed canon objects and then
def get_canon_labels(
    part_pairs, part_canonicals, part_names, rescale=True, include_z=False
):
    # Generates adjacency between all listed parts if none provided
    if part_pairs is None:
        part_pairs = [
            {part_1: part_canonicals[part_1], part_2: part_canonicals[part_2]}
            for part_1, part_2 in itertools.combinations(part_names, r=2)
            if part_1 != part_2
        ]

    # Doing adjustment to the centered/scaled parts to accurately approximate these labels
    part_adjustment = {
        part: pos_quat_to_transform(
            part_canonicals[part].center_transform, (0, 0, 0, 1)
        )
        for part in part_names
    }

    # Realigning canonical parts - TODO make less bespoke
    if rescale:
        ten_scaled_parts = scale_points_circle(
            [part_canonicals[part].canonical_pcl for part in part_names], base_scale=10
        )

        adjusted_part_canon = {
            part_names[i]: transform_pcd(
                ten_scaled_parts[i], part_adjustment[part_names[i]]
            )
            for i in range(len(part_names))
        }

        contact_parts = scale_points_circle(
            [adjusted_part_canon[part] for part in part_names], base_scale=0.1
        )
    else:
        all_parts = [part_canonicals[part].canonical_pcl for part in part_names]
        adjusted_part_canon = {
            part_names[i]: transform_pcd(all_parts[i], part_adjustment[part_names[i]])
            for i in range(len(part_names))
        }
        contact_parts = [adjusted_part_canon[part] for part in part_names]

    # TODO fix canon teapot scaling
    if part_names[0] == "body":
        contact_parts[0] = scale_points_circle([contact_parts[0]], base_scale=0.075)[0]

    # Recreating the part pairs with the correct relative poses
    contact_pairs = []
    for pair in part_pairs:
        contact_pairs.append(
            {p: contact_parts[part_names.index(p)] for p in pair.keys()}
        )

    canon_labels = get_part_labels(contact_pairs, include_z=include_z)
    return canon_labels


# Utility function for visualizing the optimization process
def transform_history_to_mat(tf_history):
    pose_history = []
    for transform in tf_history:
        pos, rot = transform
        quat = rotm_to_quat(rot)
        transform_mats = []
        for p, q in zip(pos, quat):
            transform_mat = pos_quat_to_transform(p, q)
            transform_mats.append(transform_mat)
        pose_history.append(transform_mats)
    return pose_history
