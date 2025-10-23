"""
Part-based shape warping utilities for transferring skills between different object parts.
Extends the core shape warping functionality to handle multi-part objects and skill transfer.
"""

import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional, Union
from numpy.typing import NDArray

from shape_warping import utils, viz_utils
from shape_warping.skill_transfer import warp_interaction_points, infer_relpose
from shape_warping.shape_reconstruction import ObjectWarpingSE2Batch, warp_to_pcd_se2


def load_part_meshes(mesh_paths: Dict[str, str]) -> Dict[str, trimesh.Trimesh]:
    """
    Load multiple part meshes from file paths.

    Args:
        mesh_paths: Dictionary mapping part names to mesh file paths

    Returns:
        Dictionary mapping part names to loaded meshes
    """
    meshes = {}
    for part_name, path in mesh_paths.items():
        try:
            meshes[part_name] = utils.trimesh_load_object(path)
            print(f"Loaded {part_name} mesh from {path}")
        except Exception as e:
            print(f"Failed to load {part_name} mesh from {path}: {e}")
    return meshes


def convert_meshes_to_pointclouds(
    meshes: Dict[str, trimesh.Trimesh],
    num_points: int = 1000,
    scale_factor: float = 0.2,
) -> Dict[str, NDArray]:
    """
    Convert dictionary of meshes to point clouds with consistent scaling.

    Args:
        meshes: Dictionary of meshes {part_name: mesh}
        num_points: Number of points to sample from each mesh surface
        scale_factor: Scaling factor for point clouds

    Returns:
        Dictionary of point clouds {part_name: pcd}
    """
    pcds = {}
    pcd_list = []
    part_order = []

    # Convert each mesh to point cloud
    for part_name, mesh in meshes.items():
        pcd = utils.trimesh_create_verts_surface(mesh, num_points)
        pcds[part_name] = pcd
        pcd_list.append(pcd)
        part_order.append(part_name)

    # Scale consistently across all parts
    if pcd_list:
        scaled_pcds = utils.scale_points_circle(pcd_list, base_scale=scale_factor)
        for i, part_name in enumerate(part_order):
            pcds[part_name] = scaled_pcds[i]

    return pcds


def get_relational_labels(part_pcds, warp_models, part_relationships, include_z=False):
    """
    Helper function for getting the relational labels that help with shape warping.
    """

    # These are for improving the results of shape warping, but aren't used for the final pose inference
    child_relational_labels = utils.get_part_labels(
        [
            {p: part_pcds["child"][p] for p in relation}
            for relation in part_relationships["child"]
        ],
        include_z=include_z,
    )

    parent_relational_labels = utils.get_part_labels(
        [
            {p: part_pcds["parent"][p] for p in relation}
            for relation in part_relationships["parent"]
        ],
        include_z=include_z,
    )

    canon_child_relational_labels = utils.get_canon_labels(
        [
            {p: warp_models["child"][p].canonical_pcl for p in relation}
            for relation in part_relationships["child"]
        ],
        warp_models["child"],
        list(part_pcds["child"].keys()),
        rescale=False,
        include_z=include_z,
    )

    canon_parent_relational_labels = utils.get_canon_labels(
        [
            {p: warp_models["parent"][p].canonical_pcl for p in relation}
            for relation in part_relationships["parent"]
        ],
        warp_models["parent"],
        list(part_pcds["parent"].keys()),
        rescale=False,
        include_z=include_z,
    )

    return (
        child_relational_labels,
        parent_relational_labels,
        canon_child_relational_labels,
        canon_parent_relational_labels,
    )


def fit_parts_to_shape_models(
    part_pcds: Dict[str, NDArray],
    shape_models: Dict[str, utils.CanonShape],
    device: str = "cpu",
    lr: float = 1e-2,
    n_steps: int = 100,
) -> Tuple[Dict[str, NDArray], Dict[str, utils.ObjParam]]:
    """
    Fit part point clouds to their corresponding shape models.

    Args:
        part_pcds: Dictionary of part point clouds
        shape_models: Dictionary of CanonShape models for each part
        device: Device for optimization ("cpu" or "cuda")
        lr: Learning rate for optimization
        n_steps: Number of optimization steps

    Returns:
        Tuple of (warped_pcds, reconstruction_params)
    """
    warped_pcds = {}
    reconstruction_params = {}

    inference_kwargs = {
        "train_latents": True,
        "train_poses": True,
        "train_centers": True,
        "train_scales": True,
    }

    for part_name in part_pcds.keys():
        if part_name not in shape_models:
            print(f"Warning: No shape model found for part '{part_name}', skipping...")
            continue

        print(f"Fitting {part_name} to shape model...")

        warp_model = shape_models[part_name]

        warp_reconstruction = ObjectWarpingSE2Batch(
            warp_model,
            part_pcds[part_name],
            device=device,
            lr=lr,
            n_steps=n_steps,
        )

        warped_pcd, _, params = warp_to_pcd_se2(
            warp_reconstruction,
            n_angles=5,
            n_batches=1,
            inference_kwargs=inference_kwargs,
        )

        warped_pcds[part_name] = warped_pcd
        reconstruction_params[part_name] = params

    return warped_pcds, reconstruction_params


def extract_part_interaction_points(
    source_pcds: Dict[str, NDArray],
    target_pcds: Dict[str, NDArray],
    source_models: Dict[str, utils.CanonShape],
    target_models: Dict[str, utils.CanonShape],
    source_params: Dict[str, utils.ObjParam],
    target_params: Dict[str, utils.ObjParam],
    part_pairs: List[Tuple[str, str]],
    delta: float = 0.055,
) -> Dict[Tuple[str, str], Dict]:
    """
    Extract interaction points between pairs of parts.

    Args:
        source_pcds: Source part point clouds
        target_pcds: Target part point clouds
        source_models: Source shape models
        target_models: Target shape models
        source_params: Source reconstruction parameters
        target_params: Target reconstruction parameters
        part_pairs: List of (source_part, target_part) tuples
        delta: Distance threshold for nearby points

    Returns:
        Dictionary mapping part pairs to interaction points
    """
    from shape_warping.skill_transfer import get_interaction_points

    interaction_points = {}

    for source_part, target_part in part_pairs:
        if (
            source_part not in source_pcds
            or target_part not in target_pcds
            or source_part not in source_models
            or target_part not in target_models
        ):
            print(f"Warning: Missing data for pair ({source_part}, {target_part})")
            continue

        print(
            f"Extracting interaction points between {source_part} and {target_part}..."
        )

        knns, deltas, target_indices = get_interaction_points(
            source_pcds[source_part],
            target_pcds[target_part],
            source_models[source_part],
            source_params[source_part],
            target_models[target_part],
            target_params[target_part],
            delta=delta,
        )

        interaction_points[(source_part, target_part)] = {
            "knns": knns,
            "deltas": deltas,
            "target_indices": target_indices,
        }

    return interaction_points


def transfer_skills_between_part_pairs(
    interaction_points: Dict[Tuple[str, str], Dict],
    source_models: Dict[str, utils.CanonShape],
    target_models: Dict[str, utils.CanonShape],
    source_params: Dict[str, utils.ObjParam],
    target_params: Dict[str, utils.ObjParam],
    part_mapping: Dict[str, str],
) -> Dict[str, Dict]:
    """
    Transfer skills between corresponding part pairs using interaction points.

    Args:
        interaction_points: Interaction points from demonstration
        source_models: Source shape models
        target_models: Target shape models
        source_params: Source reconstruction parameters
        target_params: Target reconstruction parameters
        part_mapping: Mapping from source parts to target parts

    Returns:
        Dictionary of transferred skills for each target part
    """
    transferred_skills = {}

    for source_part, target_part in part_mapping.items():
        # Find interaction points for this part pair
        interaction_key = None
        for key in interaction_points.keys():
            if key[0] == source_part or key[1] == source_part:
                interaction_key = key
                break

        if interaction_key is None:
            print(f"Warning: No interaction points found for source part {source_part}")
            continue

        if (
            source_part not in source_models
            or target_part not in target_models
            or source_part not in source_params
            or target_part not in target_params
        ):
            print(f"Warning: Missing models/params for {source_part} -> {target_part}")
            continue

        print(f"Transferring skill from {source_part} to {target_part}...")

        part_interaction_points = interaction_points[interaction_key]

        # Transfer interaction points using shape warping
        transferred_points = warp_interaction_points(
            part_interaction_points,
            source_models[source_part],
            target_models[target_part],
            source_params[source_part],
            target_params[target_part],
        )

        # Compute relative transform for skill execution
        relative_transform = infer_relpose(
            part_interaction_points,
            source_models[source_part],
            target_models[target_part],
            source_params[source_part],
            target_params[target_part],
        )

        transferred_skills[target_part] = {
            "interaction_points": transferred_points,
            "relative_transform": relative_transform,
            "source_part": source_part,
        }

    return transferred_skills


def apply_skill_transforms(
    target_pcds: Dict[str, NDArray], transferred_skills: Dict[str, Dict]
) -> Dict[str, NDArray]:
    """
    Apply skill transforms to target point clouds.

    Args:
        target_pcds: Target part point clouds
        transferred_skills: Dictionary of transferred skills with transforms

    Returns:
        Dictionary of transformed point clouds
    """
    transformed_pcds = {}

    for part_name, pcd in target_pcds.items():
        if part_name in transferred_skills:
            skill_data = transferred_skills[part_name]
            if "relative_transform" in skill_data:
                transform = skill_data["relative_transform"]
                transformed_pcd = utils.transform_pcd(pcd, transform)
                transformed_pcds[part_name] = transformed_pcd
                print(f"Applied skill transform to {part_name}")
            else:
                transformed_pcds[part_name] = pcd
        else:
            transformed_pcds[part_name] = pcd

    return transformed_pcds


def visualize_part_warping_results(
    original_pcds: Dict[str, NDArray],
    warped_pcds: Dict[str, NDArray],
    transferred_skills: Optional[Dict[str, Dict]] = None,
    transformed_pcds: Optional[Dict[str, NDArray]] = None,
    title_prefix: str = "",
) -> None:
    """
    Visualize the results of part warping and skill transfer.

    Args:
        original_pcds: Original part point clouds
        warped_pcds: Warped part point clouds
        transferred_skills: Optional skill transfer results
        transformed_pcds: Optional transformed point clouds after skill application
        title_prefix: Prefix for visualization titles
    """
    viz_dict = {}

    # Add original and warped point clouds
    for part_name in original_pcds.keys():
        viz_dict[f"{title_prefix}{part_name} Original"] = original_pcds[part_name]
        if part_name in warped_pcds:
            viz_dict[f"{title_prefix}{part_name} Warped"] = warped_pcds[part_name]

    # Add skill interaction points if available
    if transferred_skills:
        for part_name, skill_data in transferred_skills.items():
            if "interaction_points" in skill_data:
                points = skill_data["interaction_points"]
                if "parent" in points:
                    viz_dict[f"{title_prefix}{part_name} Skill Points"] = points[
                        "parent"
                    ]
                if "child" in points:
                    viz_dict[f"{title_prefix}{part_name} Child Points"] = points[
                        "child"
                    ]

    # Add transformed point clouds if available
    if transformed_pcds:
        for part_name, pcd in transformed_pcds.items():
            viz_dict[f"{title_prefix}{part_name} Transformed"] = pcd

    # Create and show visualization
    if viz_dict:
        fig = viz_utils.show_pcds_plotly(viz_dict)
        fig.show()
    else:
        print("No data to visualize")


def validate_part_data(
    part_pcds: Dict[str, NDArray],
    shape_models: Dict[str, utils.CanonShape],
    required_parts: List[str],
) -> bool:
    """
    Validate that all required parts have both point clouds and shape models.

    Args:
        part_pcds: Dictionary of part point clouds
        shape_models: Dictionary of shape models
        required_parts: List of required part names

    Returns:
        True if all required data is available, False otherwise
    """
    missing_pcds = [part for part in required_parts if part not in part_pcds]
    missing_models = [part for part in required_parts if part not in shape_models]

    if missing_pcds:
        print(f"Missing point clouds for parts: {missing_pcds}")

    if missing_models:
        print(f"Missing shape models for parts: {missing_models}")

    return len(missing_pcds) == 0 and len(missing_models) == 0
