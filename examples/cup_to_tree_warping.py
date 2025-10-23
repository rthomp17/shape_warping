import numpy as np
import trimesh
import os.path as osp
from shape_warping import utils, viz_utils
from shape_warping.skill_transfer import *
from shape_warping.shape_reconstruction import ObjectWarpingSE2Batch, warp_to_pcd_se2
from shape_warping.part_warping import (
    load_part_meshes,
    convert_meshes_to_pointclouds,
    fit_parts_to_shape_models,
    extract_part_interaction_points,
    transfer_skills_between_part_pairs,
    apply_skill_transforms,
    visualize_part_warping_results,
    validate_part_data,
)


def demonstrate_cup_to_tree_warping(
    mesh_paths, model_paths, skill_demo_points=None, visualize=True
):
    """
    Main function demonstrating shape warping from cup/handle to branch/trunk using part warping utilities.

    Args:
        mesh_paths: Dictionary with keys 'cup', 'handle', 'branch', 'trunk' mapping to mesh file paths
        model_paths: Dictionary with keys 'cup', 'handle', 'branch', 'trunk' mapping to shape model paths
        skill_demo_points: Optional skill demonstration points for transfer
        visualize: Whether to show visualizations

    Returns:
        Dictionary containing warped point clouds and transferred skills
    """

    # Load meshes using part warping utilities
    print("Loading meshes...")
    source_meshes = load_part_meshes(
        {"cup": mesh_paths["cup"], "handle": mesh_paths["handle"]}
    )

    target_meshes = load_part_meshes(
        {"branch": mesh_paths["branch"], "trunk": mesh_paths["trunk"]}
    )

    # Convert to point clouds
    print("Converting meshes to point clouds...")
    source_pcds = convert_meshes_to_pointclouds(
        source_meshes, num_points=1000, scale_factor=0.2
    )
    target_pcds = convert_meshes_to_pointclouds(
        target_meshes, num_points=1000, scale_factor=0.2
    )

    # Load shape models
    print("Loading shape models...")
    source_models = {
        "cup": utils.CanonShape.from_pickle(model_paths["cup"]),
        "handle": utils.CanonShape.from_pickle(model_paths["handle"]),
    }

    target_models = {
        "branch": utils.CanonShape.from_pickle(model_paths["branch"]),
        "trunk": utils.CanonShape.from_pickle(model_paths["trunk"]),
    }

    # Validate data
    if not validate_part_data(source_pcds, source_models, ["cup", "handle"]):
        raise ValueError("Missing source part data")
    if not validate_part_data(target_pcds, target_models, ["branch", "trunk"]):
        raise ValueError("Missing target part data")

    # Fit parts to shape models
    print("Warping source parts...")
    source_warped, source_params = fit_parts_to_shape_models(source_pcds, source_models)

    print("Warping target parts...")
    target_warped, target_params = fit_parts_to_shape_models(target_pcds, target_models)

    # Visualize warping results
    if visualize:
        print("Visualizing warping results...")
        visualize_part_warping_results(
            source_pcds, source_warped, title_prefix="Source "
        )
        visualize_part_warping_results(
            target_pcds, target_warped, title_prefix="Target "
        )

    results = {
        "source_original": source_pcds,
        "target_original": target_pcds,
        "source_warped": source_warped,
        "target_warped": target_warped,
        "source_params": source_params,
        "target_params": target_params,
    }

    # Transfer skills if demonstration points provided
    if skill_demo_points is not None:
        print("Transferring skills between parts...")
        part_mapping = {"cup": "branch", "handle": "trunk"}

        # Extract interaction points from demonstration
        part_pairs = [("cup", "handle")]  # Example pair for skill demonstration
        interaction_points = extract_part_interaction_points(
            source_pcds,
            source_pcds,  # Using source for demo
            source_models,
            source_models,
            source_params,
            source_params,
            part_pairs,
        )

        # Transfer skills to target parts
        transferred_skills = transfer_skills_between_part_pairs(
            interaction_points,
            source_models,
            target_models,
            source_params,
            target_params,
            part_mapping,
        )

        # Apply skill transforms
        transformed_pcds = apply_skill_transforms(target_warped, transferred_skills)

        results["transferred_skills"] = transferred_skills
        results["transformed_pcds"] = transformed_pcds

        # Visualize skill transfer results
        if visualize:
            print("Visualizing skill transfer results...")
            visualize_part_warping_results(
                target_warped,
                target_warped,
                transferred_skills,
                transformed_pcds,
                title_prefix="Skill Transfer ",
            )

    return results


# Example usage
if __name__ == "__main__":
    # Example paths - users would provide their own
    mesh_paths = {
        "cup": "example_data/cup_meshes/cup.obj",
        "handle": "example_data/handle_meshes/handle.obj",
        "branch": "example_data/branch_meshes/branch.obj",
        "trunk": "example_data/trunk_meshes/trunk.obj",
    }

    model_paths = {
        "cup": "example_data/cup_models/cup_shape_model",
        "handle": "example_data/handle_models/handle_shape_model",
        "branch": "example_data/branch_models/branch_shape_model",
        "trunk": "example_data/trunk_models/trunk_shape_model",
    }

    # Check if example files exist before running
    all_paths = {**mesh_paths, **model_paths}
    all_exist = all(osp.exists(path) for path in all_paths.values())

    if all_exist:
        print("Running cup to tree warping demonstration...")

        # Example skill demonstration points (would come from actual demonstration)
        example_skill_points = {
            ("cup", "handle"): {
                "knns": np.array([[0, 1, 2]]),  # Example indices
                "deltas": np.array([[[0.1, 0.0, 0.0]]]),  # Example deltas
                "target_indices": np.array([10]),  # Example target indices
            }
        }

        results = demonstrate_cup_to_tree_warping(
            mesh_paths,
            model_paths,
            skill_demo_points=example_skill_points,
            visualize=True,
        )
        print("Demonstration completed!")
        print(f"Results contain: {list(results.keys())}")

    else:
        print("Example files not found. Please provide valid mesh and model paths.")
        print("Required files:")
        for name, path in all_paths.items():
            exists = "✓" if osp.exists(path) else "✗"
            print(f"  {exists} {name}: {path}")

        print("\nTo use this script with your own data:")
        print("1. Provide mesh files (.obj) for cup, handle, branch, and trunk parts")
        print(
            "2. Provide trained shape models for each part (saved with utils.CanonShape)"
        )
        print("3. Optionally provide skill demonstration points for transfer")
        print("4. Run: python cup_to_tree_warping.py")
