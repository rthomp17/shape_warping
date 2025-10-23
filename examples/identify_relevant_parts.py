from shape_warping import utils, viz_utils
from shape_warping.skill_transfer import (
    infer_relpose,
    DemoScene,
    TestScene,
    get_demo_interaction_points_by_parts,
    ALIGNMENT_ONLY_INFERENCE_KWARGS,
)
from shape_warping.shape_reconstruction import (
    ObjectSE3Batch,
    warp_to_pcd_se3,
    mask_and_cost_batch_pt,
    get_relational_labels,
)

import os.path as osp
import numpy as np
import itertools
import pickle

child_object_parts = ["cup", "handle"]
parent_object_parts = ["trunk", "branch"]


def construct_heuristic(
    part_pairs, demo, warp_models, part_pcds, part_relationships, viz=False
):
    """
    Creates a compositional objective that when optimized, aligns all listed pairs of object parts (part pairs) on a test object (part_pcds) in accordance to their alignment in the provided demo
    """

    (
        child_relational_labels,
        _,
        canon_child_relational_labels,
        _,
    ) = get_relational_labels(part_pcds, warp_models, part_relationships)

    constraint_component_pcds = {}
    transform_estimates = {}

    # TODO: This repeats work across programs and parts, should fix that
    for pair in part_pairs:
        child_part, parent_part = pair
        child_reconstruction = demo.warp_models["child"][child_part].to_transformed_pcd(
            demo.reconstruction_params["child"][child_part]
        )

        child_params = demo.reconstruction_params["child"][child_part]
        parent_params = demo.reconstruction_params["parent"][parent_part]

        parent_to_child_part_transform_estimate = infer_relpose(
            demo.interaction_points[parent_part][child_part],
            warp_models["child"][child_part],
            warp_models["parent"][parent_part],
            child_params,
            parent_params,
        )

        constraint_component_pcds[pair] = child_reconstruction
        transform_estimates[pair] = parent_to_child_part_transform_estimate

    # Hacking to work with the warp optimization interface
    constraint_shape = utils.ConstraintShape.from_part_reconstructions(
        constraint_component_pcds, transform_estimates
    )

    if viz:
        viz_parent = {part: part_pcds["parent"][part] for part in part_pcds["parent"]}
        fig = viz_utils.show_pcds_plotly(
            {
                "constraint": utils.transform_pcd(constraint_shape.canonical_pcl, constraint_shape.center_transform),
            }
            | viz_parent
        )
        fig.show()

    # Transferring the relational labels
    # TODO: This doesn't handle multiple part relationships per part yet
    source_labels = np.array(
        list(
            itertools.chain(
                *[
                    child_relational_labels[part][0]
                    for _, part in enumerate([pair[0] for pair in part_pairs])
                ]
            )
        )
    )

    canon_labels = np.array(
        list(
            itertools.chain(
                *[
                    canon_child_relational_labels[part][0]
                    for _, part in enumerate([pair[0] for pair in part_pairs])
                ]
            )
        )
    )

    constraint_shape.canon_labels = canon_labels

    cost_function = lambda source, target, canon_part_labels: mask_and_cost_batch_pt(
        source, canon_part_labels, target, [source_labels], switch_sides=True
    )  # To handle the fact we're inverting the transform

    return constraint_shape, cost_function


def test_heuristic(warp_models, demo, test_scene, relevant_parts, viz=False):
    constraint_shape, cost_function = construct_heuristic(
        relevant_parts,
        demo,
        warp_models,
        test_scene.segmented_pcds,
        {
            "parent": test_scene.parent_part_relationships,
            "child": test_scene.child_part_relationships,
        },
        viz=True,
    )

    # Only align parts of the object that are in the heuristic
    alignment_obs = np.concatenate(
        [test_scene.segmented_pcds["child"][part[0]] for part in relevant_parts]
    )

    # Final alignment considering only the parts in relevant_parts
    combined_warp = ObjectSE3Batch(
        constraint_shape,
        alignment_obs,
        canon_labels=[constraint_shape.canon_labels],
        cost_function=cost_function,
        device="cuda",
        lr=1e-2,
        n_steps=100,
    )

    _, _, combined_params = warp_to_pcd_se3(
        combined_warp,
        n_angles=15,
        n_batches=12,
        inference_kwargs=ALIGNMENT_ONLY_INFERENCE_KWARGS,
    )

    final_transform = constraint_shape.center_transform @ np.linalg.inv(
        utils.pos_quat_to_transform(combined_params.position, combined_params.quat)
    )

    # Scene after Final transform

    if viz:
        fig = viz_utils.show_pcds_plotly(
            {
                "parent": demo.final_pcds["parent"],
                "aligned_alignment_observation": utils.transform_pcd(
                    alignment_obs, final_transform, True
                ),
                "constraint": utils.transform_pcd(
                    constraint_shape.canonical_pcl,
                    constraint_shape.center_transform,
                    True,
                ),
                "child": utils.transform_pcd(
                    demo.initial_pcds["child"], final_transform, True
                ),
            },
            title="PCD Alignment Constraint for the given part relationships"
        )
        fig.show()
    cost = utils.pose_distance(final_transform, demo.child_transform)
    return cost


if __name__ == "__main__":
    demo_path = osp.join(
        "example_data", "mug_on_rack_demos", "mug_on_tree_demonstration.pkl"
    )
    demo = DemoScene.from_pickle(demo_path)

    demo.show_demo()

    # Load shape warping models. These have been pretrained from ~5 examples.
    warp_models = {
        "parent": {
            "trunk": utils.CanonShape.from_pickle(
                "example_data/rack_models/example_pretrained_trunk"
            ),
            "branch": utils.CanonShape.from_pickle(
                "example_data/rack_models/example_pretrained_branch"
            ),
        },
        "child": {
            "cup": utils.CanonShape.from_pickle(
                "example_data/mug_models/example_pretrained_cup"
            ),
            "handle": utils.CanonShape.from_pickle(
                "example_data/mug_models/example_pretrained_handle"
            ),
        },
    }

    # Get the interaction point and demo shape info if it's not already in the demo dict
    if demo.interaction_points is None:
        demo.interaction_points = get_demo_interaction_points_by_parts(
            demo, warp_models
        )
        demo_data = pickle.load(open(demo_path, "rb"))
        demo_data["interaction_points"] = demo.interaction_points
        demo_data["child_warp_model"] = warp_models["child"]
        demo_data["parent_warp_model"] = warp_models["parent"]
        demo_data["child_reconstruction_params"] = demo.reconstruction_params["child"]
        demo_data["parent_reconstruction_params"] = demo.reconstruction_params["parent"]
        pickle.dump(demo_data, open(demo_path, "wb"))

    test_part_pair = [("handle", "branch")]

    # We evaluate the heuristic by reconstructing the demonstration transform
    cost = test_heuristic(
        warp_models,
        demo,
        TestScene(demo.initial_pcds, demo.segmented_initial_pcds),
        test_part_pair,
    )
    print("Part relationship evaluation cost:", cost)
