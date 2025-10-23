from shape_warping import utils, viz_utils
from shape_warping.skill_transfer import (
    infer_relpose,
    DemoScene,
    TestScene,
    get_interaction_points,
    warp_interaction_points,
)
from shape_warping.shape_reconstruction import (
    ObjectWarpingSE3Batch,
    ObjectSE3Batch,
    warp_to_pcd_se3,
    ObjectWarpingSE2Batch,
    warp_to_pcd_se3_hemisphere,
    warp_to_pcd_se2,
    mask_and_cost_batch_pt,
)
import os.path as osp
import numpy as np
import itertools
import pickle

child_object_parts = ["cup", "handle"]
parent_object_parts = ["trunk", "branch"]

# For shape warping and feature transfer
WARPING_INFERENCE_KWARGS = {
    "train_latents": True,
    "train_poses": True,
    "train_centers": True,
}

# For the final pose inference, after warping is done
ALIGNMENT_ONLY_INFERENCE_KWARGS = {
    "train_poses": True,
    "train_centers": True,
}


def reconstruct_part_shape(
    warp_model,
    target_pcd,
    target_labels,
    canon_labels,
    inference_kwargs,
    viz=False,
    warp_mode="SE3",
):
    cost_function = (
        lambda source,
        test_object,
        canon_part_labels,
        latent,
        scale,
        initial: mask_and_cost_batch_pt(
            test_object,
            target_labels,
            source,
            canon_part_labels,
        )
    )
    if warp_mode == "SE3":
        warp = ObjectWarpingSE3Batch(
            warp_model,
            target_pcd,
            canon_labels=canon_labels,
            cost_function=cost_function,
            device="cuda",
            lr=1e-2,
            n_steps=100,
            object_size_reg=0.2,
        )

        reconstructed_pcd, _, warp_params = warp_to_pcd_se3(
            warp,
            n_angles=5,
            n_batches=12,
            inference_kwargs=inference_kwargs,
        )
    elif warp_mode == "SE2":
        warp = ObjectWarpingSE2Batch(
            warp_model,
            target_pcd,
            canon_labels=canon_labels,
            cost_function=cost_function,
            device="cuda",
            lr=1e-2,
            n_steps=100,
            object_size_reg=0.2,
        )
        reconstructed_pcd, _, warp_params = warp_to_pcd_se2(
            warp,
            n_angles=5,
            n_batches=12,
            inference_kwargs=inference_kwargs,
        )
    else:
        raise NotImplementedError()

    if viz:
        for i, label in enumerate(target_labels):
            fig = viz_utils.show_pcds_plotly(
                {
                    f"label_{i}_target_0": target_pcd[label == 0],
                    f"label_{i}_target_1": target_pcd[label == 1],
                    f"label_{i}_reconstruction_0": reconstructed_pcd[
                        canon_labels[i] == 0
                    ],
                    f"label_{i}_reconstruction_1": reconstructed_pcd[
                        canon_labels[i] == 1
                    ],
                }
            )
            fig.show()

    return reconstructed_pcd, warp_params


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


def construct_heuristic(
    part_pairs, demo, warp_models, part_pcds, part_relationships, viz=False
):
    """
    Creates a compositional objective that when optimized, aligns all listed pairs of object parts (part pairs) on a test object (part_pcds) in accordance to their alignment in the provided demo
    """

    (
        child_relational_labels,
        parent_relational_labels,
        canon_child_relational_labels,
        canon_parent_relational_labels,
    ) = get_relational_labels(part_pcds, warp_models, part_relationships)

    constraint_component_pcds = {}
    transform_estimates = {}

    # TODO: This repeats work across programs and parts, should fix that
    for pair in part_pairs:
        child_part, parent_part = pair

        # child_reconstruction, child_params = reconstruct_part_shape(
        #     warp_models['child'][child_part],
        #     part_pcds['child'][child_part],
        #     child_relational_labels[child_part],
        #     canon_labels=canon_child_relational_labels[child_part],
        #     inference_kwargs=WARPING_INFERENCE_KWARGS,
        # )
        # parent_reconstruction, parent_params = reconstruct_part_shape(
        #     warp_models['parent'][parent_part],
        #     part_pcds['parent'][parent_part],
        #     parent_relational_labels[parent_part],
        #     canon_labels=canon_parent_relational_labels[parent_part],
        #     inference_kwargs=WARPING_INFERENCE_KWARGS,
        # )

        child_reconstruction = demo.warp_models["child"][child_part].to_transformed_pcd(
            demo.child_reconstruction_params[child_part]
        )
        parent_reconstruction = demo.warp_models["parent"][
            parent_part
        ].to_transformed_pcd(demo.parent_reconstruction_params[parent_part])
        child_params = demo.child_reconstruction_params[child_part]
        parent_params = demo.parent_reconstruction_params[parent_part]

        parent_to_child_part_transform_estimate = infer_relpose(
            demo.interaction_points[parent_part][child_part],
            warp_models["child"][child_part],
            warp_models["parent"][parent_part],
            child_params,
            parent_params,
        )

        constraint_component_pcds[pair] = child_reconstruction
        #     # utils.transform_pcd(
        #     #     child_reconstruction, parent_to_child_part_transform_estimate
        #     # )
        # )
        transform_estimates[pair] = parent_to_child_part_transform_estimate

    # Hacking to work with the warp optimization interface
    combined_pcl = np.concatenate(list(constraint_component_pcds.values()), axis=0)
    # center_transform = utils.pos_quat_to_transform(
    #     np.mean(np.unique(utils.trunc(combined_pcl), axis=0), axis=0),
    #     np.array([0.0, 0.0, 0.0, 1.0]),
    # )

    constraint_shape = utils.ConstraintShape.from_part_reconstructions(
        constraint_component_pcds, transform_estimates
    )

    if viz:
        viz_parent = {part: part_pcds["parent"][part] for part in part_pcds["parent"]}
        fig = viz_utils.show_pcds_plotly(
            {
                "constraint": constraint_shape.canonical_pcl,
            }
            | viz_parent
        )
        fig.show()

    source_pcds = [
        part_pcds["child"][part[0]] for part in [pair for pair in part_pairs]
    ]
    target_pcds = [
        constraint_component_pcds[part] for part in [pair for pair in part_pairs]
    ]

    # These are both arrays of "i for every point in the ith part"
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

    combined_complete, combined_costs, combined_params = warp_to_pcd_se3(
        combined_warp,
        n_angles=15,
        n_batches=12,
        inference_kwargs=ALIGNMENT_ONLY_INFERENCE_KWARGS,
    )

    final_transform = constraint_shape.center_transform @ np.linalg.inv(
        utils.pos_quat_to_transform(combined_params.position, combined_params.quat)
    )

    # record the similarity of the reconstructed pose
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
            }
        )
        fig.show()
    cost = utils.pose_distance(final_transform, demo.child_transform)
    return cost


def get_demo_interaction_points_by_parts(demo, warp_models, viz=False):
    all_part_pairs = itertools.product(demo.parent_part_names, demo.child_part_names)

    # Just for shape warping purposes
    (
        child_relational_labels,
        parent_relational_labels,
        canon_child_relational_labels,
        canon_parent_relational_labels,
    ) = get_relational_labels(
        demo.segmented_initial_pcds,
        warp_models,
        demo.part_relationships,
        include_z=True,
    )

    if viz:
        fig = viz_utils.show_pcds_plotly(
            {
                "child_cup_0": demo.segmented_final_pcds["child"]["cup"][
                    child_relational_labels["cup"][0] == 0
                ],
                "child_cup_1": demo.segmented_final_pcds["child"]["cup"][
                    child_relational_labels["cup"][0] == 1
                ],
                "child_handle_0": demo.segmented_final_pcds["child"]["handle"][
                    child_relational_labels["handle"][0] == 0
                ],
                "child_handle_1": demo.segmented_final_pcds["child"]["handle"][
                    child_relational_labels["handle"][0] == 1
                ],
                "parent_branch_0": demo.segmented_final_pcds["parent"]["branch"][
                    parent_relational_labels["branch"][0] == 0
                ],
                "parent_branch_1": demo.segmented_final_pcds["parent"]["branch"][
                    parent_relational_labels["branch"][0] == 1
                ],
                "parent_trunk_0": demo.segmented_final_pcds["parent"]["trunk"][
                    parent_relational_labels["trunk"][0] == 0
                ],
                "parent_trunk_1": demo.segmented_final_pcds["parent"]["trunk"][
                    parent_relational_labels["trunk"][0] == 1
                ],
            }
        )
        fig.show()

        fig = viz_utils.show_pcds_plotly(
            {
                "canon_cup_0": warp_models["child"]["cup"].canonical_pcl[
                    canon_child_relational_labels["cup"][0] == 0
                ],
                "canon_cup_1": warp_models["child"]["cup"].canonical_pcl[
                    canon_child_relational_labels["cup"][0] == 1
                ],
                "canon_handle_0": warp_models["child"]["handle"].canonical_pcl[
                    canon_child_relational_labels["handle"][0] == 0
                ],
                "canon_handle_1": warp_models["child"]["handle"].canonical_pcl[
                    canon_child_relational_labels["handle"][0] == 1
                ],
                "canon_branch_0": warp_models["parent"]["branch"].canonical_pcl[
                    canon_parent_relational_labels["branch"][0] == 0
                ],
                "canon_branch_1": warp_models["parent"]["branch"].canonical_pcl[
                    canon_parent_relational_labels["branch"][0] == 1
                ],
                "canon_trunk_0": warp_models["parent"]["trunk"].canonical_pcl[
                    canon_parent_relational_labels["trunk"][0] == 0
                ],
                "canon_trunk_1": warp_models["parent"]["trunk"].canonical_pcl[
                    canon_parent_relational_labels["trunk"][0] == 1
                ],
            }
        )
        fig.show()

    # demo_reconstructions = pickle.load(open("demo_reconstructions.pkl", "rb"))
    # demo_reconstruction_params = pickle.load(open("demo_reconstruction_params.pkl", "rb"))
    # demo.reconstruction_params = demo_reconstruction_params
    demo_reconstruction_params = {"child": {}, "parent": {}}
    demo_reconstructions = {"child": {}, "parent": {}}
    for part in demo.parent_part_names:
        part_reconstruction, part_params = reconstruct_part_shape(
            warp_models["parent"][part],
            demo.segmented_initial_pcds["parent"][part],
            parent_relational_labels[part],
            canon_labels=canon_parent_relational_labels[part],
            inference_kwargs=WARPING_INFERENCE_KWARGS,
            warp_mode="SE3",
            viz=True,
        )
        demo_reconstruction_params["parent"][part] = part_params
        demo_reconstructions["parent"][part] = part_reconstruction

    for part in demo.child_part_names:
        part_reconstruction, part_params = reconstruct_part_shape(
            warp_models["child"][part],
            demo.segmented_initial_pcds["child"][part],
            child_relational_labels[part],
            canon_labels=canon_child_relational_labels[part],
            inference_kwargs=WARPING_INFERENCE_KWARGS,
            warp_mode="SE3",
            viz=True,
        )

        part_params = utils.update_reconstruction_params_with_transform(
            part_params, demo.child_transform
        )
        part_reconstruction = warp_models["child"][part].to_transformed_pcd(part_params)

        demo_reconstruction_params["child"][part] = part_params
        demo_reconstructions["child"][part] = part_reconstruction

    # pickle.dump(demo_reconstructions, open("demo_reconstructions.pkl", "wb"))
    # pickle.dump(demo_reconstruction_params, open("demo_reconstruction_params.pkl", "wb"))

    part_pair_interaction_points = {}
    for part_pair in all_part_pairs:
        parent_part, child_part = part_pair

        if parent_part not in part_pair_interaction_points.keys():
            part_pair_interaction_points[parent_part] = {}

        knns, deltas, i_2 = get_interaction_points(
            warp_models["child"][child_part],
            demo_reconstruction_params["child"][child_part],
            warp_models["parent"][parent_part],
            demo_reconstruction_params["parent"][parent_part],
            mesh_vertices_only=False,
        )

        part_pair_interaction_points[parent_part][child_part] = {
            "knns": knns,
            "deltas": deltas,
            "target_indices": i_2,
        }

        warped_part_pair_interaction_points = warp_interaction_points(
            part_pair_interaction_points[parent_part][child_part],
            warp_models["child"][child_part],
            warp_models["parent"][parent_part],
            demo_reconstruction_params["child"][child_part],
            demo_reconstruction_params["parent"][parent_part],
        )

        fig = viz_utils.show_pcds_plotly(
            {
                "child": demo.segmented_final_pcds["child"][child_part],
                "parent": demo.segmented_final_pcds["parent"][parent_part],
                "child_reconstruction": demo_reconstructions["child"][child_part],
                "parent_reconstruction": demo_reconstructions["parent"][parent_part],
                "child_interaction_points": warped_part_pair_interaction_points[
                    "child"
                ],
                "parent_interaction_points": warped_part_pair_interaction_points[
                    "parent"
                ],
            }
        )
        fig.show()

    return part_pair_interaction_points


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
    demo_data = pickle.load(open(demo_path, "rb"))
    demo.child_reconstruction_params = demo_data["child_reconstruction_params"]
    demo.parent_reconstruction_params = demo_data["parent_reconstruction_params"]
    demo.warp_models = warp_models

    test_part_pair = [("handle", "branch")]
    # We evaluate the heuristic by reconstructing the demo
    cost = test_heuristic(
        warp_models,
        demo,
        TestScene(demo.initial_pcds, demo.segmented_initial_pcds),
        test_part_pair,
    )
    print(cost)
