import os
import os.path as osp
import numpy as np
from shape_warping import utils



def update_reconstruction_params_with_transform(demo_child_params, demo_transform):

    child_transform = utils.pos_quat_to_transform(demo_child_params.position, demo_child_params.quat)
    new_child_transform = np.matmul(demo_transform, child_transform)
    demo_child_params.position, demo_child_params.quat = utils.transform_to_pos_quat(new_child_transform)

    return demo_child_params


def get_knn_and_deltas(obj, vps, k=10, show=False):
    """Anchor virtual points on an object using k-nearest-neighbors."""
    if show:
        viz_utils.show_pcds_pyplot({
            "obj": obj,
            "vps": vps
        })

    dists = np.sum(np.square(obj[None] - vps[:, None]), axis=-1)
    knn_list = []
    deltas_list = []

    for i in range(dists.shape[0]):
        # Get K closest points, compute relative vectors.
        knn = np.argpartition(dists[i], k)[:k]
        deltas = vps[i: i + 1] - obj[knn]
        knn_list.append(knn)
        deltas_list.append(deltas)

    knn_list = np.stack(knn_list)
    deltas_list = np.stack(deltas_list)
    return knn_list, deltas_list

def get_interaction_points(source, target, canon_source_obj,
                           source_obj_param, canon_target_obj,
                           target_obj_param, delta=0.055,):

    source_pcd = canon_source_obj.to_transformed_pcd(source_obj_param)
    target_pcd = canon_target_obj.to_transformed_pcd(target_obj_param)
            
    # viz_utils.show_pcds_plotly({
    #     "pcd": source_pcd,
    #     "warp": target_pcd
    # }, center=False)

    dist = np.sqrt(np.sum(np.square(source_pcd[:, None] - target_pcd[None]), axis=-1))
    print("@@", np.min(dist))
    indices = np.where(dist <= delta)
    pos_source = source_pcd[indices[0]]
    pos_target = target_pcd[indices[1]]

    assert len(pos_source) > 0, "No nearby points in demonstration."
    print("# nearby points:", len(pos_source))
    if len(pos_source) < 10:
        print("WARNING: Too few nearby points.")

    max_pairs = 100
    if len(pos_source) > max_pairs:
        pos_source, indices2 = utils.farthest_point_sample(pos_source, max_pairs)
        pos_target = pos_target[indices2]

    # Points on target in canonical target object coordinates.
    pos_target_target_coords = utils.transform_pcd(pos_target, np.linalg.inv(target_obj_param.get_transform()))
    # Points on target in canonical source object coordinates.
    pos_target_source_coords = utils.transform_pcd(pos_target, np.linalg.inv(source_obj_param.get_transform()))

    full_source_pcd = canon_source_obj.to_pcd(source_obj_param)
    full_target_pcd = canon_target_obj.to_pcd(target_obj_param)

    knns, deltas = get_knn_and_deltas(full_source_pcd, pos_target_source_coords)

    dist_2 = np.sqrt(np.sum(np.square(full_target_pcd[:, None] - pos_target_target_coords[None]), axis=2))
    i_2 = np.argmin(dist_2, axis=0).transpose()

    return knns, deltas, i_2


def warp_interaction_points(interaction_points, child_model, parent_model, child_reconstruction_params, parent_reconstruction_params):
    knns = interaction_points['knns']
    deltas = interaction_points['deltas']
    target_indices = interaction_points['target_indices']

    # Warping the interaction points to the object shape
    anchors = child_model.to_pcd(child_reconstruction_params)[
        knns
    ]
    targets_child = np.mean(
        anchors + deltas, axis=1
    )
    targets_parent = parent_model.to_pcd(parent_reconstruction_params)[
        target_indices
    ] 

    child_transform  = utils.pos_quat_to_transform(child_reconstruction_params.position, 
                                                   child_reconstruction_params.quat)
    parent_transform  = utils.pos_quat_to_transform(parent_reconstruction_params.position, 
                                                    parent_reconstruction_params.quat)

    child_targets = utils.transform_pcd(targets_child,
                                        child_transform)

    parent_targets = utils.transform_pcd(targets_parent,
                                         parent_transform)

    return {'parent':parent_targets, 'child': child_targets}


def infer_relpose(
        interaction_points,
        child_model, 
        parent_model,
        child_reconstruction_params, 
        parent_reconstruction_params,
    ):
    
    knns = interaction_points['knns']
    deltas = interaction_points['deltas']
    target_indices = interaction_points['target_indices']

    # Warping the interaction points to the object shape
    anchors = child_model.to_pcd(child_reconstruction_params)[
        knns
    ]
    targets_child = np.mean(
        anchors + deltas, axis=1
    )
    targets_parent = parent_model.to_pcd(parent_reconstruction_params)[
        target_indices
    ] 

    # Canonical source obj to canonical target obj.
    trans_cs_to_ct, _, _ = utils.best_fit_transform(targets_child, targets_parent)

    trans_s_to_b = utils.pos_quat_to_transform(
        child_reconstruction_params.position, child_reconstruction_params.quat
    )

    trans_t_to_b = utils.pos_quat_to_transform(
        parent_reconstruction_params.position, parent_reconstruction_params.quat
    )
    # Compute relative transform.
    trans_s_to_t = trans_t_to_b @ trans_cs_to_ct @ np.linalg.inv(trans_s_to_b)

    # if final_alignment:
    #     #transform the source pcd by s_to_t
    #     #use that to make the canon constraint

    # TODO move this data saving behind a flag or generally elsewhere
    # if experiment_id is not None:
    #     best_idx = np.argmin(warp.cost_history[-1])
    #     best_transform_history = []
    #     best_transforms = []
    #     step_names = []
    #     for transform, cost in zip(warp.transform_history, warp.cost_history):
    #         best_trans = transform[best_idx]
    #         best_transforms.append(best_trans)
    #         best_transform_history.append(
    #             utils.transform_pcd(self.canon_target.canonical_pcl, best_trans)
    #         )
    #         step_names.append(f"COST: {cost[best_idx]}")

        # viz_utils.show_pcds_video_animation_plotly(
        #     moving_pcl_name='Constraint Transformation',
        #     moving_pcl_frames=best_transform_history,
        #     static_pcls={"Start Pointcloud": source_downsampled},
        #     step_names=step_names,
        #     file_name = experiment_id,
        # )

        # slider_fig = viz_utils.show_pcds_slider_animation_plotly(
        #     moving_pcl_name="Constraint Transformation",
        #     moving_pcl_frames=best_transform_history,
        #     static_pcls={"Start Pointcloud": target_pcd},
        #     step_names=step_names,
        # )
        # slider_fig.show()
        # pickle.dump(slider_fig, open(experiment_id + "_slider_fig.pkl", "wb"))

        # result_fig = viz_utils.show_pcds_plotly(
        #     {
        #         "child pcl": source_pcd,
        #         "warped_child": source_pcd_complete,
        #         "transformed child": utils.transform_pcd(source_pcd, trans_s_to_t),
        #         #'t2_trans_pcd':utils.transform_pcd(source_pcd, np.linalg.inv(utils.pos_quat_to_transform(combined_params.position, combined_params.quat))),
        #         "warped_parent": self.canon_target.to_transformed_pcd(target_param),
        #         "parent pcl": target_pcd,
        #     }
        # )
        # result_fig.show()
        # pickle.dump(
        #     result_fig, open(experiment_id + "_final_transform_fig.pkl", "wb")
        # )

    trans_s_to_t = trans_t_to_b @ trans_cs_to_ct @ np.linalg.inv(trans_s_to_b)
    return trans_s_to_t

def mesh_transform_interaction_points(interaction_points, child_mesh_vertices, parent_mesh_vertices):
    print("Not yet implemented")
    pass

def get_mug_mesh(pcl_id):
    obj_file_path = f"example_data/mug_meshes/{pcl_id}/models/model_normalized.obj"
    mesh = utils.trimesh_load_object(obj_file_path)
    return mesh

def get_rack_mesh(pcl_id):
    mesh = utils.trimesh_load_object("example_data/rack_meshes/" + f"{pcl_id}")
    return mesh

def load_all_shapenet_files(obj_type):
    mesh_data_dirs = {
        "mug": "mug_meshes",
        # 'bottle': 'bottle_centered_obj_normalized',
        #"bowl": "bowl_centered_obj_normalized",
        "syn_rack_easy": "rack_meshes",
        # 'syn_container': 'box_containers_unnormalized'
    }
    mesh_data_dirs = {
        k: osp.join('example_data', v)
        for k, v in mesh_data_dirs.items()
    }
    # bad_ids = {
    #     "syn_rack_easy": [],
    #     "bowl": bad_shapenet_bowls_ids_list,
    #     "mug": bad_shapenet_mug_ids_list,
    #     "bottle": bad_shapenet_bottles_ids_list,
    #     "syn_container": [],
    # }

    mesh_names = {}
    for k, v in mesh_data_dirs.items():
        # get train samples
        objects_raw = os.listdir(v)
        objects_filtered = [
            fn
            for fn in objects_raw
            if "_dec" not in fn
        ]
        # objects_filtered = objects_raw
        # total_filtered = len(objects_filtered)
        # train_n = int(total_filtered * 0.9)
        # test_n = total_filtered - train_n

        # train_objects = sorted(objects_filtered)[:train_n]
        # test_objects = sorted(objects_filtered)[train_n:]

        # log_info('\n\n\nTest objects: ')
        # log_info(test_objects)
        # # log_info('\n\n\n')

        mesh_names[k] = objects_filtered

    # obj_classes = list(mesh_names.keys())

    # scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
    # scale_default = cfg.MESH_SCALE_DEFAULT

    # # cfg.OBJ_SAMPLE_Y_HIGH_LOW = [0.3, -0.3]
    # cfg.OBJ_SAMPLE_Y_HIGH_LOW = [-0.35, 0.175]
    # x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
    # y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
    # table_z = cfg.TABLE_Z

    return mesh_names[obj_type]

