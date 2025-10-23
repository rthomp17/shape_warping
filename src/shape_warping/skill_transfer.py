import os
import os.path as osp
import numpy as np
from shape_warping import utils, viz_utils
import pickle
import itertools

class TestScene():
    max_points_per_object = 2000

    def __init__(self,pcds, segmented_pcds):
        self.pcds = pcds
        self.segmented_pcds = segmented_pcds

        for object_type in ('parent', 'child'):
            self.pcds[object_type], _ = utils.farthest_point_sample(
                                self.pcds[object_type], min(DemoScene.max_points_per_object, len(self.pcds[object_type]))
                            )
        for object_type in ('parent', 'child'):
            for part in self.segmented_pcds[object_type].keys():
                self.segmented_pcds[object_type][part], _ = utils.farthest_point_sample(
                                self.segmented_pcds[object_type][part], min(DemoScene.max_points_per_object, len(self.segmented_pcds[object_type][part]))
                            )
            

        self.get_part_relationships()

    def get_part_relationships(self, num_close_threshold=20, distance_threshold=0.05):
        #Generate all possible pairs of parts
        all_child_part_pairs = itertools.combinations(self.segmented_pcds['child'].keys(), r=2)
        all_parent_part_pairs = itertools.combinations(self.segmented_pcds['parent'].keys(), r=2)

        self.child_part_relationships = []
        for pair in all_child_part_pairs:
            #Get the distance between closest points on the two objects
            dist = utils.nearest_neighbor_distances(self.segmented_pcds['child'][pair[0]], self.segmented_pcds['child'][pair[1]])
            close_points = dist < distance_threshold
            if np.sum(close_points) > num_close_threshold:
                self.child_part_relationships.append(pair)

        self.parent_part_relationships = []
        for pair in all_parent_part_pairs:
            #Get the distance between closest points on the two objects
            dist = utils.nearest_neighbor_distances(self.segmented_pcds['parent'][pair[0]], self.segmented_pcds['parent'][pair[1]])
            close_points = dist < distance_threshold
            if np.sum(close_points) > num_close_threshold:
                self.parent_part_relationships.append(pair)

    @staticmethod
    def from_pickle(path):
        with open(path, 'rb') as f:
            test_scene = pickle.load(f)
        
        pcds = {
            'parent': test_scene["parent_pcd"].item()['parent'],
            'child': test_scene["child_pcd"].item()['child']
        }
        segmented_pcds = {
            'parent': {part: test_scene["segmented_pcd"].item()['parent'][part] for part in test_scene["segmented_pcd"].item()['parent']}, 
            'child': {part: test_scene["segmented_pcd"].item()['child'][part] for part in test_scene["segmented_pcd"].item()['child']}
        }
        return TestScene(pcds, segmented_pcds)

class DemoScene():
    max_points_per_object = 2000

    def __init__(self,
                 initial_pcds,
                 segmented_initial_pcds,
                 final_pcds,
                 segmented_final_pcds,
                 start_poses,
                 final_poses, 
                 interaction_points = None):
        
        self.initial_pcds = initial_pcds
        self.segmented_initial_pcds = segmented_initial_pcds
        self.final_pcds = final_pcds
        self.segmented_final_pcds = segmented_final_pcds
        self.start_poses = start_poses
        self.final_poses = final_poses
        self.child_part_names = sorted(list(self.segmented_final_pcds['child'].keys()))
        self.parent_part_names = sorted(list(self.segmented_final_pcds['parent'].keys()))
    
        self.parent_transform = np.matmul(
                        utils.pos_quat_to_transform(self.final_poses['parent'][:3], self.final_poses['parent'][3:]),
                        np.linalg.inv(utils.pos_quat_to_transform(self.start_poses['parent'][:3], self.start_poses['parent'][3:])))
        self.child_transform = np.matmul(
                        utils.pos_quat_to_transform(self.final_poses['child'][:3], self.final_poses['child'][3:]),
                        np.linalg.inv(utils.pos_quat_to_transform(self.start_poses['child'][:3], self.start_poses['child'][3:])))
        
        self.interaction_points = interaction_points
        self.get_part_relationships()
        self.part_relationships = {'parent': self.parent_part_relationships, 'child': self.child_part_relationships}


    def show_demo(self):
         # Just to make visualization easier
        good_demo_camera_view = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=.5, y=.5, z=.5),
            eye=dict(x=-1, y=-1, z=.5)
        )

        # Visualize the skill demonstration
        demo_fig = viz_utils.show_pcds_plotly({
                                    "Parent Object Initial Pose": self.initial_pcds['parent'],
                                    "Child Object Initial Pose": self.initial_pcds['child'],
                                    "Parent Object End Pose": self.final_pcds['parent'],
                                    "Child Object End Pose": self.final_pcds['child']}, 
                                    camera=good_demo_camera_view)

        demo_fig.show()

    def get_part_relationships(self, num_close_threshold=20, distance_threshold=0.05):
        #Generate all possible pairs of parts
        all_child_part_pairs = itertools.combinations(self.segmented_initial_pcds['child'].keys(), r=2)
        all_parent_part_pairs = itertools.combinations(self.segmented_initial_pcds['parent'].keys(), r=2)

        self.child_part_relationships = []
        for pair in all_child_part_pairs:
            #Get the distance between closest points on the two objects
            dist = utils.nearest_neighbor_distances(self.segmented_initial_pcds['child'][pair[0]], self.segmented_initial_pcds['child'][pair[1]])
            close_points = dist < distance_threshold
            if np.sum(close_points) > num_close_threshold:
                self.child_part_relationships.append(pair)

        self.parent_part_relationships = []
        for pair in all_parent_part_pairs:
            #Get the distance between closest points on the two objects
            dist = utils.nearest_neighbor_distances(self.segmented_initial_pcds['parent'][pair[0]], self.segmented_initial_pcds['parent'][pair[1]])
            close_points = dist < distance_threshold
            if np.sum(close_points) > num_close_threshold:
                self.parent_part_relationships.append(pair)


    @staticmethod
    def from_pickle(path):
        with open(path, 'rb') as f:
            demo = pickle.load(f)

        initial_pcds = {
        'parent': demo["start_pcds"]['parent'],
        'child': demo["start_pcds"]['child']
        }

        segmented_initial_pcds = {
            'parent': {part: demo["segmented_start_pcds"]['parent'][part] for part in demo["segmented_start_pcds"]['parent']}, 
            'child': {part: demo["segmented_start_pcds"]['child'][part] for part in demo["segmented_start_pcds"]['child']}
        }

        segmented_final_pcds = {
            'parent': {part: demo["segmented_final_pcds"]['parent'][part] for part in demo["segmented_final_pcds"]['parent']}, 
            'child': {part: demo["segmented_final_pcds"]['child'][part] for part in demo["segmented_final_pcds"]['child']}
        }

        start_poses = {
        'parent': demo["start_object_poses"]['parent'],
        'child': demo["start_object_poses"]['child']
        }

        final_pcds = {
        'parent': demo["final_pcds"]['parent'],
        'child': demo["final_pcds"]['child']
        }

        final_poses = {
        'parent': demo["final_object_poses"]['parent'],
        'child': demo["final_object_poses"]['child']
        }

         # Subsamples points for faster inference/to prevent CUDA from running out of memory
        for object_type in ('parent', 'child'):
            initial_pcds[object_type], _ = utils.farthest_point_sample(
                                initial_pcds[object_type], DemoScene.max_points_per_object
                            )
            final_pcds[object_type], _ = utils.farthest_point_sample(
                                final_pcds[object_type], DemoScene.max_points_per_object
                            )
            
        for object_type in ('parent', 'child'):
            for part in segmented_initial_pcds[object_type].keys():
                segmented_initial_pcds[object_type][part], _ = utils.farthest_point_sample(
                                segmented_initial_pcds[object_type][part], min(DemoScene.max_points_per_object, len(segmented_initial_pcds[object_type][part]))
                            )
                segmented_final_pcds[object_type][part], _ = utils.farthest_point_sample(
                                segmented_final_pcds[object_type][part], min(DemoScene.max_points_per_object, len(segmented_final_pcds[object_type][part]))
                            )   
        if 'interaction_points' in demo.keys():
            interaction_points = demo["interaction_points"]
        else:
            interaction_points = None
        return DemoScene(initial_pcds, segmented_initial_pcds, final_pcds, segmented_final_pcds, start_poses, final_poses, interaction_points)

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

def get_interaction_points(canon_source_obj,
                           source_obj_param, canon_target_obj,
                           target_obj_param, delta=0.055,
                           mesh_vertices_only=False):

    source_pcd = canon_source_obj.to_transformed_pcd(source_obj_param)
    target_pcd = canon_target_obj.to_transformed_pcd(target_obj_param)

    if mesh_vertices_only:
        source_pcd=source_pcd[:len(canon_source_obj.mesh_vertices)]
        target_pcd=target_pcd[:len(canon_target_obj.mesh_vertices)]

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

    if mesh_vertices_only:
        full_source_pcd=full_source_pcd[:len(canon_source_obj.mesh_vertices)]
        full_target_pcd=full_target_pcd[:len(canon_target_obj.mesh_vertices)]

    knns, deltas = get_knn_and_deltas(full_source_pcd, pos_target_source_coords)

    dist_2 = np.sqrt(np.sum(np.square(full_target_pcd[:, None] - pos_target_target_coords[None]), axis=2))
    i_2 = np.argmin(dist_2, axis=0).transpose()

    return knns, deltas, i_2

def construct_part_alignment_objective(alignment_constraint_pcd, source_pcd_parts):
    constraint_component_pcds = alignment_constraint_pcd.component_pcds
    if len(constraint_component_pcds) == 1:
            combined_canon_part_labels = np.zeros(component_pcls[0].shape[0])
    else:
        combined_canon_part_labels = np.array(
            list(
                itertools.chain(
                    *[
                        [i for _ in range(constraint_component_pcds[i].shape[0])]
                        for i in range(len(constraint_component_pcds))
                    ]
                )
            )
        )

    cost_function = (
                    lambda source, target, canon_part_labels: 
                    mask_and_cost_batch_pt(
                        source,
                        canon_part_labels,
                        target,
                        [source_labels[source_downsampled_indices]], ))




def mesh_edit_interaction_points(interaction_points, edited_child_mesh, edited_parent_mesh, child_pose, parent_pose):
    knns = interaction_points['knns']
    deltas = interaction_points['deltas']
    target_indices = interaction_points['target_indices']

    # Warping the interaction points to the object shape
    anchors = edited_child_mesh.vertices[
        knns
    ]
    targets_child = np.mean(
        anchors + deltas, axis=1
    )
    targets_parent = edited_parent_mesh.vertices[
        target_indices
    ] 

    child_transform  = child_pose
    parent_transform  = parent_pose

    child_targets = utils.transform_pcd(targets_child,
                                        child_transform)

    parent_targets = utils.transform_pcd(targets_parent,
                                         parent_transform)

    return {'parent':parent_targets, 'child': child_targets}


def warp_interaction_points(interaction_points, child_model, parent_model, child_reconstruction_params, parent_reconstruction_params):
    knns = interaction_points['knns']
    deltas = interaction_points['deltas']
    target_indices = interaction_points['target_indices']

    # Warping the interaction points to the object shape
    anchors = child_model.to_pcd(child_reconstruction_params)[
        knns
    ]
    targets_parent = np.mean(
        anchors + deltas, axis=1
    )
    targets_child = parent_model.to_pcd(parent_reconstruction_params)[
        target_indices
    ] 

    child_transform  = utils.pos_quat_to_transform(child_reconstruction_params.position, 
                                                   child_reconstruction_params.quat)
    parent_transform  = utils.pos_quat_to_transform(parent_reconstruction_params.position, 
                                                    parent_reconstruction_params.quat)

    child_targets = utils.transform_pcd(targets_child,
                                        parent_transform)

    parent_targets = utils.transform_pcd(targets_parent,
                                         child_transform)

    return {'parent':parent_targets, 'child': child_targets}


#TODO: I think there may be a mismatch in the child and parent here from the original interaction warping code
def mesh_edit_infer_relpose(interaction_points,
                            edited_child_mesh, 
                            edited_parent_mesh,
                            child_mesh_pose,
                            parent_mesh_pose):
    
    knns = interaction_points['knns']
    deltas = interaction_points['deltas']
    target_indices = interaction_points['target_indices']

    # Warping the interaction points to the object shape
    anchors = edited_child_mesh.vertices[
        knns
    ]
    targets_child = np.mean(
        anchors + deltas, axis=1
    )
    targets_parent = edited_parent_mesh.vertices[
        target_indices
    ] 
        # Canonical source obj to canonical target obj.
    trans_cs_to_ct, _, _ = utils.best_fit_transform(targets_child, targets_parent)

    trans_s_to_b = child_mesh_pose
    trans_t_to_b = parent_mesh_pose

    # Compute relative transform.
    trans_s_to_t = trans_t_to_b @ trans_cs_to_ct @ np.linalg.inv(trans_s_to_b)
    return trans_s_to_t

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
    raise NotImplementedError()




"""
Mesh loading helper files
TODO: Move somewhere more appropriate
"""
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

