from shape_warping import utils, viz_utils
import trimesh
import numpy as np
import copy as cp


def learn_warps(meshes, n_dimensions, num_surface_samples=10000):
    # These are for tracking the mesh vertices and faces vs sampled surface points
    # Which is what lets us model warping the canonical mesh
    small_surface_points = []
    surface_points = []
    mesh_points = []
    hybrid_points = []
    faces = []
    centers = []

    # Sampling a uniform pcl for each mesh
    for mesh in meshes:
        translation_matrix = np.eye(4)
        t = mesh.centroid
        sp = utils.trimesh_create_verts_surface(
            mesh, num_surface_samples=num_surface_samples
        )
        ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
        mp, f = utils.trimesh_get_vertices_and_faces(mesh)
        ssp, sp, mp, t = utils.scale_points_circle(
            [ssp, sp, mp, np.atleast_2d(t)], base_scale=0.1
        )
        h = np.concatenate([mp, sp])  # Order important!
        translation_matrix[:3, 3] = t.squeeze()

        centers.append(t)
        small_surface_points.append(ssp)
        surface_points.append(sp)
        mesh_points.append(mp)
        faces.append(f)
        hybrid_points.append(h)

    # Picks the object to be the base for warping
    canonical_idx = utils.sst_pick_canonical(hybrid_points)
    print(f"Canonical obj index: {canonical_idx}.")

    tmp_obj_points = cp.copy(small_surface_points)
    tmp_obj_points[canonical_idx] = hybrid_points[canonical_idx]

    # Warping
    warps, _ = utils.warp_gen(canonical_idx, tmp_obj_points, alpha=0.01, visualize=True)

    # Learning the PCA encoding
    _, pca = utils.pca_transform(warps, n_dimensions=n_dimensions)

    warp_results = {
        "pca": pca,
        "canonical_idx": canonical_idx,
        "canonical_pcl": hybrid_points[canonical_idx],
        "canonical_mesh_points": mesh_points[canonical_idx],
        "canonical_mesh_faces": faces[canonical_idx],
        "canonical_center_transform": centers[canonical_idx],
    }

    return warp_results
