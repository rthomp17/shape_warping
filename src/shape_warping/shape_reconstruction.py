# Code by Ondrej Biza and Skye Thompson

from ast import Call
from typing import List, Optional, Tuple, Union, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import torch
from torch import nn, optim

from shape_warping import utils, viz_utils

PARAM_1 = {"lr": 1e-2, "n_steps": 150, "n_samples": 1000, "object_size_reg": 0.01}


def cost_batch_pt(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the one-sided Chamfer distance between two batches of point clouds in pytorch."""
    # B x N x K
    diff = torch.sqrt(
        torch.sum(torch.square(source[:, :, None] - target[:, None, :]), dim=3
        )
    )
    diff_flat = diff.view(diff.shape[0] * diff.shape[1], diff.shape[2])
    c_flat = diff_flat[list(range(len(diff_flat))), torch.argmin(diff_flat, dim=1)]
    c = c_flat.view(diff.shape[0], diff.shape[1])
    return torch.mean(c, dim=1)

# Mask by the provided labels. Only cost between shared labels counts
def mask_and_cost_batch_pt(target, source_labels, source, target_labels, switch_sides=False):
    summed_cost = None
    weights = [1,1]

    assert len(source_labels) == len(target_labels)
    for source_label, target_label in zip(source_labels, target_labels):
        for label, w in zip(np.unique(source_label), weights):
            
            if switch_sides:
                part_cost = cost_batch_pt(target[:, torch.from_numpy(target_label)==label], source[:, torch.from_numpy(source_label)==label], )
            else:
                part_cost = cost_batch_pt(source[:, torch.from_numpy(source_label)==label], target[:, torch.from_numpy(target_label)==label])
            if summed_cost is None:
                summed_cost = part_cost * w
            else:
                summed_cost += part_cost * w

    return summed_cost

class ObjectWarping:
    """Base class for inference of object shape, pose and scale with gradient descent."""

    def __init__(
        self,
        canon_obj: utils.CanonShape,
        pcd: NDArray[np.float32],
        device: "cpu",
        lr: float,
        n_steps: int,
        cost_function: Callable = cost_batch_pt,
        n_samples: Optional[int] = None,
        canon_labels: Optional[List[int]] = None,
        object_size_reg: Optional[float] = None,
        init_scale: float = 1.0,
    ):
        """Generic init functions that save the canonical object and the observed point cloud."""
        self.device = device
        self.pca = canon_obj.pca
        self.lr = lr
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.object_size_reg = object_size_reg
        self.init_scale = init_scale
        self.cost_function = cost_function
        self.transform_history = None
        self.cost_history = None
        self.final_cost = None
        self.canon_labels = canon_labels

        # This change is to eliminate some of the issues with outliers/doubled points skewing the mean
        # in the future, should probably be addressed some other, better way
        # either by looking at improving warps, improving sampling, or by improving the transform fit
        def trunc(values, decs=2):
            return np.trunc(values * 10**decs) / (10**decs)

        self.global_means = np.mean(np.unique(trunc(pcd), axis=0), axis=0)
        pcd = pcd - self.global_means[None]
        self.canonical_pcl = torch.tensor(
            canon_obj.canonical_pcl, dtype=torch.float32, device=self.device
        )
        self.pcd = torch.tensor(pcd, dtype=torch.float32, device=self.device)

        if canon_obj.pca is not None:
            self.means = torch.tensor(
                canon_obj.pca.mean_, dtype=torch.float32, device=self.device
            )
            self.components = torch.tensor(
                canon_obj.pca.components_, dtype=torch.float32, device=self.device
            )
        else:
            self.means = None
            self.components = None

    def initialize_params_and_opt(
        self,
        initial_poses: NDArray[np.float32],
        initial_centers: Optional[NDArray[np.float32]] = None,
        initial_latents: Optional[NDArray[np.float32]] = None,
        initial_scales: Optional[NDArray[np.float32]] = None,
        train_latents: bool = True,
        train_centers: bool = True,
        train_poses: bool = True,
        train_scales: bool = True,
    ):
        """Initialize parameters to be optimized and the optimizer."""
        raise NotImplementedError()

    def create_warped_transformed_pcd(
        self,
        components: torch.Tensor,
        means: torch.Tensor,
        canonical_pcl: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Warp and transform canonical object."""
        raise NotImplementedError()

    def assemble_output(
        self, cost: torch.Tensor
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:
        """Assemble all optimized parameters."""
        raise NotImplementedError()

    def subsample(
        self, num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly subsample the canonical object, including its PCA projection and descriptors."""
        indices = np.random.randint(0, self.components.shape[1] // 3, num_samples)
        means_ = self.means.view(-1, 3)[indices].view(-1)
        components_ = self.components.view(self.components.shape[0], -1, 3)[:, indices]
        components_ = components_.reshape(self.components.shape[0], -1)
        canonical_obj_pt_ = self.canonical_pcl[indices]

        index_order = np.argsort(indices)
        if self.canon_labels is not None:
            self.subsampled_canon_labels = [cl[indices] for cl in self.canon_labels]

        return (
            means_,
            components_,
            canonical_obj_pt_,
        )

    def inference(
        self,
        initial_poses: NDArray[np.float32],
        initial_centers: Optional[NDArray[np.float32]] = None,
        initial_latents: Optional[NDArray[np.float32]] = None,
        initial_scales: Optional[NDArray[np.float32]] = None,
        train_latents: bool = True,
        train_centers: bool = True,
        train_poses: bool = True,
        train_scales: bool = True,
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:
        """Run inference for a batch of initial poses."""
        self.initialize_params_and_opt(
            initial_poses,
            initial_centers,
            initial_latents,
            initial_scales,
            train_latents,
            train_centers,
            train_poses,
            train_scales,
        )

        # Reset transformation and cost histories
        transform_history = []
        cost_history = []
        latent_history = []
        scale_history = []

        for _ in range(self.n_steps):

            if self.n_samples is not None:
                (
                    means_,
                    components_,
                    canonical_pcl_,
                ) = self.subsample(self.n_samples)
            else:
                means_ = self.means
                components_ = self.components
                canonical_pcl_ = self.canonical_pcl

            self.optim.zero_grad()
            new_pcd = self.create_warped_transformed_pcd(
                components_,
                means_,
                canonical_pcl_,
            )

            # Saving optimization history for visualization
            try:
                transform_history.append(
                        (self.center_param.detach().cpu().numpy() + self.global_means,
                        torch.bmm(orthogonalize(self.pose_param), self.initial_poses)
                        .detach()
                        .cpu()
                        .numpy(),
                    ))
            except IndexError:
                transform_history.append(
                        (self.center_param.detach().cpu().numpy() + self.global_means,
                        yaw_to_rot_batch_pt(self.pose_param).detach()
                        .cpu()
                        .numpy()
                        ,)
                    )
            try:
                latent_history.append(self.latent_param.detach().cpu().numpy())
            except AttributeError:
                latent_history.append(None)
            scale_history.append(self.scale_param.detach().cpu().numpy())

            if self.cost_function == cost_batch_pt:
                cost = self.cost_function(self.pcd[None], new_pcd)
            else:
                # Different procedures for pure pose optimization versus warping
                # TODO Move to classes or kwarg format to clean up
                if not hasattr(self, 'latent_param'):
                    latent_param, initial_latents = None, None
                    if self.n_samples is None:
                        cost = self.cost_function(self.pcd[None], new_pcd, self.canon_labels,)
                    else:
                        cost = self.cost_function(self.pcd[None], new_pcd, self.subsampled_canon_labels, )
                else:
                    latent_param, initial_latents = self.latent_param.data, self.initial_latents_pt
                    if self.n_samples is None:
                        cost = self.cost_function(self.pcd[None], new_pcd, self.canon_labels, latent_param, self.scale_param.data, initial_latents)
                    else:
                        cost = self.cost_function(self.pcd[None], new_pcd, self.subsampled_canon_labels, latent_param, self.scale_param.data, initial_latents)
            
            #Prevents local minima from warping objects to be giant (consequence of the one-sided chamfer metric)
            #TODO move this out to a more generalized cost function structure
            if self.object_size_reg is not None:
                size = torch.max(
                    torch.sqrt(torch.sum(torch.square(new_pcd), dim=-1)), dim=-1
                )[0]
                cost = cost + self.object_size_reg * size
            cost.sum().backward()

            # Saving cost history for visualization
            cost_history.append(cost.detach().cpu().numpy())
            self.optim.step()

        if self.cost_history is None:
            self.cost_history = cost_history
            self.transform_history = utils.transform_history_to_mat(transform_history)
            self.latent_history = latent_history
            self.scale_history = scale_history
        else: 
            self.cost_history = np.concatenate([self.cost_history, cost_history], axis=1,)
            self.transform_history = np.concatenate([self.transform_history, utils.transform_history_to_mat(transform_history)], axis=1, dtype=object)
            try:
                self.latent_history = np.concatenate([self.latent_history, latent_history], axis=1,)
            except np.exceptions.AxisError:
                self.latent_history = latent_history
            self.scale_history  = np.concatenate([self.scale_history, scale_history], axis=1,)

        return self.assemble_output(cost)


class ObjectWarpingSE3Batch(ObjectWarping):
    """Object shape and pose warping in SE3."""

    def initialize_params_and_opt(
        self,
        initial_poses: NDArray[np.float32],
        initial_centers: Optional[NDArray[np.float32]] = None,
        initial_latents: Optional[NDArray[np.float32]] = None,
        initial_scales: Optional[NDArray[np.float32]] = None,
        train_latents: bool = True,
        train_centers: bool = True,
        train_poses: bool = True,
        train_scales: bool = True,
    ):
        n_angles = len(initial_poses)

        # Initial rotation matrices.
        self.initial_poses = torch.tensor(
            initial_poses, dtype=torch.float32, device=self.device
        )

        # This 2x3 vectors will turn into an identity rotation matrix.
        unit_ortho = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        unit_ortho = np.repeat(unit_ortho[None], n_angles, axis=0)
        init_ortho_pt = torch.tensor(
            unit_ortho, dtype=torch.float32, device=self.device
        )

        if initial_centers is None:
            initial_centers_pt = torch.zeros(
                (n_angles, 3), dtype=torch.float32, device=self.device
            )
        else:
            initial_centers_pt = torch.tensor(
                initial_centers - self.global_means[None],
                dtype=torch.float32,
                device=self.device,
            )

        if initial_latents is None:
            initial_latents_pt = torch.from_numpy(self.pca.components_ @ (-self.pca.mean_)).float().to(self.device).repeat(n_angles, 1)
            self.initial_latents_pt = initial_latents_pt
        else:
            initial_latents_pt = torch.tensor(
                initial_latents, dtype=torch.float32, device=self.device
            )
            self.initial_latents_pt = initial_latents_pt

        if initial_scales is None:
            initial_scales_pt = (
                torch.ones((n_angles, 3), dtype=torch.float32, device=self.device)
                * self.init_scale
            )
        else:
            initial_scales_pt = torch.tensor(
                initial_scales, dtype=torch.float32, device=self.device
            )

        self.latent_param = nn.Parameter(initial_latents_pt, requires_grad=True)
        self.center_param = nn.Parameter(initial_centers_pt, requires_grad=True)
        self.pose_param = nn.Parameter(init_ortho_pt, requires_grad=True)
        self.scale_param = nn.Parameter(initial_scales_pt, requires_grad=True)

        self.params = []
        if train_latents:
            self.params.append(self.latent_param)
        if train_centers:
            self.params.append(self.center_param)
        if train_poses:
            self.params.append(self.pose_param)
        if train_scales:
            self.params.append(self.scale_param)

        self.optim = optim.Adam(self.params, lr=self.lr)

    def create_warped_transformed_pcd(
        self, components: torch.Tensor, means: torch.Tensor, canonical_pcl: torch.Tensor
    ) -> torch.Tensor:
        """Warp and transform canonical object. Differentiable."""
        rotm = orthogonalize(self.pose_param)
        rotm = torch.bmm(rotm, self.initial_poses)
        deltas = torch.matmul(self.latent_param, components) + means
        deltas = deltas.view((self.latent_param.shape[0], -1, 3))

        new_pcd = canonical_pcl[None] + deltas
        new_pcd = new_pcd * self.scale_param[:, None]
        new_pcd = (
            torch.bmm(new_pcd, rotm.permute((0, 2, 1))) + self.center_param[:, None]
        )
        return new_pcd

    def assemble_output(
        self, cost: torch.Tensor
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:
        """Output numpy arrays."""
        all_costs = []
        all_new_pcds = []
        all_parameters = []

        with torch.no_grad():
            new_pcd = self.create_warped_transformed_pcd(
                self.components, self.means, self.canonical_pcl
            )
            rotm = orthogonalize(self.pose_param)
            rotm = torch.bmm(rotm, self.initial_poses)

            new_pcd = new_pcd.cpu().numpy()
            new_pcd = new_pcd + self.global_means[None, None]

            for i in range(len(self.latent_param)):
                all_costs.append(cost[i].item())
                all_new_pcds.append(new_pcd[i])

                position = self.center_param[i].cpu().numpy() + self.global_means
                position = position.astype(np.float64)
                quat = utils.rotm_to_quat(rotm[i].cpu().numpy())
                latent = self.latent_param[i].cpu().numpy()
                scale = self.scale_param[i].cpu().numpy()

                obj_param = utils.ObjParam(position, quat, latent, scale)
                all_parameters.append(obj_param)

        return all_costs, all_new_pcds, all_parameters


class ObjectWarpingSE2Batch(ObjectWarping):
    """Object shape and planar pose warping."""

    def initialize_params_and_opt(
        self,
        initial_poses: NDArray[np.float32],
        initial_centers: Optional[NDArray[np.float32]] = None,
        initial_latents: Optional[NDArray[np.float32]] = None,
        initial_scales: Optional[NDArray[np.float32]] = None,
        train_latents: bool = True,
        train_centers: bool = True,
        train_poses: bool = True,
        train_scales: bool = True,
    ):

        # Initial poses are yaw angles here.
        n_angles = len(initial_poses)
        initial_poses_pt = torch.tensor(
            initial_poses, dtype=torch.float32, device=self.device
        )

        if initial_centers is None:
            initial_centers_pt = torch.zeros(
                (n_angles, 3), dtype=torch.float32, device=self.device
            )
        else:
            initial_centers_pt = torch.tensor(
                initial_centers - self.global_means[None],
                dtype=torch.float32,
                device=self.device,
            )

        if initial_latents is None:
            initial_latents_pt = torch.from_numpy(self.pca.components_ @ (-self.pca.mean_)).float().to(self.device).repeat(n_angles, 1)
            self.initial_latents_pt = initial_latents_pt
            # initial_latents_pt = torch.zeros(
            #     (n_angles, self.pca.n_components),
            #     dtype=torch.float32,
            #     device=self.device,
            # )
        else:
            initial_latents_pt = torch.tensor(
                initial_latents, dtype=torch.float32, device=self.device
            )

        if initial_scales is None:
            initial_scales_pt = (
                torch.ones((n_angles, 3), dtype=torch.float32, device=self.device)
                * self.init_scale
            )
        else:
            initial_scales_pt = torch.tensor(
                initial_scales, dtype=torch.float32, device=self.device
            )

        self.latent_param = nn.Parameter(initial_latents_pt, requires_grad=True)
        self.center_param = nn.Parameter(initial_centers_pt, requires_grad=True)
        self.pose_param = nn.Parameter(initial_poses_pt, requires_grad=True)
        self.scale_param = nn.Parameter(initial_scales_pt, requires_grad=True)

        self.params = []
        if train_latents:
            self.params.append(self.latent_param)
        if train_centers:
            self.params.append(self.center_param)
        if train_poses:
            self.params.append(self.pose_param)
        if train_scales:
            self.params.append(self.scale_param)

        self.optim = optim.Adam(self.params, lr=self.lr)

    def create_warped_transformed_pcd(
        self, components: torch.Tensor, means: torch.Tensor, canonical_pcl: torch.Tensor
    ) -> torch.Tensor:
        """Warp and transform canonical object. Differentiable."""
        rotm = yaw_to_rot_batch_pt(self.pose_param)
        deltas = torch.matmul(self.latent_param, components) + means
        deltas = deltas.view((self.latent_param.shape[0], -1, 3))
        new_pcd = canonical_pcl[None] + deltas
        new_pcd = new_pcd * self.scale_param[:, None]
        new_pcd = (
            torch.bmm(new_pcd, rotm.permute((0, 2, 1))) + self.center_param[:, None]
        )
        return new_pcd

    def assemble_output(
        self, cost: torch.Tensor
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:
        """Output numpy arrays."""
        all_costs = []
        all_new_pcds = []
        all_parameters = []

        with torch.no_grad():
            new_pcd = self.create_warped_transformed_pcd(
                self.components,
                self.means,
                self.canonical_pcl,
            )
            rotm = yaw_to_rot_batch_pt(self.pose_param)

            new_pcd = new_pcd.cpu().numpy()
            new_pcd = new_pcd + self.global_means[None, None]

            for i in range(len(self.latent_param)):
                all_costs.append(cost[i].item())
                all_new_pcds.append(new_pcd[i])

                position = self.center_param[i].cpu().numpy() + self.global_means
                position = position.astype(np.float64)
                quat = utils.rotm_to_quat(rotm[i].cpu().numpy())
                latent = self.latent_param[i].cpu().numpy()
                scale = self.scale_param[i].cpu().numpy()

                obj_param = utils.ObjParam(position, quat, latent, scale)
                all_parameters.append(obj_param)

        return all_costs, all_new_pcds, all_parameters


class ObjectSE3Batch(ObjectWarping):
    """Object pose gradient descent in SE3."""

    def initialize_params_and_opt(
        self,
        initial_poses: NDArray[np.float32],
        initial_centers: Optional[NDArray[np.float32]] = None,
        initial_latents: Optional[NDArray[np.float32]] = None,
        initial_scales: Optional[NDArray[np.float32]] = None,
        train_latents: bool = True,
        train_centers: bool = True,
        train_poses: bool = True,
        train_scales: bool = True,
    ):
        n_angles = len(initial_poses)

        # Initial rotation matrices.
        self.initial_poses = torch.tensor(
            initial_poses, dtype=torch.float32, device=self.device
        )

        # This 2x3 vectors will turn into an identity rotation matrix.
        unit_ortho = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        unit_ortho = np.repeat(unit_ortho[None], n_angles, axis=0)
        init_ortho_pt = torch.tensor(
            unit_ortho, dtype=torch.float32, device=self.device
        )

        if initial_centers is None:
            initial_centers_pt = torch.zeros(
                (n_angles, 3), dtype=torch.float32, device=self.device
            )
        else:
            initial_centers_pt = torch.tensor(
                initial_centers - self.global_means[None],
                dtype=torch.float32,
                device=self.device,
            )

        if initial_scales is None:
            initial_scales_pt = (
                torch.ones((n_angles, 3), dtype=torch.float32, device=self.device)
                * self.init_scale
            )
        else:
            initial_scales_pt = torch.tensor(
                initial_scales, dtype=torch.float32, device=self.device
            )

        self.center_param = nn.Parameter(initial_centers_pt, requires_grad=True)
        self.pose_param = nn.Parameter(init_ortho_pt, requires_grad=True)
        self.scale_param = nn.Parameter(initial_scales_pt, requires_grad=True)

        self.params = []
        if train_centers:
            self.params.append(self.center_param)
        if train_poses:
            self.params.append(self.pose_param)
        if train_scales:
            self.params.append(self.scale_param)

        self.optim = optim.Adam(self.params, lr=self.lr)

    def create_warped_transformed_pcd(
        self,
        components: Optional[torch.Tensor],
        means: Optional[torch.Tensor],
        canonical_pcd: torch.Tensor,
    ) -> torch.Tensor:
        """Transform canonical object. Differentiable."""
        rotm = orthogonalize(self.pose_param)
        rotm = torch.bmm(rotm, self.initial_poses)
        new_pcd = torch.repeat_interleave(
            canonical_pcd[None], len(self.pose_param), dim=0
        )
        new_pcd = new_pcd * self.scale_param[:, None]
        new_pcd = (
            torch.bmm(new_pcd, rotm.permute((0, 2, 1))) + self.center_param[:, None]
        )
        return new_pcd

    def subsample(
        self, num_samples: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Randomly subsample the canonical object, including its PCA projection."""
        indices = np.random.randint(0, self.canonical_pcl.shape[0], num_samples)
        return None, None, self.canonical_pcl[indices]

    def assemble_output(
        self, cost: torch.Tensor
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:
        """Output numpy arrays."""
        all_costs = []
        all_new_pcds = []
        all_parameters = []

        with torch.no_grad():
            new_pcd = self.create_warped_transformed_pcd(None, None, self.canonical_pcl)
            rotm = orthogonalize(self.pose_param)
            rotm = torch.bmm(rotm, self.initial_poses)

            new_pcd = new_pcd.cpu().numpy()
            new_pcd = new_pcd + self.global_means[None, None]

            for i in range(len(self.center_param)):
                all_costs.append(cost[i].item())
                all_new_pcds.append(new_pcd[i])

                position = self.center_param[i].cpu().numpy() + self.global_means
                position = position.astype(np.float64)
                quat = utils.rotm_to_quat(rotm[i].cpu().numpy())
                scale = self.scale_param[i].cpu().numpy()

                obj_param = utils.ObjParam(position, quat, None, scale)

                all_parameters.append(obj_param)

        return all_costs, all_new_pcds, all_parameters

class ObjectSE2Batch(ObjectWarping):
    """Object pose gradient descent in SE2."""

    def initialize_params_and_opt(
        self,
        initial_poses: NDArray[np.float32],
        initial_centers: Optional[NDArray[np.float32]] = None,
        initial_latents: Optional[NDArray[np.float32]] = None,
        initial_scales: Optional[NDArray[np.float32]] = None,
        train_latents: bool = True,
        train_centers: bool = True,
        train_poses: bool = True,
        train_scales: bool = True,
    ):
        n_angles = len(initial_poses)
        initial_poses_pt = torch.tensor(
            initial_poses, dtype=torch.float32, device=self.device
        )

        if initial_centers is None:
            initial_centers_pt = torch.zeros(
                (n_angles, 3), dtype=torch.float32, device=self.device
            )
        else:
            initial_centers_pt = torch.tensor(
                initial_centers - self.global_means[None],
                dtype=torch.float32,
                device=self.device,
            )

        if initial_scales is None:
            initial_scales_pt = (
                torch.ones((n_angles, 3), dtype=torch.float32, device=self.device)
                * self.init_scale
            )
        else:
            initial_scales_pt = torch.tensor(
                initial_scales, dtype=torch.float32, device=self.device
            )

        self.center_param = nn.Parameter(initial_centers_pt, requires_grad=True)
        self.pose_param = nn.Parameter(initial_poses_pt, requires_grad=True)
        self.scale_param = nn.Parameter(initial_scales_pt, requires_grad=True)

        self.params = []
        if train_centers:
            self.params.append(self.center_param)
        if train_poses:
            self.params.append(self.pose_param)
        if train_scales:
            self.params.append(self.scale_param)

        self.optim = optim.Adam(self.params, lr=self.lr)

    def create_warped_transformed_pcd(
        self,
        components: Optional[torch.Tensor],
        means: Optional[torch.Tensor],
        canonical_pcl: torch.Tensor,
    ) -> torch.Tensor:
        """Warp and transform canonical object. Differentiable."""
        rotm = yaw_to_rot_batch_pt(self.pose_param)
        new_pcd = torch.repeat_interleave(
            canonical_pcl[None], len(self.pose_param), dim=0
        )
        new_pcd = new_pcd * self.scale_param[:, None]
        new_pcd = (
            torch.bmm(new_pcd, rotm.permute((0, 2, 1))) + self.center_param[:, None]
        )
        return (new_pcd,)

    def subsample(self, num_samples: int) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """Randomly subsample the canonical object, including its PCA projection."""
        indices = np.random.randint(0, self.canonical_pcl.shape[0], num_samples)
        canonical_obj_pt_ = self.canonical_pcl[indices]
        index_order = np.argsort(indices)
        return (
            None,
            None,
            self.canonical_pcl[indices],
        )

    def assemble_output(
        self, cost: torch.Tensor
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:
        """Output numpy arrays."""
        all_costs = []
        all_new_pcds = []
        all_parameters = []

        with torch.no_grad():
            new_pcd = self.create_warped_transformed_pcd(
                None,
                None,
                self.canonical_pcl,
            )
            rotm = yaw_to_rot_batch_pt(self.pose_param)

            new_pcd = new_pcd.cpu().numpy()
            new_pcd = new_pcd + self.global_means[None, None]

            for i in range(len(self.center_param)):
                all_costs.append(cost[i].item())
                all_new_pcds.append(new_pcd[i])

                position = self.center_param[i].cpu().numpy() + self.global_means
                position = position.astype(np.float64)
                quat = utils.rotm_to_quat(rotm[i].cpu().numpy())
                scale = self.scale_param[i].cpu().numpy()

                obj_param = utils.ObjParam(position, quat, None, scale)

                all_parameters.append(obj_param)

        return all_costs, all_new_pcds, all_parameters


def warp_to_pcd_se3(
    object_warping: Union[ObjectWarpingSE3Batch, ObjectSE3Batch],
    n_angles: int = 50,
    n_batches: int = 3,
    inference_kwargs={},
) -> Tuple[NDArray, float, utils.ObjParam]:
    poses = random_rots(n_angles * n_batches)

    all_costs, all_new_pcds, all_parameters = [], [], []

    for batch_idx in range(n_batches):
        poses_batch = poses[batch_idx * n_angles : (batch_idx + 1) * n_angles]
        batch_costs, batch_new_pcds, batch_parameters = object_warping.inference(
            poses_batch, **inference_kwargs
        )
        all_costs += batch_costs
        all_new_pcds += batch_new_pcds
        all_parameters += batch_parameters

    best_idx = np.argmin(all_costs)
    return all_new_pcds[best_idx], all_costs[best_idx], all_parameters[best_idx]


def warp_to_pcd_se3_hemisphere(
    object_warping: Union[ObjectWarpingSE3Batch, ObjectSE3Batch],
    n_angles: int = 50,
    n_batches: int = 3,
    inference_kwargs={},
) -> Tuple[NDArray, float, utils.ObjParam]:
    poses = random_rots_hemisphere(n_angles * n_batches)

    all_costs, all_new_pcds, all_parameters = [], [], []

    for batch_idx in range(n_batches):
        poses_batch = poses[batch_idx * n_angles : (batch_idx + 1) * n_angles]
        batch_costs, batch_new_pcds, batch_parameters = object_warping.inference(
            poses_batch, **inference_kwargs
        )
        all_costs += batch_costs
        all_new_pcds += batch_new_pcds
        all_parameters += batch_parameters

    best_idx = np.argmin(all_costs)
    return all_new_pcds[best_idx], all_costs[best_idx], all_parameters[best_idx]


def warp_to_pcd_se2(
    object_warping: Union[ObjectWarpingSE2Batch, ObjectSE2Batch],
    n_angles: int = 50,
    n_batches: int = 3,
    inference_kwargs={},
) -> Tuple[NDArray, float, utils.ObjParam]:
    start_angles = []
    for i in range(n_angles * n_batches):
        angle = i * (2 * np.pi / (n_angles * n_batches))
        start_angles.append(angle)
    start_angles = np.array(start_angles, dtype=np.float32)[:, None]

    all_costs, all_new_pcds, all_parameters = [], [], []

    for batch_idx in range(n_batches):
        start_angles_batch = start_angles[
            batch_idx * n_angles : (batch_idx + 1) * n_angles
        ]
        batch_costs, batch_new_pcds, batch_parameters = object_warping.inference(
            start_angles_batch, **inference_kwargs
        )
        all_costs += batch_costs
        all_new_pcds += batch_new_pcds
        all_parameters += batch_parameters

    best_idx = np.argmin(all_costs)
    return all_new_pcds[best_idx], all_costs[best_idx], all_parameters[best_idx]


def warp_to_pcd(
    object_warping: Union[ObjectWarpingSE2Batch, ObjectSE2Batch],
    n_batches: int = 3,
    inference_kwargs={},
) -> Tuple[NDArray, float, utils.ObjParam]:
    all_costs, all_new_pcds, all_parameters = [], [], []

    for batch_idx in range(n_batches):
        batch_costs, batch_new_pcds, batch_parameters = object_warping.inference(
            **inference_kwargs
        )
        all_costs += batch_costs
        all_new_pcds += batch_new_pcds
        all_parameters += batch_parameters

    best_idx = np.argmin(all_costs)
    return all_new_pcds[best_idx], all_costs[best_idx], all_parameters[best_idx]


def orthogonalize(x: torch.Tensor) -> torch.Tensor:
    """
    Produce an orthogonal frame from two vectors
    x: [B, 2, 3]
    """
    # u = torch.zeros([x.shape[0],3,3], dtype=torch.float32, device=x.device)
    u0 = x[:, 0] / torch.norm(x[:, 0], dim=1)[:, None]
    u1 = x[:, 1] - (torch.sum(u0 * x[:, 1], dim=1))[:, None] * u0
    u1 = u1 / torch.norm(u1, dim=1)[:, None]
    u2 = torch.cross(u0, u1, dim=1)
    return torch.stack([u0, u1, u2], dim=1)


def cost_pt(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the one-sided Chamfer distance between two point clouds in pytorch."""
    diff = torch.sqrt(torch.sum(torch.square(source[:, None] - target[None, :]), dim=2))
    c = diff[list(range(len(diff))), torch.argmin(diff, dim=1)]
    return torch.mean(c)


def random_rots(num: int) -> NDArray[np.float64]:
    """Sample random rotation matrices."""
    return Rotation.random(num=num).as_matrix().astype(np.float64)


def random_rots_hemisphere(num: int) -> NDArray[np.float32]:
    """TODO: Double check this is correct."""
    rots = random_rots(num * 10)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    out = np.einsum("bnk,kl->bnl", rots, z[:, None])[:, :, 0]
    mask = out[..., 2] >= 0
    return rots[mask][:num]


def yaw_to_rot_pt(yaw: torch.Tensor) -> torch.Tensor:
    """Yaw angle to a rotation matrix in pytorch."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    t0 = torch.zeros(1, device=c.device)
    t1 = torch.ones(1, device=c.device)

    return torch.stack(
        [torch.cat([c, -s, t0]), torch.cat([s, c, t0]), torch.cat([t0, t0, t1])], dim=0
    )


def yaw_to_rot_batch_pt(yaw: torch.Tensor) -> torch.Tensor:
    """Yaw angle to a batch of rotation matrices in pytorch."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    t0 = torch.zeros((yaw.shape[0], 1), device=c.device)
    t1 = torch.ones((yaw.shape[0], 1), device=c.device)

    return torch.stack(
        [
            torch.cat([c, -s, t0], dim=1),
            torch.cat([s, c, t0], dim=1),
            torch.cat([t0, t0, t1], dim=1),
        ],
        dim=1,
    )
