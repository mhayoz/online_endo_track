import torch
from torch import nn
from src.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from src.utils.general_utils import strip_symmetric, build_scaling_rotation, build_inv_cov, inverse_sigmoid, build_rotation
from src.scene.deformation import ExplicitDeformation, ExplicitSparseDeformation
from src.utils.flow_utils import get_surface_pts
from functools import partial


class GaussianModel(nn.Module):
    def __init__(self, cfg, n_classes=7):
        super().__init__()
        self.cfg = cfg
        self.active_sh_degree = 0
        self.max_sh_degree = 1
        self._xyz = torch.empty(0)
        self._semantics = torch.empty(0)
        if cfg["deform_network"]['model'] == 'sparse':
            self._deformation = ExplicitSparseDeformation(subsample=cfg['deform_network']['subsample'])
        elif cfg["deform_network"]['model'] == 'dense':
            self._deformation = ExplicitDeformation()
        else:
            raise NotImplementedError
        # self.grid = TriPlaneGrid()
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.hooks = None
        self.n_classes = n_classes

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def forward(self, deform=True):
        """
            apply deformation and return 3D Gaussians for rasterization
        """
        if deform:
            xyz, scales, rots = self._deformation(self._xyz, self._scaling, self._rotation, init=self.training)
        else:
            xyz, scales, rots = self._xyz, self._scaling, self._rotation
        scales = self.scaling_activation(scales)
        rots = self.rotation_activation(rots)
        opacity = self.opacity_activation(self._opacity)
        return xyz, scales, rots, opacity, self.get_features, self.get_semantics

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_semantics(self):
        return self._semantics

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def enable_spherical_harmonics(self):
        self.active_sh_degree = 1

    def add_from_pcd(self, rgb, depth, c2w, camera, mask, downsample: int=2, semantics=None):
        semantics = torch.ones((*depth.shape, self.n_classes)).squeeze(0) if semantics is None else semantics.squeeze(0)
        # reproject points to 3D
        H, W, fx, fy, cx, cy = camera.get_params()
        points = get_surface_pts(depth, fx, fy, cx, cy, c2w, 'cuda')
        point_cloud = points[:, ::downsample, ::downsample].reshape(-1, 3).cuda()
        # filter points in already visited areas
        color = RGB2SH(rgb[:, ::downsample, ::downsample].reshape(-1, 3).cuda())
        features = torch.zeros((color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = color
        features[:, 3:, 1:] = 0.0
        semantics = semantics[::downsample, ::downsample].reshape(-1,self.n_classes).cuda()

        dist2 = torch.clamp_min(distCUDA2(point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.6*torch.ones((point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        selected_pts_mask = mask[:, ::downsample, ::downsample].reshape(-1).cuda()

        if not selected_pts_mask.float().mean() > 0.0:
            return 0.0

        self.densification_postfix(point_cloud[selected_pts_mask],
                                   features[:,:,0:1][selected_pts_mask].transpose(1, 2),
                                   features[:,:,1:][selected_pts_mask].transpose(1, 2), opacities[selected_pts_mask],
                                   scales[selected_pts_mask], rots[selected_pts_mask],
                                   semantics[selected_pts_mask])
        return selected_pts_mask.float().mean()

    def create_from_pcd(self, rgb, depth, c2w, camera, tool_mask=None, spatial_lr_scale : float=1.0, downsample: int=2, semantics=None):
        with torch.no_grad():
            tool_mask = torch.ones_like(depth).bool() if tool_mask is None else tool_mask
            semantics = torch.ones((*depth.shape, self.n_classes)).squeeze(0) if semantics is None else semantics.squeeze(0)
            self.spatial_lr_scale = spatial_lr_scale
            # reproject points to 3D
            H, W, fx, fy, cx, cy = camera.get_params()
            points = get_surface_pts(depth, fx, fy, cx, cy, c2w, depth.device).squeeze(0)
            rgb_norm = camera.inverse_splotlight_render(rgb.cuda(), points[...,2].cuda()).squeeze(0)
            fused_point_cloud = points[::downsample, ::downsample][tool_mask.squeeze(0)[::downsample, ::downsample]].reshape(-1,3).cuda()
            fused_color = RGB2SH(rgb_norm[::downsample, ::downsample][tool_mask.squeeze(0)[::downsample, ::downsample]].reshape(-1,3))
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
            semantics = semantics[::downsample, ::downsample][tool_mask.squeeze(0)[::downsample, ::downsample]].reshape(-1,self.n_classes).cuda()

            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1

            opacities = inverse_sigmoid(0.6*torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._semantics = semantics
        self._deformation = self._deformation.to("cuda")
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._deformation.add_gaussians(self._xyz.shape[0], self._xyz)

    def training_setup(self, training_args):
        self.percent_dense = training_args["percent_dense"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args["position_lr_init"] * self.spatial_lr_scale, "name": "xyz"},
            {'params': self._deformation.parameters(), 'lr': training_args["deformation_lr_init"] * self.spatial_lr_scale, "name": "deformation"},
            {'params': [self._features_dc], 'lr': training_args["feature_lr"], "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args["feature_lr"] / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args["opacity_lr"], "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args["scaling_lr"], "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args["rotation_lr"], "name": "rotation"}
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            optimizable_tensors[group["name"]] = []
            for idx in range(len(group['params'])):
                if len(mask) != len(group['params'][idx]):
                    continue
                stored_state = self.optimizer.state.get(group['params'][idx], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][idx]]
                    group["params"][idx] = nn.Parameter((group["params"][idx][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][idx]] = stored_state

                    optimizable_tensors[group["name"]].append(group["params"][idx])
                else:
                    group["params"][idx] = nn.Parameter(group["params"][idx][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]].append(group["params"][idx])
        # squeeze tensors
        for key in optimizable_tensors:
            if len(optimizable_tensors[key]) == 1:
                optimizable_tensors[key] = optimizable_tensors[key][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        grad_weighing = self.hooks is not None
        self.enable_grad_weighing(False)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.gradient_accum = self.gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self._semantics = self._semantics[valid_points_mask]
        self.enable_grad_weighing(grad_weighing)
        self._deformation.replace(optimizable_tensors['deformation'], optimizable_tensors['xyz'], reinit=True)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in tensors_dict:
                continue
            optimizable_tensors[group["name"]] = []
            if len(group['params']) == 1:
                extension_tensors = [tensors_dict[group["name"]]]
            else:
                extension_tensors = tensors_dict[group["name"]]
            for idx, extension_tensor in enumerate(extension_tensors):
                stored_state = self.optimizer.state.get(group['params'][idx], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][idx]]
                    group["params"][idx] = nn.Parameter(torch.cat((group["params"][idx], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][idx]] = stored_state

                    optimizable_tensors[group["name"]].append(group["params"][idx])
                else:
                    group["params"][idx] = nn.Parameter(torch.cat((group["params"][idx], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]].append(group["params"][idx])
        # squeeze tensors
        for key in optimizable_tensors:
            if len(optimizable_tensors[key]) == 1:
                optimizable_tensors[key] = optimizable_tensors[key][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantics):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation
       }

        if isinstance(self._deformation, ExplicitDeformation):
            new_deformation = self._deformation.get_new_params(new_xyz.shape)
            d['deformation'] = new_deformation
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        grad_weighing = self.hooks is not None
        self.enable_grad_weighing(False)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.gradient_accum = torch.cat((self.gradient_accum, torch.zeros((new_xyz.shape[0], 1), device="cuda")), dim=0)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._semantics = torch.cat((self._semantics, new_semantics), dim=0)
        self.enable_grad_weighing(grad_weighing)
        self._deformation.replace(optimizable_tensors['deformation'], optimizable_tensors['xyz'], reinit=False)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # exclude Gaussians which are settled based on grad_weighing
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.gradient_accum.squeeze() < self.cfg['visit_offset'])

        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_semantics = self._semantics[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantics)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # exclude Gaussians which are settled based on grad_weighing
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.gradient_accum.squeeze() < self.cfg['visit_offset'])
        new_xyz = self._xyz[selected_pts_mask] 
        # - 0.001 * self._xyz.grad[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantics = self._semantics[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantics)

    def densify(self, max_grad):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, 1.0)
        self.densify_and_split(grads, max_grad, 1.0)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.gradient_accum[update_filter] += 1

    def compute_regulation(self, visibility_filter):
        return self._deformation.reg_loss(visibility_filter)

    def get_closest_gaussian(self, point, use_cov=False, k=1):
        xyz, scales, rots, opacity, features, _ = self.forward()
        delta = point - xyz
        if use_cov:
            inv_cov = build_inv_cov(scales, rots)
            affinity = (opacity*torch.exp(-(delta[..., None, :] @ inv_cov @ delta[..., None]).squeeze(1)/2)).squeeze(1)
        else:
            affinity = -torch.linalg.norm(delta, dim=-1, ord=2)#(opacity * torch.exp(-(delta[..., None, :] @ delta[..., None]).squeeze(1) / 2)).squeeze(1)
        idx = torch.topk(affinity, k=k).indices
        return self._xyz[idx].mean(dim=0), idx, xyz[idx].mean(dim=0), affinity[idx]

    def enable_grad_weighing(self, enable=True):
        visit_alpha = self.cfg['visit_alpha']
        visit_offset = self.cfg['visit_offset']
        if visit_alpha is not None:
            if enable and (self.hooks is None):
                norm = self.grad_weighing(torch.ones(1, device="cuda"),
                                          torch.zeros(1, device="cuda"),
                                          visit_alpha, visit_offset,
                                          torch.ones(1, device="cuda"))
                # register grad hook
                hooks = []
                for param in [self._xyz, self._rotation, self._scaling, self._opacity, self._features_dc]:
                    hooks.append(param.register_hook(
                        partial(self.grad_weighing, visited=self.gradient_accum, visit_alpha=visit_alpha,
                                visit_offset=visit_offset, norm=norm)))
                hooks.append(self._features_rest.register_hook(
                    partial(self.grad_weighing, visited=self.gradient_accum, visit_alpha=visit_alpha,
                            visit_offset=visit_offset, norm=norm, offset=0.001)))
                self.hooks = hooks
            else:
                if self.hooks is not None:
                    [h.remove() for h in self.hooks]
                    self.hooks = None

    @staticmethod
    def grad_weighing(grad, visited, visit_alpha, visit_offset, norm, offset=0.0):
        """
            weight gradient by visit function -> points that have often been updated will get smaller gradient
        """
        #ToDo make broadcasting without transpose as it uses non contiguous views
        return (grad.transpose(0, -1) * ((1.0+offset) - torch.sigmoid(visit_alpha * (visited.squeeze() - visit_offset))) / norm).transpose(0, -1)

    def reset_optimizer(self):
        for group in self.optimizer.param_groups:
            for idx in range(len(group['params'])):
                stored_state = self.optimizer.state.get(group['params'][idx], None)
                if stored_state is not None:
                    stored_state["exp_avg"][:] = 0.0
                    stored_state["exp_avg_sq"][:] = 0.0
                    stored_state["state_step"] = 0
                    self.optimizer.state[group['params'][idx]] = stored_state
