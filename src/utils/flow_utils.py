import numpy as np
import torch
import cv2


def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0)
    b = c2w.shape[0]
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1).to(device)

    dirs = dirs.reshape(1, -1, 1, 3).expand(b, -1, -1, -1)
    rays_d = torch.sum(dirs * c2w[:, None, :3, :3], -1)
    rays_o = c2w[:, :3, -1][:, None, :].expand(rays_d.shape)
    return rays_o, rays_d, dirs.squeeze(-2)


def get_surface_pts(depth, fx, fy, cx, cy, c2w, device):
    b, H, W = depth.shape
    rays_o, rays_d, _ = get_rays(H, W, fx, fy, cx, cy, c2w, device)
    pts = rays_o + depth.view(b, -1, 1)*rays_d
    return pts.view(b,H,W,3)


def reproject(pts2d, depth, fx, fy, cx, cy, c2ws):
    dirs = torch.stack([(pts2d[..., 0] - cx) / fx, (pts2d[...,1] - cy) / fy, torch.ones_like(pts2d[...,0], device=pts2d[0].device)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs.unsqueeze(-2) * c2ws[:, :3, :3], -1)
    rays_o = c2ws[:, :3, -1].expand(rays_d.shape)
    pts = rays_o + depth[pts2d[..., 1], pts2d[..., 0], None] * rays_d
    return pts


def remap_from_flow(x, flow):
    # get optical flow correspondences
    n, _, h, w = flow.shape
    row_coords, col_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    flow_off = torch.empty_like(flow)
    flow_off[:, 1] = 2 * (flow[:, 1] + row_coords.to(flow.device)) / (h - 1) - 1
    flow_off[:, 0] = 2 * (flow[:, 0] + col_coords.to(flow.device)) / (w - 1) - 1
    x = torch.nn.functional.grid_sample(x, flow_off.permute(0, 2, 3, 1), align_corners=True)
    valid = (x > 0).any(dim=1).unsqueeze(1)
    return x, valid


def get_scene_flow(raft, img1, img2, depth1, depth2, mask, camera):
    optical_flow = raft(2 * img1.permute(0, 3, 1, 2) - 1.0, 2 * img2.permute(0, 3, 1, 2) - 1.0)[-1]
    depth_interp = depth2.clone()
    depth_interp[~mask] = depth1[~mask]
    H, W, fx, fy, cx, cy = camera.get_params()
    src_pts = get_surface_pts(depth1, fx, fy, cx, cy, camera.c2w, depth1.device)
    target_pts = get_surface_pts(depth2, fx, fy, cx, cy, camera.c2w, depth2.device)
    target_remapped, valid = remap_from_flow(target_pts.permute(0,3,1,2), optical_flow)
    scene_flow = target_remapped.permute(0,2,3,1) - src_pts
    return scene_flow.squeeze(), src_pts.squeeze(), valid.squeeze()


def get_depth_from_raft(raft, img1, img2, baseline):
    flow = raft(2 * img1.permute(0, 3, 1, 2) - 1.0, 2 * img2.permute(0, 3, 1, 2) - 1.0)[-1]
    baseline_t = baseline * torch.ones_like(flow[:, 0])
    depth = baseline_t / -flow[:, 0]
    depth = torch.from_numpy(
        cv2.bilateralFilter(depth.cpu().numpy().squeeze(), d=-1, sigmaColor=2.5, sigmaSpace=2.5)).cuda().unsqueeze(0)
    valid = flow[:, 1].abs() < 1.5
    return depth, valid