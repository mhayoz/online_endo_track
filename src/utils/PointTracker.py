import torch
import pickle
import numpy as np
from src.utils.flow_utils import reproject
from src.utils.renderer import render
from src.utils.camera import Camera


def mte(predicted, gt, valid):
    error = gt - predicted
    l2_2ds = []
    for i in range(error.shape[0]):
        l2_2ds.append(np.nanmedian(np.linalg.norm(error[i][valid[i]], ord=2, axis=-1)))
    return np.mean(np.asarray(l2_2ds))


def delta_2d(predicted, gt, valid, H, W, thrs=(1, 2, 4, 8, 16)):
    d_sum = 0.0
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_pt = np.array([sx_, sy_]).reshape([1, 1, 2])
    for thr in thrs:
        # note we exclude timestep0 from this eval
        d_ = (np.linalg.norm(predicted[:, 1:]/ sc_pt - gt[:, 1:] / sc_pt, axis=-1, ord=2) < thr)  # B,S-1,N
        d_sum += np.mean(d_[valid[:, 1:]])
    d_avg = d_sum / len(thrs)
    return d_avg


def surv_2d(predicted, gt, valid, H, W, sur_thr=16):
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_pt = np.array([sx_, sy_]).reshape([1, 1, 2])
    dists = np.linalg.norm(predicted / sc_pt - gt / sc_pt, axis=-1, ord=2)  # B,S,N
    dist_ok = 1 - (dists > sur_thr) * valid  # B,S,N
    survival = np.cumprod(dist_ok, axis=1)  # B,S,N
    survival = np.mean(survival)
    return survival


class PointTracker(object):
    """
        Visualize points tracking in ESLAM
    """
    def __init__(self, cfg, net, gt_file):
        self.cfg = cfg
        self.net = net
        self.scale = cfg['scale']
        self.camera = Camera(cfg['cam'])
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.p_ref_indices = None

        with open(gt_file, 'rb') as f:
            cont = pickle.load(f)
            gt_points = cont['points']
        gt_points = np.stack([np.stack([np.array([-1, -1]) if i is None else i for i in pts]) for pts in gt_points])
        gt_points = np.stack([np.stack([np.array([0, 0]) if i is None else i for i in pts]) for pts in gt_points])
        mask = ~np.stack(cont['not_visible']).astype(bool)
        mask &= gt_points.sum(axis=-1) > 0
        gt_points = gt_points[mask[:, 0]]
        mask = mask[mask[:, 0]]
        self.gt_2d_pts = torch.tensor(gt_points).cuda()[:, slice(cfg['data']['start'],cfg['data']['stop'],cfg['data']['step'])]
        self.valid = mask[:, slice(cfg['data']['start'],cfg['data']['stop'],cfg['data']['step'])]

    def get_gt_2d_pts(self):
        return self.gt_2d_pts.cpu().numpy(), self.valid

    def init_tracking_points(self, c2w):
        pts = self.gt_2d_pts[:,0]
        H, W, fx, fy, cx, cy = self.camera.get_params()
        # render depth
        self.camera.set_c2w(c2w)
        render_pkg = render(self.camera, self.net, self.background)
        # project to 3D
        pts_3d = reproject(pts.long(), render_pkg['depth'], fx, fy, cx, cy, c2w)
        # get 3D locations of the closest Gaussian
        self.p_ref_indices = []
        self.p_ref_affinities = []
        for pt in pts_3d:
            _, ids, _, affinity = self.net.get_closest_gaussian(pt, k=3, use_cov=False)
            self.p_ref_indices.append(ids)
            self.p_ref_affinities.append(affinity)

    @torch.no_grad()
    def eval(self, c2w, idx):
        assert self.p_ref_indices is not None
        pts_gt = self.gt_2d_pts[:, idx]
        H, W, fx, fy, cx, cy = self.camera.get_params()
        # render depth
        self.camera.set_c2w(c2w)
        render_pkg = render(self.camera, self.net, self.background)
        # project to 3D
        pts_3d_gt = reproject(pts_gt, render_pkg['depth'], fx, fy, cx, cy, c2w)/self.scale
        pts_3d, pts_2d = self.get_2d_pts(c2w)
        pts_3d = pts_3d/self.scale
        # metrics
        l2_3d = torch.linalg.norm((pts_3d_gt - pts_3d), ord=2, dim=-1)[self.valid[:, idx]].mean()
        l2_2d = torch.linalg.norm(pts_gt - pts_2d, ord=2, dim=-1)[self.valid[:, idx]].mean()
        return pts_3d_gt, pts_3d, pts_2d, l2_3d, l2_2d, pts_gt

    def get_2d_pts(self, c2w, camera=None):
        H, W, fx, fy, cx, cy = self.camera.get_params() if camera is None else camera.get_params()
        pts_3d = torch.stack([self.net()[0][ids] for ids in self.p_ref_indices]) # N, K, 3
        # aggregate points
        pts_3d = torch.nanmedian(pts_3d, dim=1).values

        w2c = torch.linalg.inv(c2w)
        ref_pts_c = (w2c[:, :3, :3] @ pts_3d.T).squeeze(0).T + w2c[:, :3, 3]  # transform into camera space
        pts_2d = torch.stack((fx * ref_pts_c[..., 0] / ref_pts_c[..., 2] + cx,
                                  fy * ref_pts_c[..., 1] / ref_pts_c[..., 2] + cy), dim=-1)
        # check if 2D pts in image
        pts_2d[:, 0].clamp_(0, W - 1)
        pts_2d[:, 1].clamp_(0, H - 1)
        return pts_3d, pts_2d

    def is_initialized(self):
        return self.p_ref_indices is not None


