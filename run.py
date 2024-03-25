import numpy as np
import os
import torch
import wandb
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from scipy.spatial import KDTree
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
from torch.utils.data import DataLoader
from src.utils.camera import Camera
from src.utils.FrameVisualizer import FrameVisualizer
from src.utils.flow_utils import get_scene_flow, get_depth_from_raft
from src.utils.PointTracker import PointTracker, mte, surv_2d, delta_2d
from src.utils.datasets import StereoMIS
from src.utils.loss_utils import l1_loss
from src.utils.renderer import render
from src.scene.gaussian_model import GaussianModel


class SceneOptimizer():
    def __init__(self, cfg, args):
        self.total_iters = 0
        self.cfg = cfg
        self.args = args
        self.visualize = args.visualize
        self.scale = cfg['scale']
        self.device = cfg['device']
        self.output = cfg['data']['output']

        self.frame_reader = StereoMIS(cfg, args, scale=self.scale)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=0 if args.debug else 4)
        self.net = GaussianModel(cfg=cfg['model'])
        self.camera = Camera(cfg['cam'])
        self.visualizer = FrameVisualizer(self.output, cfg, self.net)

        self.log_freq = args.log_freq
        self.log = args.log is not None
        self.run_id = wandb.util.generate_id()
        log_cfg = cfg.copy()
        log_cfg.update(vars(args))
        if self.log:
            wandb.init(id=self.run_id, name=args.log, config=log_cfg, project='gtracker', group=args.log_group)
        self.background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        self.dbg = args.debug
        self.pt_tracker = None
        track_file = os.path.join(cfg['data']['input_folder'], 'track_pts.pckl')
        if os.path.isfile(track_file):
            self.pt_tracker = PointTracker(cfg, self.net, track_file)
        self.last_frame = None
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        self.raft = self.raft.eval()
        self.baseline = cfg['cam']['stereo_baseline']/1000.0*self.scale

    def fit(self, frame, iters, incremental):
        self.net.reset_optimizer()
        av_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]
        idx, gt_color, gt_depth, gt_c2w, tool_mask = frame
        self.camera.set_c2w(gt_c2w)
        for iter in range(1, iters+1):
            if self.cfg['training']['spherical_harmonics'] and iter > iters/2:
                self.net.enable_spherical_harmonics()
            self.total_iters += 1
            self.net.train(iter == 1)
            render_pkg = render(self.camera, self.net, self.background, deform=incremental)
            self.net.eval()
            color = render_pkg['render'][None, ...]
            depth = render_pkg['depth'][None, ...]

            # Loss
            Ll1 = self.cfg['training']['w_color']*l1_loss(color[tool_mask], gt_color[tool_mask])
            Ll1_depth = self.cfg['training']['w_depth']*l1_loss(depth[tool_mask]/self.scale, gt_depth[tool_mask]/self.scale)
            loss = Ll1 + Ll1_depth
            if incremental:
                l_rigidtrans, l_rigidrot, l_iso, l_visible = self.net.compute_regulation(render_pkg["visibility_filter"])
                def_loss = self.cfg['training']['w_def']['rigid']*l_rigidtrans + self.cfg['training']['w_def']['iso']*l_iso+ self.cfg['training']['w_def']['rot']*l_rigidrot+ self.cfg['training']['w_def']['nvisible']*l_visible
                loss += def_loss
            else:
                l_rigidtrans, l_rigidrot, l_iso, l_visible = torch.zeros_like(Ll1), torch.zeros_like(Ll1), torch.zeros_like(Ll1), torch.zeros_like(Ll1)
            loss.backward()
            viewspace_point_tensor_grad = torch.zeros_like(render_pkg["viewspace_points"])
            viewspace_point_tensor_grad += render_pkg["viewspace_points"].grad

            ########### Logging & Evaluation ###################
            with torch.no_grad():
                av_loss[0] += Ll1.item()
                av_loss[1] += Ll1_depth.item()
                av_loss[2] += l_rigidtrans.item()
                av_loss[3] += l_rigidrot.item()
                av_loss[4] += l_iso.item()
                av_loss[5] += l_visible.item()
                av_loss[-1] += 1
                if ((self.total_iters % self.log_freq) == 0) and self.log:
                    wandb.log({'color_loss': av_loss[0] / av_loss[-1],
                               'depth_loss': av_loss[1] / av_loss[-1],
                               'rigidtrans_loss': av_loss[2] / av_loss[-1],
                               'rigidrot_loss': av_loss[3] / av_loss[-1],
                               'iso_loss': av_loss[4] / av_loss[-1],
                               'visible_loss': av_loss[5] / av_loss[-1],
                               'loss': sum(av_loss[:-1]) / av_loss[-1]}, step=self.total_iters)
                    av_loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]

                self.net.add_densification_stats(viewspace_point_tensor_grad, render_pkg["visibility_filter"])
                if not incremental:
                    # Densification
                    if iter > self.cfg["training"]["densify_from_iter"] and iter % self.cfg["training"]["densification_interval"] == 0:
                        self.net.densify(self.cfg["training"]["densify_grad_threshold"])

                # Optimizer step
                if iter < iters:
                    self.net.optimizer.step()
                    self.net.optimizer.zero_grad(set_to_none=True)

    def run(self):
        torch.cuda.empty_cache()
        pt_track_stats = {"pred_2d": []}

        for ids, gt_color, gt_color_r, gt_c2w, tool_mask, semantics in tqdm(self.frame_loader, total=self.n_img):
            gt_color = gt_color.cuda()
            gt_color_r = gt_color_r.cuda()
            gt_c2w = gt_c2w.cuda()
            tool_mask = tool_mask.cuda() if tool_mask is not None else None
            semantics = semantics.float().cuda() if semantics is not None else None
            with torch.no_grad():
                gt_depth, flow_valid = get_depth_from_raft(self.raft, gt_color, gt_color_r, self.baseline)
            frame = ids, gt_color, gt_depth, gt_c2w, tool_mask

            if ids.item() == 0:
                self.net.create_from_pcd(gt_color, gt_depth, gt_c2w, self.camera, tool_mask, semantics=semantics)
                self.net.training_setup(self.cfg['training'])
                self.fit(frame, iters=self.cfg['training']['iters_first'], incremental=False)
            else:
                if ids.item() == 1:
                    if self.cfg['training']['grad_weighing']:
                        self.net.enable_grad_weighing(True)

                with torch.no_grad():
                    # add new points
                    self.camera.set_c2w(gt_c2w)
                    render_pkg = render(self.camera, self.net, self.background, deform=True)
                    mask = render_pkg['alpha'][None,...,None].squeeze(-1) < 0.95
                    mask &= tool_mask
                    self.net.add_from_pcd(gt_color, gt_depth, gt_c2w, self.camera, mask, semantics=semantics) if self.cfg['training']['add_points'] else 0.0

                    # optical flow init
                    if self.cfg['training']['optical_flow_init']:
                        scene_flow, anchor_pts, valid = get_scene_flow(self.raft, render_pkg['render'][None,...], gt_color, render_pkg['depth'][None,...],gt_depth, tool_mask, self.camera)
                        valid &= tool_mask.squeeze(0) & flow_valid.squeeze(0)
                        tree = KDTree(anchor_pts[valid].cpu().numpy())
                        neighbour_dists, neighbours = tree.query(self.net._deformation.get_deformed_means(self.net.get_xyz).cpu().numpy(), k=3)#, eps=0.1)
                        weights = torch.exp(-50.0*(torch.from_numpy(neighbour_dists).cuda()))
                        deformation = scene_flow[valid][torch.from_numpy(neighbours).cuda()]
                        self.net._deformation.init_from_flow(deformation.clamp(-0.01, 0.01), weights)

                self.fit(frame, iters=self.cfg['training']['iters'], incremental=True)
            self.last_frame = gt_color.detach()

            # eval
            with torch.no_grad():
                log_dict = {}
                if self.pt_tracker is not None:
                        if not self.pt_tracker.is_initialized():
                            self.pt_tracker.init_tracking_points(gt_c2w)
                        pts_3d_gt, pts_3d, pts_2d, l2_3d, l2_2d, pts_2d_gt = self.pt_tracker.eval(gt_c2w, ids.item())
                        pt_track_stats["pred_2d"].append(pts_2d.cpu().numpy())
                        log_dict.update({'pt_track_l2_2d': l2_2d, 'frame': ids[0].item()})
                else:
                    pts_2d, pts_2d_gt = None, None
                if self.visualize:

                    outmap, outsem, outrack = self.visualizer.save_imgs(ids.item(), gt_depth, gt_color,
                                                                        gt_c2w, pts_2d, pts_2d_gt)
                    if self.log:
                        log_dict.update({'mapping': wandb.Image(outmap),
                                         'tracking': wandb.Image(outrack) if outrack is not None else None,
                                         'semantic': wandb.Image(outsem)})
                if self.log:
                    wandb.log(log_dict)

        if self.log:
            # eval point tracking
            gt_2d, valid = self.pt_tracker.get_gt_2d_pts()
            pred_2d = np.stack(pt_track_stats["pred_2d"], axis=1)
            H, W = self.camera.get_params()[:2]
            wandb.summary['MTE_2D'] = mte(pred_2d, gt_2d, valid)
            wandb.summary['delta_2D'] = delta_2d(pred_2d, gt_2d, valid, H, W)
            wandb.summary['survival_2D'] = surv_2d(pred_2d, gt_2d, valid, H, W)
        with open(os.path.join(self.output, 'tracked.pckl'), 'wb') as f:
            pickle.dump(pt_track_stats, f)
        print('...finished')


if __name__ == "__main__":
    # Set up command line argument parser
    from src.config import load_config
    import random

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('config', type=str)
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--log', type=str)
    parser.add_argument('--log_group', type=str, default='default')
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/base.yaml')
    cfg['data']['output'] = args.output if args.output else cfg['data']['output']

    trainer = SceneOptimizer(cfg, args)
    trainer.run()
