import torch
import math

def getProjectionMatrix(znear, zfar, W, H, fx, fy, cx ,cy):
    P = torch.zeros(4, 4)

    P[0, 0] = 2 * fx / W
    P[1, 1] = 2 * fy / H
    P[0, 2] = 2 * (cx / W) - 1
    P[1, 2] = 2 * (cy / H) - 1
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[3, 2] = 1.0
    P[2, 3] = -(2 * zfar * znear) / (zfar - znear)

    return P

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class Camera():
    def __init__(self, cfg):
        self.H = cfg['H']
        self.W = cfg['W']
        self.fx = cfg['fx']
        self.fy = cfg['fy']
        self.cx = cfg['cx']
        self.cy = cfg['cy']
        self.point_src_light_model = cfg['point_src_light_model']

        # 4D Gaussians Viewpoint camera attributes
        self.FoVx = focal2fov(self.fx, self.W)
        self.FoVy = focal2fov(self.fy, self.H)
        self.image_height = self.H
        self.image_width = self.W
        self.c2w = None
        self.projection_matrix = getProjectionMatrix(cfg['znear'], cfg['zfar'], self.W, self.H, self.fx, self.fy, self.cx, self.cy).transpose(0,1).cuda()
        self.time = None
        i, j = torch.meshgrid(torch.linspace(0, self.W - 1, self.W, device="cuda"),
                              torch.linspace(0, self.H - 1, self.H, device="cuda"), indexing='ij')
        cam_rays_d = torch.stack([(i.t()-self.cx)/self.fx, (j.t()-self.cy)/self.fy, torch.ones_like(i.t(), device="cuda")], -1)
        cam_rays_d = (cam_rays_d / torch.linalg.norm(cam_rays_d, dim=-1, keepdims=True))
        self.cos_alpha = cam_rays_d[..., 2].squeeze()

    def set_c2w(self, c2w: torch.tensor):
        self.c2w = c2w.squeeze(0).clone()

    def set_time(self, time:float):
        self.time = time

    @property
    def world_view_transform(self):
        assert self.c2w is not None
        tr = torch.linalg.inv(self.c2w)
        return tr.transpose(0,1)
    @property
    def camera_center(self):
        return self.c2w[:3,3] if self.c2w is not None else None

    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    def get_params(self, scale=1.0):
        return int(scale*self.H), int(scale*self.W), scale*self.fx, scale*self.fy, scale*self.cx, scale*self.cy

    def spotlight_render(self, color, depth):
        # spotlight with cosine light spread function
        # lambertian model -> angle between surface normal and camera ray in world coordinates
        # rays_d_norm = (-rays_d / torch.linalg.norm(rays_d, dim=-1, keepdims=True))
        # cos_theta = (rays_d_norm * normals).sum(dim=-1).clamp(0.5,1)
        # we approximate the square law 1/z² by c1*exp(-c2*z) in the range of [0.05 to 0.3] which is more stable.
        # assume gamma = 2 -> cos^1/2 = sqrt(cos) torch.sqrt(cos_theta[..., None])
        if self.point_src_light_model:
            return torch.square(self.cos_alpha.view(depth.shape)[..., None]) * color #* torch.exp(-1.3 * depth[..., None])
        else:
            return color

    def inverse_splotlight_render(self, color, depth):
        if self.point_src_light_model:
            return color / (torch.square(self.cos_alpha.view(depth.shape)[..., None])) #* torch.exp(-1.3 * depth[..., None]))
        else:
            return color
