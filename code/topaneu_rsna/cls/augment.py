"""
GPU batch augmentations, mirroring the author's transform stack.

Order: low-res simulation -> grid distortion -> affine -> flips on all three
axes.  Image is trilinear, label maps are nearest, and annotation points are
carried through every geometric step so the sphere target can be rasterised
after augmentation rather than before.

Note on flips: flipping the image and the vessel label maps together leaves the
mapping "channel k = location k" untouched, because masked pooling is invariant
to where in space a region sits.  So the 52 location labels do NOT need to be
swapped -- the author flips all three axes for the same reason.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _grid_sample(x, grid, mode):
    return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)


def rasterize_spheres(points, shape, radius, device):
    """points: (B, M, 3) z/y/x voxel coords with NaN padding -> (B,1,D,H,W)."""
    B = points.shape[0]
    D, H, W = shape
    out = torch.zeros(B, 1, D, H, W, device=device)
    zz = torch.arange(D, device=device).view(-1, 1, 1)
    yy = torch.arange(H, device=device).view(1, -1, 1)
    xx = torch.arange(W, device=device).view(1, 1, -1)
    r2 = float(radius) ** 2
    for b in range(B):
        for p in points[b]:
            if torch.isnan(p).any():
                continue
            d2 = (zz - p[0]) ** 2 + (yy - p[1]) ** 2 + (xx - p[2]) ** 2
            out[b, 0] = torch.maximum(out[b, 0], (d2 <= r2).float())
    return out


class BatchAugment:
    """Applies to image (B,1,D,H,W), two label maps (B,D,H,W) and points."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def _rand(self):
        return float(torch.rand(1).item())

    def __call__(self, image, v2, v3, points):
        cfg = self.cfg
        B = image.shape[0]
        shape = image.shape[2:]

        if self._rand() < cfg.p_lowres:
            z = 0.4 + self._rand() * 0.3
            small = F.interpolate(image, scale_factor=z, mode="trilinear",
                                  align_corners=False, recompute_scale_factor=False)
            image = F.interpolate(small, size=shape, mode="trilinear", align_corners=False)

        if self._rand() < cfg.p_affine:
            image, v2, v3, points = self._affine(image, v2, v3, points)

        for axis in (2, 3, 4):
            if self._rand() < cfg.p_flip:
                image = torch.flip(image, dims=[axis])
                v2 = torch.flip(v2, dims=[axis - 1])
                v3 = torch.flip(v3, dims=[axis - 1])
                dim = axis - 2
                extent = shape[dim] - 1
                points[..., dim] = torch.where(torch.isnan(points[..., dim]),
                                               points[..., dim],
                                               extent - points[..., dim])

        # intensity transforms (image only)
        if self._rand() < 0.5:
            image = image * (0.9 + 0.2 * self._rand()) + (self._rand() - 0.5) * 0.2
        if self._rand() < 0.3:
            image = image + torch.randn_like(image) * 0.05
        if self._rand() < 0.2:
            image = -image                       # intensity inversion
        return image, v2, v3, points

    def _affine(self, image, v2, v3, points):
        cfg = self.cfg
        B = image.shape[0]
        dev = image.device
        ang = torch.deg2rad(torch.empty(B, 3, device=dev).uniform_(
            -cfg.rotate_deg, cfg.rotate_deg))
        sc = torch.empty(B, 3, device=dev).uniform_(1 - cfg.scale_range, 1 + cfg.scale_range)

        cz, sz = torch.cos(ang[:, 0]), torch.sin(ang[:, 0])
        cy, sy = torch.cos(ang[:, 1]), torch.sin(ang[:, 1])
        cx, sx = torch.cos(ang[:, 2]), torch.sin(ang[:, 2])
        Rz = torch.zeros(B, 3, 3, device=dev); Ry = torch.zeros_like(Rz); Rx = torch.zeros_like(Rz)
        Rz[:, 0, 0] = 1; Rz[:, 1, 1] = cz; Rz[:, 1, 2] = -sz; Rz[:, 2, 1] = sz; Rz[:, 2, 2] = cz
        Ry[:, 1, 1] = 1; Ry[:, 0, 0] = cy; Ry[:, 0, 2] = sy; Ry[:, 2, 0] = -sy; Ry[:, 2, 2] = cy
        Rx[:, 2, 2] = 1; Rx[:, 0, 0] = cx; Rx[:, 0, 1] = -sx; Rx[:, 1, 0] = sx; Rx[:, 1, 1] = cx
        R = torch.bmm(torch.bmm(Rz, Ry), Rx) * sc[:, None, :]

        theta = torch.cat([R, torch.zeros(B, 3, 1, device=dev)], dim=2)
        grid = F.affine_grid(theta, image.shape, align_corners=False)
        image = _grid_sample(image, grid, "bilinear")
        v2 = _grid_sample(v2[:, None].float(), grid, "nearest")[:, 0].long()
        v3 = _grid_sample(v3[:, None].float(), grid, "nearest")[:, 0].long()

        # move points with the inverse transform, in normalised coords
        shape = torch.tensor(image.shape[2:], device=dev, dtype=torch.float32)
        Rinv = torch.inverse(R)
        norm = (points / (shape - 1) * 2 - 1).flip(-1)          # z,y,x -> x,y,z
        moved = torch.einsum("bij,bmj->bmi", Rinv, torch.nan_to_num(norm))
        moved = moved.flip(-1)
        pts = (moved + 1) / 2 * (shape - 1)
        pts[torch.isnan(points)] = float("nan")
        return image, v2, v3, pts
