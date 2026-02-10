import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

import trimesh
import kiui

try:
    import nvdiffrast.torch as dr
except Exception:
    dr = None


def _to_tensor(x, device, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _resolve_mesh_data(mesh):
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        v = mesh.vertices
        f = mesh.faces
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        if torch.is_tensor(f):
            f = f.detach().cpu().numpy()
        v = np.asarray(v, dtype=np.float32)
        f = np.asarray(f, dtype=np.int64)
        if not hasattr(mesh, "visual") or mesh.visual is None or getattr(mesh.visual, "uv", None) is None:
            raise ValueError("Mesh has no UV coordinates.")
        vt = mesh.visual.uv
        if torch.is_tensor(vt):
            vt = vt.detach().cpu().numpy()
        vt = np.asarray(vt, dtype=np.float32)
        ft = f
        vn = getattr(mesh, "vertex_normals", None)
        if vn is None or len(vn) == 0:
            vn = trimesh.Trimesh(vertices=v, faces=f, process=False).vertex_normals
        if torch.is_tensor(vn):
            vn = vn.detach().cpu().numpy()
        vn = np.asarray(vn, dtype=np.float32)
        return v, f, vt, ft, vn

    if hasattr(mesh, "v") and hasattr(mesh, "f"):
        v = mesh.v
        f = mesh.f
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        if torch.is_tensor(f):
            f = f.detach().cpu().numpy()
        v = np.asarray(v, dtype=np.float32)
        f = np.asarray(f, dtype=np.int64)
        vt = getattr(mesh, "vt", None)
        ft = getattr(mesh, "ft", None)
        if vt is None or ft is None:
            raise ValueError("Mesh missing UV coordinates (vt/ft).")
        if torch.is_tensor(vt):
            vt = vt.detach().cpu().numpy()
        if torch.is_tensor(ft):
            ft = ft.detach().cpu().numpy()
        vt = np.asarray(vt, dtype=np.float32)
        ft = np.asarray(ft, dtype=np.int64)
        vn = getattr(mesh, "vn", None)
        if vn is None or len(vn) == 0:
            vn = trimesh.Trimesh(vertices=v, faces=f, process=False).vertex_normals
        if torch.is_tensor(vn):
            vn = vn.detach().cpu().numpy()
        vn = np.asarray(vn, dtype=np.float32)
        return v, f, vt, ft, vn

    raise TypeError("Unsupported mesh type for normal export.")


def compute_vertex_tangents(
    v: torch.Tensor,
    f: torch.Tensor,
    vt: torch.Tensor,
    ft: torch.Tensor,
    vn: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # face tangents/bitangents
    p0 = v[f[:, 0]]
    p1 = v[f[:, 1]]
    p2 = v[f[:, 2]]

    uv0 = vt[ft[:, 0]]
    uv1 = vt[ft[:, 1]]
    uv2 = vt[ft[:, 2]]

    dp1 = p1 - p0
    dp2 = p2 - p0
    duv1 = uv1 - uv0
    duv2 = uv2 - uv0

    denom = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    r = torch.zeros_like(denom)
    valid = denom.abs() > 1e-8
    r[valid] = 1.0 / denom[valid]

    t = (dp1 * duv2[:, 1:2] - dp2 * duv1[:, 1:2]) * r[:, None]
    b = (dp2 * duv1[:, 0:1] - dp1 * duv2[:, 0:1]) * r[:, None]

    vt_count = vt.shape[0]
    tan = torch.zeros((vt_count, 3), device=v.device)
    bitan = torch.zeros((vt_count, 3), device=v.device)
    nrm = torch.zeros((vt_count, 3), device=v.device)

    for corner in range(3):
        uv_idx = ft[:, corner]
        v_idx = f[:, corner]
        tan.index_add_(0, uv_idx, t)
        bitan.index_add_(0, uv_idx, b)
        nrm.index_add_(0, uv_idx, vn[v_idx])

    nrm = F.normalize(nrm, dim=-1, eps=1e-6)
    tan = tan - nrm * (tan * nrm).sum(dim=-1, keepdim=True)
    tan = F.normalize(tan, dim=-1, eps=1e-6)
    bitan = F.normalize(bitan, dim=-1, eps=1e-6)
    handed = torch.sign((torch.cross(nrm, tan, dim=-1) * bitan).sum(dim=-1, keepdim=True))
    handed = torch.where(handed == 0, torch.ones_like(handed), handed)
    bitan = F.normalize(torch.cross(nrm, tan, dim=-1) * handed, dim=-1, eps=1e-6)

    return tan, bitan, nrm


def save_normal_map(mesh, pred_world_normal_map, save_path, flip_green=False):
    if dr is None:
        raise RuntimeError("nvdiffrast is required for save_normal_map.")

    v, f, vt, ft, vn = _resolve_mesh_data(mesh)

    if torch.is_tensor(pred_world_normal_map):
        normal = pred_world_normal_map.detach()
        device = normal.device
    else:
        normal = torch.tensor(pred_world_normal_map)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normal = normal.to(device=device, dtype=torch.float32)
    if normal.ndim == 3 and normal.shape[0] in (3, 4):
        normal = normal[:3].permute(1, 2, 0)
    if normal.ndim == 3 and normal.shape[2] == 4:
        normal = normal[:, :, :3]

    if normal.max() > 1.5:
        normal = normal / 255.0
    if normal.min() >= 0.0:
        normal = normal * 2.0 - 1.0
    normal = F.normalize(normal, dim=-1, eps=1e-6)

    h, w = normal.shape[:2]

    v = _to_tensor(v, device)
    f = _to_tensor(f, device, dtype=torch.int64)
    vt = _to_tensor(vt, device)
    ft = _to_tensor(ft, device, dtype=torch.int64)
    vn = _to_tensor(vn, device)

    tan, bitan, nrm = compute_vertex_tangents(v, f, vt, ft, vn)

    glctx = dr.RasterizeCudaContext() if torch.cuda.is_available() else dr.RasterizeGLContext()
    uv = vt * 2.0 - 1.0
    uv = torch.cat([uv, torch.zeros_like(uv[:, :1]), torch.ones_like(uv[:, :1])], dim=-1)
    ft_i32 = ft.to(dtype=torch.int32).contiguous()
    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft_i32, (h, w))

    t_map, _ = dr.interpolate(tan.unsqueeze(0), rast, ft_i32)
    b_map, _ = dr.interpolate(bitan.unsqueeze(0), rast, ft_i32)
    n_map, _ = dr.interpolate(nrm.unsqueeze(0), rast, ft_i32)
    mask, _ = dr.interpolate(torch.ones_like(vt[:, :1]).unsqueeze(0), rast, ft_i32)

    t_map = F.normalize(t_map.squeeze(0), dim=-1, eps=1e-6)
    b_map = F.normalize(b_map.squeeze(0), dim=-1, eps=1e-6)
    n_map = F.normalize(n_map.squeeze(0), dim=-1, eps=1e-6)
    mask = mask.squeeze(0)

    n_tan = torch.stack(
        [
            (normal * t_map).sum(dim=-1),
            (normal * b_map).sum(dim=-1),
            (normal * n_map).sum(dim=-1),
        ],
        dim=-1,
    )
    n_tan = F.normalize(n_tan, dim=-1, eps=1e-6)
    if flip_green:
        n_tan[..., 1] = -n_tan[..., 1]

    n_tan = (n_tan * 0.5 + 0.5).clamp(0, 1)
    if mask is not None:
        bg = torch.tensor([0.5, 0.5, 1.0], device=device).view(1, 1, 3)
        n_tan = n_tan * mask + bg * (1 - mask)

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    kiui.write_image(save_path, n_tan.detach().cpu().numpy())
