import os
import sys
import csv
import json
import subprocess
import copy
import time
from datetime import datetime, timedelta
from dataclasses import asdict, is_dataclass
import tyro
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import trimesh
from core.regression_models import TexGaussian
from core.options import AllConfigs, Options
from core.gs import GaussianRenderer
from external.clip import tokenize
from export_utils import save_normal_map

# LongCLIP support - lazy import
_longclip_module = None
_longclip_tokenize = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LONGCLIP_ROOT = os.path.join(SCRIPT_DIR, "third_party", "Long-CLIP")


def resolve_longclip_module(longclip_root=None):
    """Resolve and import LongCLIP module from third_party or installed package."""
    global _longclip_module, _longclip_tokenize
    if _longclip_module is not None:
        return _longclip_module, _longclip_tokenize

    last_exc = None
    # Try installed package first
    try:
        import longclip as longclip_mod
        _longclip_module = longclip_mod
        _longclip_tokenize = longclip_mod.tokenize
        return _longclip_module, _longclip_tokenize
    except Exception as exc:
        last_exc = exc

    # Try third_party paths
    candidates = []
    if longclip_root:
        candidates.append(longclip_root)
    if os.path.isdir(DEFAULT_LONGCLIP_ROOT) and DEFAULT_LONGCLIP_ROOT not in candidates:
        candidates.append(DEFAULT_LONGCLIP_ROOT)

    for root in candidates:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from model import longclip as longclip_mod
            _longclip_module = longclip_mod
            _longclip_tokenize = longclip_mod.tokenize
            print(f"[INFO] Loaded LongCLIP from: {root}")
            return _longclip_module, _longclip_tokenize
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"LongCLIP is not available; install longclip or check third_party/Long-CLIP exists "
        f"(default: {DEFAULT_LONGCLIP_ROOT})"
    ) from last_exc


def load_longclip_model(longclip_model_path, device, longclip_root=None):
    """Load LongCLIP model from checkpoint."""
    longclip_mod, longclip_tok = resolve_longclip_module(longclip_root)

    if not longclip_model_path:
        raise ValueError("LongCLIP model path is empty.")

    # Handle relative paths
    if not os.path.isabs(longclip_model_path):
        # Try relative to current working directory
        if not os.path.isfile(longclip_model_path):
            # Try relative to script directory
            script_relative = os.path.join(SCRIPT_DIR, longclip_model_path)
            if os.path.isfile(script_relative):
                longclip_model_path = script_relative
            else:
                # Try default location
                default_path = os.path.join(DEFAULT_LONGCLIP_ROOT, "checkpoints", "longclip-L.pt")
                if os.path.isfile(default_path):
                    longclip_model_path = default_path

    if not os.path.isfile(longclip_model_path):
        raise FileNotFoundError(f"LongCLIP model not found: {longclip_model_path}")

    print(f"[INFO] Loading LongCLIP from: {longclip_model_path}")
    # Load to CPU first, then move to device to avoid GPU memory fragmentation
    model, preprocess = longclip_mod.load(longclip_model_path, device="cpu")
    model = model.to(device)
    model.eval()
    return model, preprocess, longclip_tok

from ocnn.octree import Octree, Points
import ocnn

import nvdiffrast.torch as dr

import kiui
from kiui.mesh import Mesh
from kiui.op import uv_padding
from kiui.cam import orbit_camera, get_perspective

from safetensors.torch import load_file

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Converter(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt
        self.device = torch.device("cuda")

        self.tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1

        self.gs_renderer = GaussianRenderer(opt)
        self.enable_normal_head = str2bool(getattr(self.opt, "use_normal_head", "False"))
        
        if self.opt.force_cuda_rast:
            self.glctx = dr.RasterizeCudaContext()
        else:
            self.glctx = dr.RasterizeGLContext()
    
        self.proj = torch.from_numpy(get_perspective(self.opt.fovy)).float().to(self.device)
        self.v = self.f = None
        self.vt = self.ft = None
        self.deform = None

        self.model = TexGaussian(opt, self.device)

        self.pointcloud_dir = self.opt.pointcloud_dir

        # LongCLIP support
        self.longclip_model = None
        self.longclip_tokenize = None
        if self.opt.use_longclip:
            self._init_longclip()

        self.text_embedding = None
        if self.opt.use_text and self.opt.text_prompt:
            self.set_text_prompt(self.opt.text_prompt)

    def _init_longclip(self):
        """Initialize LongCLIP model for text encoding."""
        print("[INFO] Initializing LongCLIP for text encoding...")
        self.longclip_model, _, self.longclip_tokenize = load_longclip_model(
            self.opt.longclip_model,
            self.device,
            longclip_root=DEFAULT_LONGCLIP_ROOT
        )
        # Delete visual encoder to save memory (we only need text encoding)
        if hasattr(self.longclip_model, 'visual'):
            del self.longclip_model.visual
        self.longclip_model.requires_grad_(False)
        self.longclip_model.eval()
        print("[INFO] LongCLIP initialized successfully.")
    
    def normalize_mesh(self):
        self.mesh.vertices = self.mesh.vertices - self.mesh.bounding_box.centroid
        distances = np.linalg.norm(self.mesh.vertices, axis=1)
        self.mesh.vertices /= np.max(distances)

    def set_text_prompt(self, text_prompt: str):
        """Update text prompt and re-encode embedding.
        
        If use_longclip is enabled, uses LongCLIP for encoding (supports longer context).
        Otherwise, uses the standard CLIP encoder from the model.
        """
        self.opt.text_prompt = text_prompt
        if not self.opt.use_text:
            self.text_embedding = None
            return

        if self.opt.use_longclip and self.longclip_model is not None:
            # Use LongCLIP for text encoding
            context_length = getattr(self.opt, 'longclip_context_length', 248)
            token = self.longclip_tokenize(text_prompt, context_length=context_length, truncate=True)
            token = token.to(self.device)
            with torch.no_grad():
                # LongCLIP encode_text returns [batch, dim], we need [batch, seq, dim] format
                # Use token_embedding + transformer like standard CLIP for full sequence
                x = self.longclip_model.token_embedding(token)
                x = x + self.longclip_model.positional_embedding[:x.shape[1]]
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.longclip_model.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.longclip_model.ln_final(x)
                self.text_embedding = x.float()  # [bs, context_length, dim]
            print(f"[INFO] Text encoded with LongCLIP (context_length={context_length})")
        else:
            # Use standard CLIP encoder
            token = tokenize(text_prompt)
            token = token.to(self.device)
            self.text_embedding = self.model.text_encoder.encode(token).float()  # [bs, 77, 768]

    def load_mesh(self, path, num_samples = 200000):
        self.mesh = trimesh.load(path, force = 'mesh')
        self.normalize_mesh()

        point, idx = trimesh.sample.sample_surface(self.mesh, num_samples)
        normals = self.mesh.face_normals[idx]
        
        points_gt = Points(points = torch.from_numpy(point).float(), normals = torch.from_numpy(normals).float())
        points_gt.clip(min=-1, max=1)

        points = [points_gt]
        points = [pts.cuda(non_blocking = True) for pts in points]
        
        octrees = [self.points2octree(pts) for pts in points]
        octree_in = ocnn.octree.merge_octrees(octrees)

        octree_in.construct_all_neigh()

        xyzb = octree_in.xyzb(depth = octree_in.depth, nempty = True)
        x, y, z, b = xyzb
        xyz = torch.stack([x,y,z], dim = 1)
        octree_in.position = 2 * xyz / (2 ** octree_in.depth) - 1

        self.octree_in = octree_in

        self.input_data = self.octree_in.get_input_feature(feature = self.opt.input_feature, nempty = True) 

    def points2octree(self, points):
        octree = ocnn.octree.Octree(depth = self.opt.input_depth, full_depth = self.opt.full_depth)
        octree.build_octree(points)
        return octree
    
    def load_ckpt(self, ckpt_path):

        print('Start loading checkpoint')

        if ckpt_path.endswith('safetensors'):
            ckpt = load_file(ckpt_path, device='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        
        state_dict = self.model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict:
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                print(f'[WARN] unexpected param {k}: {v.shape}')
       
    @torch.no_grad()
    def render_gs(self, pose, use_material = False, use_normal = False):
    
        cam_poses = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

        batch_id = self.octree_in.batch_id(self.opt.input_depth, nempty = True)
        
        if use_normal and hasattr(self, "normal_gaussians"):
            out = self.gs_renderer.render(self.normal_gaussians, batch_id, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0))
        elif use_material:
            out = self.gs_renderer.render(self.mr_gaussians, batch_id, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0))
        else:
            out = self.gs_renderer.render(self.gaussians, batch_id, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0))
        
        image = out['image'].squeeze(1).squeeze(0) # [C, H, W]
        alpha = out['alpha'].squeeze(2).squeeze(1).squeeze(0) # [H, W]

        return image, alpha
    
    def render_mesh(self, pose, use_material = False, use_normal = False):

        h = w = self.opt.output_size

        v = self.v
        f = self.f

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ self.proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (h, w))

        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous() # [1, H, W, 1]
        alpha = dr.antialias(alpha, rast, v_clip, f).clamp(0, 1).squeeze(-1).squeeze(0) # [H, W] important to enable gradients!
        
        texc, texc_db = dr.interpolate(self.vt.unsqueeze(0), rast, self.ft, rast_db=rast_db, diff_attrs='all')
        if use_normal:
            image = torch.sigmoid(dr.texture(self.normal_tex.unsqueeze(0), texc, uv_da=texc_db)) # [1, H, W, 3]
        elif use_material:
            image = torch.sigmoid(dr.texture(self.mr_albedo.unsqueeze(0), texc, uv_da=texc_db)) # [1, H, W, 3]
        else:
            image = torch.sigmoid(dr.texture(self.albedo.unsqueeze(0), texc, uv_da=texc_db)) # [1, H, W, 3]

        image = image.view(1, h, w, 3)
        # image = dr.antialias(image, rast, v_clip, f).clamp(0, 1)
        image = image.squeeze(0).permute(2, 0, 1).contiguous() # [3, H, W]
        if use_normal:
            bg = torch.tensor([0.5, 0.5, 1.0], device=image.device).view(3, 1, 1)
            image = alpha * image + (1 - alpha) * bg
        else:
            image = alpha * image + (1 - alpha)

        return image, alpha
    
    # uv mesh refine
    def fit_mesh_uv(self, iters=1024, resolution=512, texture_resolution=1024, padding=2):

        if self.opt.use_material:
            _, self.gaussians, self.mr_gaussians, pred_normal_world = self.model.forward_gaussians(
                self.input_data, self.octree_in, condition = self.text_embedding, data = None, ema = True
            )
        else:
            _, self.gaussians, _, pred_normal_world = self.model.forward_gaussians(
                self.input_data, self.octree_in, condition = self.text_embedding, data = None, ema = True
            )
            self.mr_gaussians = None

        if self.enable_normal_head and pred_normal_world is not None:
            normal_colors = (pred_normal_world * 0.5 + 0.5).clamp(0, 1)
            self.normal_gaussians = torch.cat([self.gaussians[:, :11], normal_colors], dim=-1)

        self.opt.output_size = resolution

        v = self.mesh.vertices.astype(np.float32)
        f = self.mesh.faces.astype(np.int32)

        self.v = torch.from_numpy(v).contiguous().float().to(self.device)
        self.f = torch.from_numpy(f).contiguous().int().to(self.device)

        # unwrap uv
        print(f"[INFO] uv unwrapping...")
        mesh = Mesh(v=self.v, f=self.f, albedo=None, device=self.device)
        mesh.auto_normal()
        mesh.auto_uv()

        self.vt = mesh.vt
        self.ft = mesh.ft

        # render uv maps
        h = w = texture_resolution
        uv = mesh.vt * 2.0 - 1.0 # uvs to range [-1, 1]
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

        rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), mesh.ft, (h, w)) # [1, h, w, 4]
        xyzs, _ = dr.interpolate(mesh.v.unsqueeze(0), rast, mesh.f) # [1, h, w, 3]
        mask, _ = dr.interpolate(torch.ones_like(mesh.v[:, :1]).unsqueeze(0), rast, mesh.f) # [1, h, w, 1]

        # masked query 
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)
        
        albedo = torch.zeros(h * w, 3, device=self.device, dtype=torch.float32)
        
        albedo = albedo.view(h, w, -1)
        mask = mask.view(h, w)
        albedo = uv_padding(albedo, mask, padding)

        if self.opt.use_material:
            mr_albedo = torch.zeros(h * w, 3, device=self.device, dtype=torch.float32)
        
            mr_albedo = mr_albedo.view(h, w, -1)
            mask = mask.view(h, w)
            mr_albedo = uv_padding(mr_albedo, mask, padding)

        # optimize texture
        self.albedo = nn.Parameter(albedo).to(self.device)

        if self.opt.use_material:
            self.mr_albedo = nn.Parameter(mr_albedo).to(self.device)

        if self.enable_normal_head:
            normal_tex = torch.zeros(h * w, 3, device=self.device, dtype=torch.float32)
            normal_tex = normal_tex.view(h, w, -1)
            normal_tex[..., 0] = 0.5
            normal_tex[..., 1] = 0.5
            normal_tex[..., 2] = 1.0
            normal_tex = uv_padding(normal_tex, mask, padding)
            self.normal_tex = nn.Parameter(normal_tex).to(self.device)
        
        optimizer = torch.optim.Adam([
            {'params': self.albedo, 'lr': 1e-1},
        ])

        if self.opt.use_material:
            mr_optimizer = torch.optim.Adam([
                {'params': self.mr_albedo, 'lr': 1e-3},
            ])
        if self.enable_normal_head:
            normal_optimizer = torch.optim.Adam([
                {'params': self.normal_tex, 'lr': 1e-2},
            ])

        vers = [-89, 89, 0, 0, 0, 0]
        hors = [0, 0, -90, 0, 90, 180]

        rad = self.opt.texture_cam_radius # np.random.uniform(1, 2)

        for (ver, hor) in zip(vers, hors):

            print(f"[INFO] fitting mesh albedo...")
            pbar = tqdm.trange(iters)

            for i in pbar:
                
                pose = orbit_camera(ver, hor, rad)
                
                image_gt, alpha_gt = self.render_gs(pose)
                image_pred, alpha_pred = self.render_mesh(pose)

                if self.opt.save_image:
                    image_gt_save = image_gt.detach().cpu().numpy()
                    image_gt_save = image_gt_save.transpose(1, 2, 0)
                    kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/albedo_gt_images/{i}.jpg', image_gt_save)

                    image_pred_save = image_pred.detach().cpu().numpy()
                    image_pred_save = image_pred_save.transpose(1, 2, 0)
                    kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/mesh_albedo_images/{i}.jpg', image_pred_save)

                loss_mse = F.mse_loss(image_pred, image_gt)
                loss = loss_mse

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                pbar.set_description(f"MSE = {loss_mse.item():.6f}")

                if self.enable_normal_head and hasattr(self, "normal_gaussians"):
                    normal_gt, _ = self.render_gs(pose, use_normal=True)
                    normal_pred, _ = self.render_mesh(pose, use_normal=True)
                    normal_loss = F.mse_loss(normal_pred, normal_gt)
                    normal_loss.backward()
                    normal_optimizer.step()
                    normal_optimizer.zero_grad()
        
        pbar = tqdm.trange(iters * 2)

        for i in pbar:

            # shrink to front view as we care more about it...
            ver = np.random.randint(-89, 89)
            hor = np.random.randint(-180, 180)
            
            pose = orbit_camera(ver, hor, rad)
            
            image_gt, alpha_gt = self.render_gs(pose)
            image_pred, alpha_pred = self.render_mesh(pose)

            if self.opt.save_image:
                image_gt_save = image_gt.detach().cpu().numpy()
                image_gt_save = image_gt_save.transpose(1, 2, 0)
                kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/albedo_gt_images/{i}.jpg', image_gt_save)

                image_pred_save = image_pred.detach().cpu().numpy()
                image_pred_save = image_pred_save.transpose(1, 2, 0)
                kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/mesh_albedo_images/{i}.jpg', image_pred_save)

            loss_mse = F.mse_loss(image_pred, image_gt)
            loss = loss_mse

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"MSE = {loss_mse.item():.6f}")

            if self.enable_normal_head and hasattr(self, "normal_gaussians"):
                normal_gt, _ = self.render_gs(pose, use_normal=True)
                normal_pred, _ = self.render_mesh(pose, use_normal=True)
                normal_loss = F.mse_loss(normal_pred, normal_gt)
                normal_loss.backward()
                normal_optimizer.step()
                normal_optimizer.zero_grad()
        
        print(f"[INFO] finished fitting mesh albedo!")

        if self.enable_normal_head and hasattr(self, "normal_tex"):
            self.normal_world_map = torch.sigmoid(self.normal_tex).detach().clamp(0, 1).cpu().numpy()
    
        if self.opt.use_material:

            for (ver, hor) in zip(vers, hors):

                print(f"[INFO] fitting mesh material...")
                pbar = tqdm.trange(iters)

                for i in pbar:
                    
                    pose = orbit_camera(ver, hor, rad)
                    
                    image_gt, alpha_gt = self.render_gs(pose, use_material = True)
                    image_pred, alpha_pred = self.render_mesh(pose, use_material = True)

                    if self.opt.save_image:
                        image_gt_save = image_gt.detach().cpu().numpy()
                        image_gt_save = image_gt_save.transpose(1, 2, 0)
                        kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/material_gt_images/{i}.jpg', image_gt_save)

                        image_pred_save = image_pred.detach().cpu().numpy()
                        image_pred_save = image_pred_save.transpose(1, 2, 0)
                        kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/mesh_material_images/{i}.jpg', image_pred_save)

                    loss_mse = F.mse_loss(image_pred, image_gt)
                    loss = loss_mse

                    loss.backward()

                    mr_optimizer.step()
                    mr_optimizer.zero_grad()

                    pbar.set_description(f"MSE = {loss_mse.item():.6f}")
        
            pbar = tqdm.trange(iters * 2)

            for i in pbar:

                # shrink to front view as we care more about it...
                ver = np.random.randint(-89, 89)
                hor = np.random.randint(-180, 180)
                
                pose = orbit_camera(ver, hor, rad)
                
                image_gt, alpha_gt = self.render_gs(pose, use_material = True)
                image_pred, alpha_pred = self.render_mesh(pose, use_material = True)

                if self.opt.save_image:
                    image_gt_save = image_gt.detach().cpu().numpy()
                    image_gt_save = image_gt_save.transpose(1, 2, 0)
                    kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/material_gt_images/{i}.jpg', image_gt_save)

                    image_pred_save = image_pred.detach().cpu().numpy()
                    image_pred_save = image_pred_save.transpose(1, 2, 0)
                    kiui.write_image(f'{self.opt.output_dir}/{self.opt.texture_name}/mesh_material_images/{i}.jpg', image_pred_save)

                loss_mse = F.mse_loss(image_pred, image_gt)
                loss = loss_mse

                loss.backward()

                mr_optimizer.step()
                mr_optimizer.zero_grad()

                pbar.set_description(f"MSE = {loss_mse.item():.6f}")
            
            print(f"[INFO] finished fitting mesh material!")


    @torch.no_grad()
    def export_mesh(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)

        v = self.mesh.vertices.astype(np.float32)

        self.v = torch.from_numpy(v).contiguous().float().to(self.device)
        
        # Export a single mesh (geometry + UVs only) and write individual PBR textures.
        mesh = Mesh(v=self.v, f=self.f, vt=self.vt, ft=self.ft, albedo=None, device=self.device)
        mesh.auto_normal()
        mesh_path = os.path.join(save_dir, 'mesh.obj')
        mesh.write(mesh_path)
        # Remove auxiliary files emitted by the writer (mtl / baked albedo).
        mtl_path = os.path.splitext(mesh_path)[0] + '.mtl'
        aux_albedo = os.path.join(save_dir, 'mesh_albedo.png')
        for p in (mtl_path, aux_albedo):
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        albedo_img = torch.sigmoid(self.albedo).detach().clamp(0, 1).cpu().numpy()
        kiui.write_image(os.path.join(save_dir, 'albedo.png'), albedo_img)

        if hasattr(self, "normal_world_map"):
            try:
                save_normal_map(
                    mesh,
                    self.normal_world_map,
                    os.path.join(save_dir, 'normal.png'),
                    flip_green=getattr(self.opt, "flip_normal_green", False),
                )
            except Exception as e:
                print(f"[WARN] normal map export failed: {e}")

        if self.opt.use_material and hasattr(self, "mr_albedo"):
            mr_img = torch.sigmoid(self.mr_albedo).detach().clamp(0, 1).cpu().numpy()
            roughness = mr_img[..., 1:2]  # G channel
            metallic = mr_img[..., 2:3]   # B channel
            kiui.write_image(os.path.join(save_dir, 'metallic.png'), metallic)
            kiui.write_image(os.path.join(save_dir, 'roughness.png'), roughness)

def load_batch_from_tsv(tsv_path: str, caption_field: str):
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    with open(tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        required_captions = ("caption_short", "caption_long")
        missing = [name for name in required_captions if name not in fieldnames]
        if missing:
            raise ValueError(
                f"TSV missing required caption columns: {', '.join(missing)} "
                f"(available: {', '.join(fieldnames) if fieldnames else '(none)'})"
            )
        if caption_field not in fieldnames:
            raise ValueError(
                f"TSV missing caption_field '{caption_field}' "
                f"(available: {', '.join(fieldnames) if fieldnames else '(none)'})"
            )
        rows = [row for row in reader]

    return rows

def to_jsonable(obj):
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # fallback for types like numpy scalars etc.
    try:
        return obj.item()
    except Exception:
        return str(obj)

def save_experiment_config(exp_dir, opt, processed_samples, skipped_samples=None, manifest_path=None, timing_info=None):
    cfg = {
        "options": to_jsonable(opt),
        "ckpt_path": opt.ckpt_path,
        "tsv_path": os.path.abspath(opt.tsv_path) if opt.tsv_path else None,
        "save_image": opt.save_image,
        "processed_samples": processed_samples,
    }
    if skipped_samples:
        cfg["skipped_samples"] = skipped_samples
    if manifest_path:
        cfg["result_tsv"] = os.path.abspath(manifest_path)
    if timing_info:
        cfg["timing"] = timing_info

    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)


def build_result_row(
    obj_id: str,
    sample_dir: str,
    caption_short: str = None,
    caption_long: str = None,
    caption_used: str = None,
):
    """Collect generated asset paths for a single sample into a TSV-ready dict."""
    sample_dir = os.path.abspath(sample_dir)

    def path_if_exists(name: str) -> str:
        p = os.path.join(sample_dir, name)
        return os.path.abspath(p) if os.path.exists(p) else ""

    row = {
        "obj_id": obj_id,
        "mesh": path_if_exists("mesh.obj"),
        "albedo": path_if_exists("albedo.png"),
        "rough": path_if_exists("roughness.png"),
        "metal": path_if_exists("metallic.png"),
        "normal": path_if_exists("normal.png"),
    }
    if caption_short is not None:
        row["caption_short"] = caption_short
    if caption_long is not None:
        row["caption_long"] = caption_long
    if caption_used is not None:
        row["caption_used"] = caption_used
    return row


def write_result_manifest(tsv_path: str, rows):
    """Write generated asset info to a TSV following split_dataset-style columns."""
    if not rows:
        print(f"[WARN] No rows to write for manifest {tsv_path}")
        return

    fieldnames = ["obj_id", "mesh", "albedo", "rough", "metal", "normal"]
    caption_fields = ["caption_short", "caption_long", "caption_used"]
    for name in caption_fields:
        if any(name in r for r in rows):
            fieldnames.append(name)

    tsv_dir = os.path.dirname(tsv_path) or "."
    os.makedirs(tsv_dir, exist_ok=True)

    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"[INFO] Saved generated manifest to {tsv_path}")


def parse_gpu_ids(gpu_ids_str: str):
    """Parse GPU IDs from comma-separated string."""
    gpu_ids_str = gpu_ids_str.strip()
    # Handle both formats: "0,1,2" and "[0,1,2]"
    if gpu_ids_str.startswith('[') and gpu_ids_str.endswith(']'):
        gpu_ids_str = gpu_ids_str[1:-1]
    return [int(x.strip()) for x in gpu_ids_str.split(',') if x.strip()]


def estimate_workers_per_gpu(gpu_id: int, model_memory_gb: float = 6.0, safety_margin: float = 0.85):
    """Estimate optimal number of workers per GPU based on available memory.
    
    Args:
        gpu_id: GPU device ID to query
        model_memory_gb: Estimated memory per worker in GB (model + data)
        safety_margin: Fraction of GPU memory to use (0.85 = 85%)
    
    Returns:
        Recommended number of workers for this GPU
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 1
        
        # Query GPU memory
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        total_memory_gb = total_memory / (1024 ** 3)
        
        # Calculate available memory with safety margin
        available_gb = total_memory_gb * safety_margin
        
        # Estimate workers (at least 1, at most 4 to avoid diminishing returns)
        estimated = int(available_gb / model_memory_gb)
        workers = max(1, min(estimated, 4))
        
        return workers, total_memory_gb
    except Exception as e:
        print(f"[WARN] Failed to query GPU {gpu_id} memory: {e}")
        return 1, 0


def calculate_workers_per_gpu(gpu_ids: list, workers_per_gpu_str: str):
    """Calculate workers per GPU from string config.
    
    Args:
        gpu_ids: List of GPU IDs to use
        workers_per_gpu_str: Either 'auto' or a number string
    
    Returns:
        Integer number of workers per GPU
    """
    workers_per_gpu_str = str(workers_per_gpu_str).strip().lower()
    
    if workers_per_gpu_str == 'auto':
        # Auto-detect based on GPU memory
        if not gpu_ids:
            return 1
        
        # Query first GPU to estimate (assume all GPUs are similar)
        try:
            import torch
            # Temporarily set visible device to query
            first_gpu = gpu_ids[0]
            workers, total_mem = estimate_workers_per_gpu(first_gpu)
            print(f"[INFO] Auto-detected GPU memory: {total_mem:.1f} GB")
            print(f"[INFO] Auto-calculated workers_per_gpu: {workers}")
            return workers
        except Exception as e:
            print(f"[WARN] Auto-detection failed, using default: {e}")
            return 2  # Safe default
    else:
        # Manual specification
        try:
            return max(1, int(workers_per_gpu_str))
        except ValueError:
            print(f"[WARN] Invalid workers_per_gpu '{workers_per_gpu_str}', using default 1")
            return 1


def run_single_gpu_worker(gpu_id: int, worker_id: int, rows_subset: list, opt, tsv_dir: str, textures_dir: str):
    """Run inference on a single GPU for a subset of samples.
    
    This function should be called after CUDA_VISIBLE_DEVICES is set.
    
    Args:
        gpu_id: Physical GPU ID
        worker_id: Local worker ID on this GPU (0, 1, 2, ...)
        rows_subset: List of (global_idx, row) tuples to process
        opt: Options object
        tsv_dir: Directory containing the TSV file
        textures_dir: Output directory for textures
    """
    import torch
    
    worker_tag = f"[GPU {gpu_id} W{worker_id}]"
    
    # Verify we're on the correct GPU
    if torch.cuda.is_available():
        print(f"{worker_tag} CUDA available, device count: {torch.cuda.device_count()}")
        print(f"{worker_tag} Current device: {torch.cuda.current_device()}")
        print(f"{worker_tag} Device name: {torch.cuda.get_device_name(0)}")
    
    # Create converter for this GPU
    converter = Converter(opt).cuda()
    converter.load_ckpt(opt.ckpt_path)
    
    processed_samples = []
    skipped_samples = []
    
    for local_idx, (global_idx, row) in enumerate(rows_subset):
        mesh_path = (row.get("mesh") or "").strip()
        caption_short = (row.get("caption_short") or "").strip()
        caption_long = (row.get("caption_long") or "").strip()
        caption = (row.get(opt.caption_field) or "").strip()
        obj_id = (row.get("obj_id") or "").strip() or f"sample_{global_idx}"

        if not mesh_path or not caption:
            print(f"{worker_tag} Skip row {global_idx}: missing mesh or caption (obj_id={obj_id})")
            skipped_samples.append({"obj_id": obj_id, "reason": "missing mesh or caption"})
            continue

        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(tsv_dir, mesh_path)

        converter.opt.texture_name = obj_id
        converter.opt.mesh_path = mesh_path
        converter.set_text_prompt(caption)

        sample_output_dir = os.path.join(textures_dir, converter.opt.texture_name)
        os.makedirs(sample_output_dir, exist_ok=True)

        print(f"{worker_tag} Processing {converter.opt.texture_name} ({local_idx + 1}/{len(rows_subset)}, global {global_idx + 1})")
        try:
            converter.load_mesh(mesh_path)
            converter.fit_mesh_uv(iters=1000)
            converter.export_mesh(sample_output_dir)

            processed_samples.append(build_result_row(
                converter.opt.texture_name,
                sample_output_dir,
                caption_short=caption_short,
                caption_long=caption_long,
                caption_used=opt.caption_field,
            ))
        except Exception as e:
            print(f"{worker_tag} Error processing {obj_id}: {e}")
            import traceback
            traceback.print_exc()
            skipped_samples.append({"obj_id": obj_id, "reason": str(e)})
    
    return processed_samples, skipped_samples


def worker_subprocess_entry():
    """Entry point for worker subprocess.
    
    This is called by subprocess with specific environment variables set.
    """
    import os
    import sys
    import json
    import pickle
    
    # Read configuration from environment
    gpu_id = int(os.environ['TEXGAUSSIAN_GPU_ID'])
    worker_id = int(os.environ.get('TEXGAUSSIAN_WORKER_ID', '0'))
    config_file = os.environ['TEXGAUSSIAN_CONFIG_FILE']
    
    worker_tag = f"[GPU {gpu_id} W{worker_id}]"
    print(f"{worker_tag} Starting subprocess...")
    print(f"{worker_tag} CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # Load configuration
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    
    opt_dict = config['opt_dict']
    rows_subset = config['rows_subset']
    tsv_dir = config['tsv_dir']
    textures_dir = config['textures_dir']
    result_file = config['result_file']
    
    # Reconstruct options
    from core.options import Options
    opt = Options(**opt_dict)
    
    # Run the actual processing
    processed_samples, skipped_samples = run_single_gpu_worker(
        gpu_id, worker_id, rows_subset, opt, tsv_dir, textures_dir
    )
    
    # Save results
    results = {
        "gpu_id": gpu_id,
        "worker_id": worker_id,
        "processed": processed_samples,
        "skipped": skipped_samples
    }
    with open(result_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"{worker_tag} Finished. Processed {len(processed_samples)}, skipped {len(skipped_samples)}")


def run_single_gpu(opt, converter, batch_rows, tsv_dir, textures_dir):
    """Run inference on a single GPU (original behavior)."""
    processed_samples = []
    skipped_samples = []

    for idx, row in enumerate(batch_rows):
        mesh_path = (row.get("mesh") or "").strip()
        caption_short = (row.get("caption_short") or "").strip()
        caption_long = (row.get("caption_long") or "").strip()
        caption = (row.get(opt.caption_field) or "").strip()
        obj_id = (row.get("obj_id") or "").strip() or f"sample_{idx}"

        if not mesh_path or not caption:
            print(f"[WARN] Skip row {idx}: missing mesh or caption (obj_id={obj_id})")
            skipped_samples.append({"obj_id": obj_id, "reason": "missing mesh or caption"})
            continue

        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(tsv_dir, mesh_path)

        converter.opt.texture_name = obj_id
        converter.opt.mesh_path = mesh_path
        converter.set_text_prompt(caption)

        sample_output_dir = os.path.join(textures_dir, converter.opt.texture_name)
        os.makedirs(sample_output_dir, exist_ok=True)

        print(f"[INFO] Processing {converter.opt.texture_name} ({idx + 1}/{len(batch_rows)})")
        converter.load_mesh(mesh_path)
        converter.fit_mesh_uv(iters=1000)
        converter.export_mesh(sample_output_dir)

        processed_samples.append(build_result_row(
            converter.opt.texture_name,
            sample_output_dir,
            caption_short=caption_short,
            caption_long=caption_long,
            caption_used=opt.caption_field,
        ))
    
    return processed_samples, skipped_samples


def run_multi_gpu(opt, batch_rows, tsv_dir, textures_dir, gpu_ids, workers_per_gpu=1):
    """Run inference on multiple GPUs in parallel using subprocesses.
    
    Each GPU can run multiple worker processes to maximize GPU utilization.
    This ensures results are identical to single-GPU mode.
    
    Args:
        opt: Options object
        batch_rows: List of sample rows to process
        tsv_dir: Directory containing the TSV file
        textures_dir: Output directory for textures
        gpu_ids: List of GPU IDs to use
        workers_per_gpu: Number of parallel workers per GPU
    """
    import subprocess
    import pickle
    import tempfile
    
    num_gpus = len(gpu_ids)
    num_samples = len(batch_rows)
    total_workers = num_gpus * workers_per_gpu
    
    # Distribute samples across all workers (round-robin)
    # Each worker is identified by (gpu_id, local_worker_id)
    worker_assignments = [[] for _ in range(total_workers)]
    for idx, row in enumerate(batch_rows):
        worker_idx = idx % total_workers
        worker_assignments[worker_idx].append((idx, row))
    
    print(f"[INFO] Distributing {num_samples} samples across {num_gpus} GPUs x {workers_per_gpu} workers = {total_workers} total workers")
    
    # Show distribution per GPU
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        gpu_total = sum(len(worker_assignments[gpu_idx * workers_per_gpu + w]) for w in range(workers_per_gpu))
        print(f"  GPU {gpu_id}: {gpu_total} samples ({workers_per_gpu} workers)")
    
    # Convert options to dict for pickling
    opt_dict = asdict(opt)
    
    # Create temporary directory for config and result files
    temp_dir = tempfile.mkdtemp(prefix="texgaussian_multiGPU_")
    print(f"[INFO] Using temp directory: {temp_dir}")
    
    # Start subprocesses
    processes = []
    result_files = []
    
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        for local_worker_id in range(workers_per_gpu):
            global_worker_id = gpu_idx * workers_per_gpu + local_worker_id
            rows_subset = worker_assignments[global_worker_id]
            
            if not rows_subset:
                continue
            
            # Create config file for this worker
            config_file = os.path.join(temp_dir, f"config_gpu{gpu_id}_worker{local_worker_id}.pkl")
            result_file = os.path.join(temp_dir, f"result_gpu{gpu_id}_worker{local_worker_id}.pkl")
            result_files.append((gpu_id, local_worker_id, result_file))
            
            config = {
                'opt_dict': opt_dict,
                'rows_subset': rows_subset,
                'tsv_dir': tsv_dir,
                'textures_dir': textures_dir,
                'result_file': result_file,
            }
            with open(config_file, 'wb') as f:
                pickle.dump(config, f)
            
            # Set up environment for subprocess
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            env['TEXGAUSSIAN_GPU_ID'] = str(gpu_id)
            env['TEXGAUSSIAN_WORKER_ID'] = str(local_worker_id)
            env['TEXGAUSSIAN_CONFIG_FILE'] = config_file
            
            # Launch subprocess
            cmd = [
                sys.executable,
                '-c',
                'from texture import worker_subprocess_entry; worker_subprocess_entry()'
            ]
            
            print(f"[INFO] Launching subprocess for GPU {gpu_id} Worker {local_worker_id} ({len(rows_subset)} samples)...")
            p = subprocess.Popen(
                cmd,
                env=env,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=None,  # Inherit stdout
                stderr=None,  # Inherit stderr
            )
            processes.append((gpu_id, local_worker_id, p))
    
    # Wait for all subprocesses to complete
    print(f"[INFO] Waiting for {len(processes)} subprocesses to complete...")
    for gpu_id, local_worker_id, p in processes:
        return_code = p.wait()
        if return_code != 0:
            print(f"[WARN] Subprocess for GPU {gpu_id} Worker {local_worker_id} exited with code {return_code}")
    
    # Collect results from all workers
    all_processed = []
    all_skipped = []
    
    for gpu_id, local_worker_id, result_file in result_files:
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
            all_processed.extend(results["processed"])
            all_skipped.extend(results["skipped"])
            print(f"[INFO] Collected results from GPU {gpu_id} Worker {local_worker_id}: {len(results['processed'])} processed, {len(results['skipped'])} skipped")
        else:
            print(f"[WARN] Result file not found for GPU {gpu_id} Worker {local_worker_id}: {result_file}")
    
    # Clean up temp files
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"[WARN] Failed to clean up temp directory: {e}")
    
    # Sort results by obj_id to maintain consistent ordering
    all_processed.sort(key=lambda x: x["obj_id"])
    all_skipped.sort(key=lambda x: x["obj_id"])
    
    return all_processed, all_skipped

if __name__ == "__main__":
    opt = tyro.cli(AllConfigs)

    opt.use_checkpoint = str2bool(opt.use_checkpoint)
    opt.use_material = str2bool(opt.use_material)
    opt.use_text = str2bool(opt.use_text)
    opt.save_image = str2bool(opt.save_image)
    opt.gaussian_loss = str2bool(opt.gaussian_loss)
    opt.use_normal_head = str2bool(opt.use_normal_head)
    opt.use_rotation_head = str2bool(opt.use_rotation_head)
    opt.use_local_pretrained_ckpt = str2bool(opt.use_local_pretrained_ckpt)
    opt.use_longclip = str2bool(opt.use_longclip)

    if opt.tsv_path is None and opt.batch_path is not None:
        opt.tsv_path = opt.batch_path

    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)
    textures_dir = os.path.join(output_dir, "textures") if opt.tsv_path else output_dir

    result_tsv_path = opt.result_tsv
    if result_tsv_path:
        if not os.path.isabs(result_tsv_path):
            result_tsv_path = os.path.abspath(os.path.join(output_dir, result_tsv_path))
    else:
        result_tsv_path = os.path.join(output_dir, "generated_manifest.tsv")
    result_tsv_path = os.path.abspath(result_tsv_path)
    opt.result_tsv = result_tsv_path

    # Parse GPU IDs
    gpu_ids = parse_gpu_ids(opt.gpu_ids)
    num_gpus = min(opt.num_gpus, len(gpu_ids)) if opt.num_gpus > 0 else len(gpu_ids)
    gpu_ids = gpu_ids[:num_gpus]
    
    # Calculate workers per GPU (auto or manual)
    workers_per_gpu = calculate_workers_per_gpu(gpu_ids, opt.workers_per_gpu)
    
    print(f"[INFO] Using {len(gpu_ids)} GPU(s): {gpu_ids}")
    print(f"[INFO] Workers per GPU: {workers_per_gpu}")

    # Record start time for inference timing
    inference_start_time = time.time()
    inference_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] Inference started at: {inference_start_datetime}")

    if opt.tsv_path:
        tsv_dir = os.path.dirname(os.path.abspath(opt.tsv_path))
        batch_rows = load_batch_from_tsv(opt.tsv_path, opt.caption_field)
        total_rows = len(batch_rows)
        
        # Apply max_samples limit if specified
        if opt.max_samples > 0 and opt.max_samples < total_rows:
            batch_rows = batch_rows[:opt.max_samples]
            print(f"[INFO] Loaded {total_rows} rows from {opt.tsv_path}, processing first {opt.max_samples} samples")
        else:
            print(f"[INFO] Loaded {total_rows} rows from {opt.tsv_path}")
        
        print(f"[INFO] Using caption field: {opt.caption_field}")
        os.makedirs(textures_dir, exist_ok=True)

        total_workers = len(gpu_ids) * workers_per_gpu
        if total_workers > 1 and len(batch_rows) > 1:
            # Multi-worker mode (multiple GPUs and/or multiple workers per GPU)
            print(f"[INFO] Running in parallel mode: {len(gpu_ids)} GPUs x {workers_per_gpu} workers = {total_workers} total workers")
            processed_samples, skipped_samples = run_multi_gpu(
                opt, batch_rows, tsv_dir, textures_dir, gpu_ids, workers_per_gpu
            )
        else:
            # Single-GPU single-worker mode
            print(f"[INFO] Running in single-GPU mode")
            converter = Converter(opt).cuda()
            converter.load_ckpt(opt.ckpt_path)
            processed_samples, skipped_samples = run_single_gpu(
                opt, converter, batch_rows, tsv_dir, textures_dir
            )

        if processed_samples:
            write_result_manifest(result_tsv_path, processed_samples)
        
        # Calculate and display timing information
        inference_end_time = time.time()
        inference_end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_time_seconds = inference_end_time - inference_start_time
        total_time_str = str(timedelta(seconds=int(total_time_seconds)))
        num_samples = len(processed_samples) if processed_samples else 0
        avg_time_per_sample = total_time_seconds / num_samples if num_samples > 0 else 0
        
        timing_info = {
            "start_time": inference_start_datetime,
            "end_time": inference_end_datetime,
            "total_seconds": round(total_time_seconds, 2),
            "total_time_formatted": total_time_str,
            "num_samples_processed": num_samples,
            "avg_seconds_per_sample": round(avg_time_per_sample, 2),
            "num_gpus": len(gpu_ids),
            "workers_per_gpu": workers_per_gpu,
            "total_workers": len(gpu_ids) * workers_per_gpu,
        }
        
        print(f"\n" + "="*60)
        print(f"[TIMING] Inference completed!")
        print(f"[TIMING] Start time: {inference_start_datetime}")
        print(f"[TIMING] End time: {inference_end_datetime}")
        print(f"[TIMING] Total time: {total_time_str} ({total_time_seconds:.2f} seconds)")
        print(f"[TIMING] Samples processed: {num_samples}")
        print(f"[TIMING] Average time per sample: {avg_time_per_sample:.2f} seconds")
        print(f"[TIMING] GPUs used: {len(gpu_ids)}, Workers per GPU: {workers_per_gpu}")
        print("="*60 + "\n")
        
        save_experiment_config(output_dir, opt, processed_samples, skipped_samples, 
                               manifest_path=result_tsv_path if processed_samples else None,
                               timing_info=timing_info)
    else:
        # Single sample mode - always single GPU
        converter = Converter(opt).cuda()
        converter.load_ckpt(opt.ckpt_path)
        
        if opt.use_text and opt.text_prompt:
            converter.set_text_prompt(opt.text_prompt)
        converter.load_mesh(opt.mesh_path)
        converter.fit_mesh_uv(iters=1000)
        sample_output_dir = os.path.join(textures_dir, opt.texture_name)
        converter.export_mesh(sample_output_dir)

        # Calculate and display timing information for single sample mode
        inference_end_time = time.time()
        inference_end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_time_seconds = inference_end_time - inference_start_time
        total_time_str = str(timedelta(seconds=int(total_time_seconds)))
        
        timing_info = {
            "start_time": inference_start_datetime,
            "end_time": inference_end_datetime,
            "total_seconds": round(total_time_seconds, 2),
            "total_time_formatted": total_time_str,
            "num_samples_processed": 1,
            "avg_seconds_per_sample": round(total_time_seconds, 2),
            "num_gpus": 1,
            "workers_per_gpu": 1,
            "total_workers": 1,
        }
        
        print(f"\n" + "="*60)
        print(f"[TIMING] Inference completed!")
        print(f"[TIMING] Start time: {inference_start_datetime}")
        print(f"[TIMING] End time: {inference_end_datetime}")
        print(f"[TIMING] Total time: {total_time_str} ({total_time_seconds:.2f} seconds)")
        print("="*60 + "\n")

        processed_samples = [build_result_row(opt.texture_name, sample_output_dir)]
        write_result_manifest(result_tsv_path, processed_samples)
        save_experiment_config(output_dir, opt, processed_samples, manifest_path=result_tsv_path, timing_info=timing_info)
