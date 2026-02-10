import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import copy

import ocnn

from kiui.lpips import LPIPS
import open3d as o3d

# from core.unet import UNet
from core.octree_unet import OctreeUNet
from core.options import Options
from core.gs import GaussianRenderer
from ocnn.octree import Octree, Points
from core.utils import *
from clip_networks.network import CLIPTextEncoder
from core.longclip_utils import load_longclip_model


class LongCLIPTextEncoder(nn.Module):
    def __init__(self, model_path, context_length=248, device='cuda'):
        super().__init__()
        self.model, _ = load_longclip_model(model_path, device)
        if hasattr(self.model, "visual"):
            del self.model.visual
        self.model.requires_grad_(False)
        self.model.eval()
        self.context_length = context_length

    def encode(self, text):
        x = self.model.token_embedding(text)
        x = x + self.model.positional_embedding[:x.shape[1]]
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x)
        return x

class TexGaussian(nn.Module):
    def __init__(self, opt, device):
        super().__init__()

        self.opt = opt
        self.device = device

        if self.opt.gaussian_loss:
            self.gaussian_mean = torch.load(self.opt.mean_path).to(self.device)
            self.gaussian_std = torch.load(self.opt.std_path).to(self.device)

        self.use_longclip = self.opt.use_longclip
        if isinstance(self.use_longclip, str):
            self.use_longclip = self.use_longclip.lower() in ('yes', 'true', 't', 'y', '1')

        if self.opt.use_text:
            if self.use_longclip:
                self.text_encoder = LongCLIPTextEncoder(
                    model_path=self.opt.longclip_model,
                    context_length=self.opt.longclip_context_length,
                    device=self.device,
                )
            else:
                if self.opt.use_local_pretrained_ckpt:
                    self.text_encoder = CLIPTextEncoder(model="./ViT-L-14.pt")
                else:
                    self.text_encoder = CLIPTextEncoder(model="ViT-L/14")
            self.text_encoder.to(self.device)

            self.text_encoder.requires_grad_(False)

            self.text_encoder.eval()

        self.use_normal_head = self.opt.use_normal_head
        if isinstance(self.use_normal_head, str):
            self.use_normal_head = self.use_normal_head.lower() in ('yes', 'true', 't', 'y', '1')
        self.use_rotation_head = self.opt.use_rotation_head
        if isinstance(self.use_rotation_head, str):
            self.use_rotation_head = self.use_rotation_head.lower() in ('yes', 'true', 't', 'y', '1')
        self.lambda_geo_normal = self.opt.lambda_geo_normal
        self.lambda_tex_normal = self.opt.lambda_tex_normal
        self.normal_residual_scale = 0.05
        self.rotation_residual_scale = 0.05

        if self.opt.use_material:
            self.opt.out_channels += 3

        if self.opt.input_feature == 'L':
            self.opt.in_channels = 3

        elif self.opt.input_feature == 'ND':
            self.opt.in_channels = 4

        self.model = OctreeUNet(
            in_channels = self.opt.in_channels,
            out_channels = self.opt.out_channels,
            model_channels = self.opt.model_channels,
            channel_mult=self.opt.channel_mult,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_attention=self.opt.up_attention,
            num_heads = self.opt.num_heads,
            context_dim = self.opt.context_dim,
            use_checkpoint = self.opt.use_checkpoint,
        )

        # ema
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.to(self.device)

        self.ema_rate = self.opt.ema_rate
        self.ema_updater = EMA(self.ema_rate)
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)

        if self.use_normal_head or self.use_rotation_head:
            normal_in = None
            if hasattr(self.model, "conv_out"):
                if hasattr(self.model.conv_out, "in_channels"):
                    normal_in = self.model.conv_out.in_channels
                elif hasattr(self.model.conv_out, "weight"):
                    normal_in = self.model.conv_out.weight.shape[1]
            if normal_in is None:
                raise RuntimeError("Cannot infer normal_head input channels from model.")

            if self.use_normal_head:
                self.normal_head = ocnn.nn.OctreeConv(
                    normal_in, 3, kernel_size=[3], nempty=True, use_bias=True
                )
                self._init_head(self.normal_head)
            if self.use_rotation_head:
                self.rotation_head = ocnn.nn.OctreeConv(
                    normal_in, 4, kernel_size=[3], nempty=True, use_bias=True
                )
                self._init_head(self.rotation_head)

        self.register_buffer("quat_identity", torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32))

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.01 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x)   # [0, 1]
        if self.opt.use_material:
            self.mr_act = lambda x: torch.sigmoid(x)
        if self.use_normal_head:
            self.normal_act = lambda x: F.normalize(x, dim=-1)

        self.input_depth = self.opt.input_depth
        self.full_depth = self.opt.full_depth

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            if self.opt.use_local_pretrained_ckpt:
                self.lpips_loss = LPIPS(net='vgg', use_url=False, checkpoint_dir = './lpips_checkpoints')
            else:
                self.lpips_loss = LPIPS(net ='vgg')
            self.lpips_loss.requires_grad_(False)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def _init_head(self, head, bias_value=None):
        for _, p in head.named_parameters():
            if p.dim() > 1:
                nn.init.zeros_(p)
            else:
                nn.init.zeros_(p)
        if bias_value is not None:
            ref_param = next(head.parameters())
            bias_tensor = torch.tensor(bias_value, device=ref_param.device, dtype=ref_param.dtype)
            for _, p in head.named_parameters():
                if p.dim() == 1 and p.numel() == bias_tensor.numel():
                    p.data.copy_(bias_tensor)
                    break

    def quat_multiply(self, q1, q2):
        r1, x1, y1, z1 = q1.unbind(dim=-1)
        r2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack([
            r1 * r2 - x1 * x2 - y1 * y2 - z1 * z2,
            r1 * x2 + x1 * r2 + y1 * z2 - z1 * y2,
            r1 * y2 - x1 * z2 + y1 * r2 + z1 * x2,
            r1 * z2 + x1 * y2 - y1 * x2 + z1 * r2,
        ], dim=-1)

    def quat_from_two_vectors(self, src, dst):
        src = F.normalize(src, dim=-1, eps=1e-6)
        dst = F.normalize(dst, dim=-1, eps=1e-6)

        cross = torch.cross(src, dst, dim=-1)
        dot = (src * dst).sum(dim=-1, keepdim=True)
        w = 1.0 + dot
        quat = torch.cat([w, cross], dim=-1)

        opposite = (w.squeeze(-1) < 1e-6)
        if opposite.any():
            ref = torch.zeros_like(src)
            ref[:, 0] = 1.0
            use_y = src[:, 0].abs() > 0.9
            ref[use_y, 0] = 0.0
            ref[use_y, 1] = 1.0
            ortho = torch.cross(src, ref, dim=-1)
            ortho = F.normalize(ortho, dim=-1, eps=1e-6)
            quat_opposite = torch.cat([torch.zeros_like(w), ortho], dim=-1)
            quat = torch.where(opposite.unsqueeze(-1), quat_opposite, quat)

        return F.normalize(quat, dim=-1, eps=1e-6)

    def normal_to_shortest_axis_quat(self, normals, scales):
        shortest_idx = torch.argmin(scales, dim=-1)
        axis_table = torch.eye(3, device=normals.device, dtype=normals.dtype)
        src_axis = axis_table[shortest_idx]
        return self.quat_from_two_vectors(src_axis, normals)

    def forward_gaussians(self, x, octree, condition = None, data = None, ema = False):
        # x: [N, 4]
        # return: Gaussians: [N, dim_t]
        input_feature = x
        mesh_normal_prior = None
        if input_feature.shape[1] >= 3:
            mesh_normal_prior = F.normalize(input_feature[:, :3], dim=-1, eps=1e-6)

        use_extra_heads = self.use_normal_head or self.use_rotation_head
        if ema:
            if use_extra_heads:
                base_out, feat = self.ema_model(input_feature, octree, condition, return_features=True)
            else:
                base_out = self.ema_model(input_feature, octree, condition)
        else:
            if use_extra_heads:
                base_out, feat = self.model(input_feature, octree, condition, return_features=True)
            else:
                base_out = self.model(input_feature, octree, condition) # [N, out_channels]

        pred_normal_world = None
        rotation_raw = None
        if use_extra_heads:
            depth = octree.depth
            if self.use_normal_head:
                normal_raw = self.normal_head(feat, octree, depth)
                if mesh_normal_prior is not None:
                    pred_normal_world = self.normal_act(
                        mesh_normal_prior + self.normal_residual_scale * normal_raw
                    )
                else:
                    pred_normal_world = self.normal_act(normal_raw)
            if self.use_rotation_head:
                rotation_raw = self.rotation_head(feat, octree, depth)

        if self.opt.gaussian_loss:
            gaussian_loss = F.mse_loss(base_out, data['gaussian'])
            zeros = base_out.new_zeros([base_out.shape[0], 4])
            gaussian_params = torch.cat([base_out[:, :4], zeros, base_out[:, 4:]], dim=1)
            gaussian_params = gaussian_params * self.gaussian_std + self.gaussian_mean
        else:
            gaussian_loss = torch.zeros(1, device = self.device)
            zeros = base_out.new_zeros([base_out.shape[0], 4])
            gaussian_params = torch.cat([base_out[:, :4], zeros, base_out[:, 4:]], dim=1)
        if rotation_raw is not None:
            if mesh_normal_prior is not None:
                scales_prior = self.scale_act(gaussian_params[:, 1:4])
                base_quat = self.normal_to_shortest_axis_quat(mesh_normal_prior, scales_prior)
                quat_identity = self.quat_identity.unsqueeze(0).to(
                    device=rotation_raw.device, dtype=rotation_raw.dtype
                )
                delta_quat = self.rot_act(
                    self.rotation_residual_scale * rotation_raw + quat_identity
                )
                gaussian_params[:, 4:8] = self.quat_multiply(delta_quat, base_quat)
            else:
                gaussian_params[:, 4:8] = rotation_raw

        pos = self.pos_act(octree.position) # [N, 3]
        opacity = self.opacity_act(gaussian_params[:, :1]) # [N, 1]
        scale = self.scale_act(gaussian_params[:, 1:4]) # [N, 3]
        rotation = self.rot_act(gaussian_params[:, 4:8]) # [N, 4]
        rgbs = self.rgb_act(gaussian_params[:, 8:11]) # [N, 3]
        if self.opt.use_material:
            mr = self.mr_act(gaussian_params[:, 11:14]) # [N, 3]

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [N, 14]
        if self.opt.use_material:
            mr_gaussians = torch.cat([pos, opacity, scale, rotation, mr], dim=-1) # [N, 14]
            return gaussian_loss, gaussians, mr_gaussians, pred_normal_world

        else:
            return gaussian_loss, gaussians, None, pred_normal_world


    def set_input(self, input=None):
        def points2octree(points):
            octree = ocnn.octree.Octree(depth = self.input_depth, full_depth = self.full_depth)
            octree.build_octree(points)
            return octree

        points = []

        for pts, norms in zip(input['points'], input['normals']):
            points.append(Points(points = pts.float(),normals = norms.float()))

        points = [pts.cuda(non_blocking=True) for pts in points]
        octrees = [points2octree(pts) for pts in points]
        octree = ocnn.octree.merge_octrees(octrees)
        octree.construct_all_neigh()

        xyzb = octree.xyzb(depth = octree.depth, nempty = True)
        x, y, z, b = xyzb
        xyz = torch.stack([x,y,z], dim = 1)
        octree.position = 2 * xyz / (2 ** octree.depth) - 1

        input['octree'] = octree

        if self.opt.gaussian_loss:
            input['gaussian'] = (input['gaussian'] - self.gaussian_mean) / self.gaussian_std
            input['gaussian'] = torch.cat([input['gaussian'][:,:4], input['gaussian'][:,8:]], dim = 1)

        if self.opt.use_text:
            text_embeds = self.text_encoder.encode(input['token']).float()
            input['text_embedding'] = text_embeds  # [bs, 77, 768]

    def forward(self, data, ema = False):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        self.set_input(data)

        octree = data['octree']

        condition = None

        if self.opt.use_text:
            condition = data['text_embedding']  # [bs, 77, 768]

        input_feature = octree.get_input_feature(feature = self.opt.input_feature, nempty = True)
        aligned_gt_mesh_normal = None
        if input_feature.shape[1] >= 3:
            aligned_gt_mesh_normal = input_feature[:, :3]
        gaussian_loss, gaussians, mr_gaussians, pred_normal_world = self.forward_gaussians(
            input_feature, octree, condition, data, ema = ema) # [N, 14]
        batch_id = octree.batch_id(self.opt.input_depth, nempty = True)

        loss += gaussian_loss

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, batch_id, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        if self.opt.use_material:
            mr_results = self.gs.render(mr_gaussians, batch_id, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
            mr_pred_images = mr_results['image']

        results['images_pred'] = pred_images
        if self.opt.use_material:
            results['mr_images_pred'] = mr_pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        if self.opt.use_material:
            mr_gt_images = data['mr_images_output']
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)  # [B, V, 3, output_size, output_size]
        if self.opt.use_material:
            mr_gt_images = mr_gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        albedo_loss = F.mse_loss(pred_images, gt_images)
        if self.opt.use_material:
            mr_loss = F.mse_loss(mr_pred_images, mr_gt_images)

        mask_loss = F.mse_loss(pred_alphas, gt_masks)

        if self.opt.use_material:
            loss_mse = albedo_loss + mr_loss + mask_loss

        else:
            loss_mse = albedo_loss + mask_loss

        results['albedo_loss'] = albedo_loss.item()
        if self.opt.use_material:
            results['mr_loss'] = mr_loss.item()
        results['mask_loss'] = mask_loss.item()
        results['gaussian_loss'] = gaussian_loss.item()

        loss = loss + loss_mse

        if self.use_normal_head or self.use_rotation_head:
            normal_losses = self.compute_normal_losses(
                gaussians, pred_normal_world, aligned_gt_mesh_normal, data, batch_id
            )
            loss_geo = normal_losses['normal_geo_loss']
            loss_tex = normal_losses['normal_tex_loss']
            if self.use_rotation_head:
                loss = loss + self.lambda_geo_normal * loss_geo
            if self.use_normal_head and pred_normal_world is not None:
                loss = loss + self.lambda_tex_normal * loss_tex
            results['normal_geo_loss'] = loss_geo.item()
            results['normal_tex_loss'] = loss_tex.item()
        else:
            results['normal_geo_loss'] = 0.0
            results['normal_tex_loss'] = 0.0

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            loss = loss + self.opt.lambda_lpips * loss_lpips
            if self.opt.use_material:
                mr_loss_lpips = self.lpips_loss(
                    F.interpolate(mr_gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                    F.interpolate(mr_pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                ).mean()
                loss = loss + self.opt.lambda_lpips * mr_loss_lpips

        results['lpips_loss'] = self.opt.lambda_lpips * loss_lpips.item()

        if self.opt.use_material:
            results['mr_lpips_loss'] = self.opt.lambda_lpips * mr_loss_lpips.item()

        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

            if self.opt.use_material:
                mr_psnr = -10 * torch.log10(torch.mean((mr_pred_images.detach() - mr_gt_images) ** 2))
                results['mr_psnr'] = mr_psnr

        del octree

        return results

    def get_shortest_axis(self, quaternions, scales):
        q = F.normalize(quaternions, dim=-1, eps=1e-6)
        r, x, y, z = q.unbind(dim=-1)
        r = r.unsqueeze(-1)
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        z = z.unsqueeze(-1)
        row0 = torch.cat([1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)], dim=-1)
        row1 = torch.cat([2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)], dim=-1)
        row2 = torch.cat([2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], dim=-1)
        rot = torch.stack([row0, row1, row2], dim=-2)  # [N, 3, 3]
        idx = torch.argmin(scales, dim=-1)
        idx = idx.view(-1, 1, 1).expand(-1, 3, 1)
        axis = torch.gather(rot, 2, idx).squeeze(-1)
        return F.normalize(axis, dim=-1, eps=1e-6)

    def compute_normal_losses(self, gaussians, pred_normal_world, aligned_gt_mesh_normal, data, batch_id):
        device = gaussians.device
        loss_geo = torch.zeros(1, device=device)
        loss_tex = torch.zeros(1, device=device)

        if self.use_rotation_head and aligned_gt_mesh_normal is not None:
            scales = gaussians[:, 4:7]
            quats = gaussians[:, 7:11]
            axis = self.get_shortest_axis(quats, scales)
            target = F.normalize(aligned_gt_mesh_normal, dim=-1, eps=1e-6)
            loss_geo = (1 - F.cosine_similarity(axis, target, dim=-1)).mean()

        if self.use_normal_head and pred_normal_world is not None and 'gt_normal_map' in data:
            normal_colors = (pred_normal_world * 0.5 + 0.5).clamp(0, 1)
            normal_gaussians = torch.cat([gaussians[:, :11], normal_colors], dim=-1)
            bg_color = torch.tensor([0.5, 0.5, 1.0], device=device, dtype=torch.float32)
            normal_results = self.gs.render(
                normal_gaussians,
                batch_id,
                data['cam_view'],
                data['cam_view_proj'],
                data['cam_pos'],
                bg_color=bg_color,
            )
            pred_normal_img = normal_results['image']
            gt_normal_img = data['gt_normal_map']
            mask = data.get('masks_output', None)
            if mask is None:
                mask = torch.ones_like(pred_normal_img[:, :, :1, ...])

            l1 = torch.abs(pred_normal_img - gt_normal_img)
            l1 = (l1 * mask).sum() / (mask.sum() * 3 + 1e-6)

            pred_unit = F.normalize(pred_normal_img * 2 - 1, dim=2, eps=1e-6)
            gt_unit = F.normalize(gt_normal_img * 2 - 1, dim=2, eps=1e-6)
            cos = 1 - (pred_unit * gt_unit).sum(dim=2, keepdim=True)
            cos = (cos * mask).sum() / (mask.sum() + 1e-6)
            loss_tex = l1 + cos

        return {'normal_geo_loss': loss_geo, 'normal_tex_loss': loss_tex}
