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


class OctreeHead(nn.Module):
    """
    Two-layer OctreeConv head with GroupNorm + SiLU nonlinearity.
    
    Much more expressive than a single linear projection for predicting
    geometric quantities like normals and rotations.
    
    Architecture: OctreeConv(in→mid) → GroupNorm → SiLU → OctreeConv(mid→out)
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, groups: int = 32):
        super().__init__()
        # Clamp groups to not exceed mid_channels
        groups = min(groups, mid_channels)
        self.conv1 = ocnn.nn.OctreeConv(in_channels, mid_channels, kernel_size=[3], nempty=True, use_bias=True)
        self.norm = ocnn.nn.OctreeGroupNorm(in_channels=mid_channels, group=groups, nempty=True)
        self.conv2 = ocnn.nn.OctreeConv(mid_channels, out_channels, kernel_size=[3], nempty=True, use_bias=True)
    
    def forward(self, data, octree, depth):
        x = self.conv1(data, octree, depth)
        x = self.norm(x, octree, depth)
        x = F.silu(x)
        x = self.conv2(x, octree, depth)
        return x


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


class TextAdapter(nn.Module):
    """
    Lightweight adapter to bridge LongCLIP text embeddings to the pretrained model.
    
    When switching from CLIP (77 tokens) to LongCLIP (248 tokens), the feature 
    distribution may differ. This adapter learns to:
    1. Adapt token-level features via a residual MLP
    2. Optionally pool/reweight long sequences
    
    Args:
        embed_dim: Text embedding dimension (768 for CLIP/LongCLIP ViT-L)
        hidden_dim: Hidden dimension for the adapter MLP
        num_layers: Number of adapter layers
        dropout: Dropout rate
        residual_scale: Initial scale for residual connection (start near identity)
    """
    def __init__(
        self, 
        embed_dim: int = 768, 
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.residual_scale = residual_scale
        
        # Token-level adapter (applied to each token independently)
        layers = []
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else embed_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.adapter = nn.Sequential(*layers)
        
        # Learnable scale parameter (initialized small for stable training)
        self.scale = nn.Parameter(torch.tensor(residual_scale))
        
        # Initialize to near-identity
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights so adapter starts as near-identity mapping.
        
        Intermediate layers use Xavier init (break symmetry, allow gradient flow),
        only the final layer is zero-initialized so output starts at zero.
        """
        linear_layers = [m for m in self.adapter if isinstance(m, nn.Linear)]
        for i, module in enumerate(linear_layers):
            if i < len(linear_layers) - 1:
                # Intermediate: Xavier init
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # Final layer: zero init (adapter output starts at 0)
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embeds: [batch, seq_len, embed_dim]
        Returns:
            adapted: [batch, seq_len, embed_dim]
        """
        # Residual adapter: output = input + scale * adapter(input)
        adapted = text_embeds + self.scale * self.adapter(text_embeds)
        return adapted

def _as_bool(v):
    """Safely convert a string/bool option to bool."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ('yes', 'true', 't', 'y', '1')
    return bool(v)


class TexGaussian(nn.Module):
    def __init__(self, opt, device):
        super().__init__()

        self.opt = opt
        self.device = device

        if _as_bool(self.opt.gaussian_loss):
            self.gaussian_mean = torch.load(self.opt.mean_path).to(self.device)
            self.gaussian_std = torch.load(self.opt.std_path).to(self.device)

        self.use_longclip = _as_bool(self.opt.use_longclip)

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
            
            # Text adapter for LongCLIP feature adaptation
            self.use_text_adapter = _as_bool(getattr(self.opt, 'use_text_adapter', False))
            
            if self.use_text_adapter and self.use_longclip:
                self.text_adapter = TextAdapter(
                    embed_dim=self.opt.context_dim,  # 768
                    hidden_dim=256,
                    num_layers=2,
                    dropout=0.1,
                    residual_scale=0.1,
                )
                self.text_adapter.to(self.device)
            else:
                self.text_adapter = None

        self.use_normal_head = _as_bool(self.opt.use_normal_head)
        self.use_rotation_head = _as_bool(self.opt.use_rotation_head)
        self.lambda_geo_normal = self.opt.lambda_geo_normal
        self.lambda_tex_normal = self.opt.lambda_tex_normal
        self.normal_loss_warmup_epochs = getattr(self.opt, 'normal_loss_warmup_epochs', 5)
        self.current_epoch = 0  # Set by training loop for warmup scheduling
        self.normal_residual_scale = 0.3    # tanh-bounded: max ~17° correction from mesh normal
        self.rotation_residual_scale = 0.1   # tanh-bounded: small rotation perturbation suffices

        if self.opt.use_material:
            self.opt.out_channels += 3

        if self.opt.input_feature == 'L':
            self.opt.in_channels = 3

        elif self.opt.input_feature == 'ND':
            self.opt.in_channels = 4

        # Check for GGCA option
        self.use_ggca = _as_bool(getattr(self.opt, 'use_ggca', False))

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
            use_ggca = self.use_ggca,
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
                self.normal_head = OctreeHead(
                    in_channels=normal_in, mid_channels=normal_in, out_channels=3, groups=32
                )
                self._init_head(self.normal_head)
                # EMA copy for normal_head
                self.ema_normal_head = copy.deepcopy(self.normal_head)
                self.ema_normal_head.to(self.device)
                set_requires_grad(self.ema_normal_head, False)
            if self.use_rotation_head:
                self.rotation_head = OctreeHead(
                    in_channels=normal_in, mid_channels=normal_in, out_channels=4, groups=32
                )
                self._init_head(self.rotation_head)
                # EMA copy for rotation_head
                self.ema_rotation_head = copy.deepcopy(self.rotation_head)
                self.ema_rotation_head.to(self.device)
                set_requires_grad(self.ema_rotation_head, False)

        # EMA copy for text_adapter (if enabled)
        if hasattr(self, 'text_adapter') and self.text_adapter is not None:
            self.ema_text_adapter = copy.deepcopy(self.text_adapter)
            self.ema_text_adapter.to(self.device)
            set_requires_grad(self.ema_text_adapter, False)

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
            self.material_act = lambda x: torch.sigmoid(x)
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
        # Also sync EMA copies of heads and adapter
        if hasattr(self, 'ema_normal_head') and hasattr(self, 'normal_head'):
            self.ema_normal_head.load_state_dict(self.normal_head.state_dict())
        if hasattr(self, 'ema_rotation_head') and hasattr(self, 'rotation_head'):
            self.ema_rotation_head.load_state_dict(self.rotation_head.state_dict())
        if hasattr(self, 'ema_text_adapter') and hasattr(self, 'text_adapter') and self.text_adapter is not None:
            self.ema_text_adapter.load_state_dict(self.text_adapter.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)
        # Update EMA for heads and adapter
        if hasattr(self, 'ema_normal_head') and hasattr(self, 'normal_head'):
            update_moving_average(self.ema_normal_head, self.normal_head, self.ema_updater)
        if hasattr(self, 'ema_rotation_head') and hasattr(self, 'rotation_head'):
            update_moving_average(self.ema_rotation_head, self.rotation_head, self.ema_updater)
        if hasattr(self, 'ema_text_adapter') and hasattr(self, 'text_adapter') and self.text_adapter is not None:
            update_moving_average(self.ema_text_adapter, self.text_adapter, self.ema_updater)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def _init_head(self, head, bias_value=None):
        """Initialize head so it starts as near-zero output (residual-safe).
        
        For OctreeHead: Kaiming-init conv1 (break symmetry), zero-init conv2 (start near zero).
        For single OctreeConv: zero-init everything.
        """
        if isinstance(head, OctreeHead):
            # First conv: Kaiming init for proper gradient flow
            for _, p in head.conv1.named_parameters():
                if p.dim() > 1:
                    nn.init.kaiming_normal_(p, nonlinearity='linear')
                else:
                    nn.init.zeros_(p)
            # Second conv: zero-init so head output starts near zero
            for _, p in head.conv2.named_parameters():
                nn.init.zeros_(p)
        else:
            for _, p in head.named_parameters():
                nn.init.zeros_(p)
        if bias_value is not None:
            # Find the final layer's bias and set it
            final_conv = head.conv2 if isinstance(head, OctreeHead) else head
            ref_param = next(final_conv.parameters())
            bias_tensor = torch.tensor(bias_value, device=ref_param.device, dtype=ref_param.dtype)
            for _, p in final_conv.named_parameters():
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

    def forward_gaussians(self, x, octree, condition = None, data = None, ema = False, condition_ggca = None):
        # x: [N, 4]
        # return: Gaussians: [N, dim_t]
        # condition: adapted text embeddings for all CAs (single embedding stream)
        # condition_ggca: same adapted embeddings for GGCA (unified with condition)
        input_feature = x
        mesh_normal_prior = None
        if input_feature.shape[1] >= 3:
            mesh_normal_prior = F.normalize(input_feature[:, :3], dim=-1, eps=1e-6)

        use_extra_heads = self.use_normal_head or self.use_rotation_head
        
        # Pass normals to GGCA if enabled
        normals_for_ggca = mesh_normal_prior if self.use_ggca else None
        
        if ema:
            if use_extra_heads:
                base_out, feat = self.ema_model(input_feature, octree, condition, return_features=True, normals=normals_for_ggca, condition_ggca=condition_ggca)
            else:
                base_out = self.ema_model(input_feature, octree, condition, normals=normals_for_ggca, condition_ggca=condition_ggca)
        else:
            if use_extra_heads:
                base_out, feat = self.model(input_feature, octree, condition, return_features=True, normals=normals_for_ggca, condition_ggca=condition_ggca)
            else:
                base_out = self.model(input_feature, octree, condition, normals=normals_for_ggca, condition_ggca=condition_ggca) # [N, out_channels]

        pred_normal_world = None
        rotation_raw = None
        if use_extra_heads:
            depth = octree.depth
            if self.use_normal_head:
                head = self.ema_normal_head if (ema and hasattr(self, 'ema_normal_head')) else self.normal_head
                normal_raw = head(feat, octree, depth)
                if mesh_normal_prior is not None:
                    # tanh bounds the residual to [-scale, +scale], preventing
                    # unbounded drift from the base mesh normal over training.
                    pred_normal_world = self.normal_act(
                        mesh_normal_prior + self.normal_residual_scale * torch.tanh(normal_raw)
                    )
                else:
                    pred_normal_world = self.normal_act(normal_raw)
            if self.use_rotation_head:
                head = self.ema_rotation_head if (ema and hasattr(self, 'ema_rotation_head')) else self.rotation_head
                rotation_raw = head(feat, octree, depth)

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
                # tanh bounds rotation residual to [-scale, +scale]
                delta_quat = self.rot_act(
                    self.rotation_residual_scale * torch.tanh(rotation_raw) + quat_identity
                )
                # base_quat * delta_quat: apply base rotation first (align to normal),
                # then small local perturbation from delta_quat
                gaussian_params[:, 4:8] = self.quat_multiply(base_quat, delta_quat)
            else:
                gaussian_params[:, 4:8] = rotation_raw

        pos = self.pos_act(octree.position) # [N, 3]
        opacity = self.opacity_act(gaussian_params[:, :1]) # [N, 1]
        scale = self.scale_act(gaussian_params[:, 1:4]) # [N, 3]
        rotation = self.rot_act(gaussian_params[:, 4:8]) # [N, 4]
        rgbs = self.rgb_act(gaussian_params[:, 8:11]) # [N, 3]
        if self.opt.use_material:
            material = self.material_act(gaussian_params[:, 11:14]) # [N, 3]

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [N, 14]
        if self.opt.use_material:
            material_gaussians = torch.cat([pos, opacity, scale, rotation, material], dim=-1) # [N, 14]
            return gaussian_loss, gaussians, material_gaussians, pred_normal_world

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
            # Apply text adapter to ALL embeddings (single embedding stream)
            # When using LongCLIP, the adapter bridges the distribution gap for
            # both frozen CrossAttention layers AND GGCA, rather than only GGCA.
            if hasattr(self, 'text_adapter') and self.text_adapter is not None:
                adapted = self.text_adapter(text_embeds)
            else:
                adapted = text_embeds
            input['text_embedding'] = adapted       # [bs, seq_len, 768]
            input['text_embedding_ggca'] = adapted   # same adapted embeddings for GGCA

    def forward(self, data, ema = False):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        self.set_input(data)

        octree = data['octree']

        condition = None
        condition_ggca = None

        if self.opt.use_text:
            condition = data['text_embedding']  # [bs, seq_len, 768] adapted embeddings
            condition_ggca = data.get('text_embedding_ggca', condition)  # same adapted for GGCA

        input_feature = octree.get_input_feature(feature = self.opt.input_feature, nempty = True)
        aligned_gt_mesh_normal = None
        if input_feature.shape[1] >= 3:
            aligned_gt_mesh_normal = input_feature[:, :3]
        gaussian_loss, gaussians, material_gaussians, pred_normal_world = self.forward_gaussians(
            input_feature, octree, condition, data, ema = ema, condition_ggca = condition_ggca) # [N, 14]
        batch_id = octree.batch_id(self.opt.input_depth, nempty = True)

        loss += gaussian_loss

        # Use black bg for all channels (albedo, material, normal) so that
        # pred and GT backgrounds are consistent and mask-edge leakage is zero.
        bg_color = torch.zeros(3, dtype=torch.float32, device=gaussians.device)

        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, batch_id, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        if self.opt.use_material:
            material_bg = torch.zeros(3, dtype=torch.float32, device=gaussians.device)
            material_results = self.gs.render(
                material_gaussians, batch_id, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=material_bg
            )
            material_pred_images = material_results['image']  # [B, V, 3, H, W]
            rough_pred = material_pred_images[:, :, 0:1, ...]
            metal_pred = material_pred_images[:, :, 1:2, ...]

        results['images_pred'] = pred_images
        if self.opt.use_material:
            results['rough_images_pred'] = rough_pred
            results['metallic_images_pred'] = metal_pred
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        if self.opt.use_material:
            rough_gt = data['rough_images_output']  # [B, V, 1, H, W]
            metal_gt = data['metallic_images_output']  # [B, V, 1, H, W]
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks  # black bg, matching pred renderer bg_color=zeros
        if self.opt.use_material:
            rough_gt = rough_gt * gt_masks  # black bg for material GT too
            metal_gt = metal_gt * gt_masks

        # Alpha-weighted losses: only penalise where Gaussians actually render
        # (pred_alpha > 0).  Coverage is handled separately by mask_loss,
        # so holes between Gaussians do not inject spurious gradients.
        alpha_w = pred_alphas.detach()  # [B, V, 1, H, W] – stop gradient
        albedo_loss = ((pred_images - gt_images) ** 2 * alpha_w).sum() / (alpha_w.sum() * 3 + 1e-6)
        if self.opt.use_material:
            # Material Gaussians share the same pos/scale/opacity → same holes as albedo
            rough_loss = ((rough_pred - rough_gt) ** 2 * alpha_w).sum() / (alpha_w.sum() + 1e-6)
            metal_loss = ((metal_pred - metal_gt) ** 2 * alpha_w).sum() / (alpha_w.sum() + 1e-6)

        mask_loss = F.mse_loss(pred_alphas, gt_masks)

        if self.opt.use_material:
            loss_mse = albedo_loss + rough_loss + metal_loss + mask_loss

        else:
            loss_mse = albedo_loss + mask_loss

        results['albedo_loss'] = albedo_loss.item()
        if self.opt.use_material:
            results['roughness_loss'] = rough_loss.item()
            results['metallic_loss'] = metal_loss.item()
        results['mask_loss'] = mask_loss.item()
        results['gaussian_loss'] = gaussian_loss.item()

        loss = loss + loss_mse

        if self.use_normal_head or self.use_rotation_head:
            normal_losses = self.compute_normal_losses(
                gaussians, pred_normal_world, aligned_gt_mesh_normal, data, batch_id
            )
            loss_geo = normal_losses['normal_geo_loss']
            loss_tex = normal_losses['normal_tex_loss']
            # Warmup: linearly ramp normal loss weights from 0 to full over warmup epochs
            warmup_epochs = max(1, self.normal_loss_warmup_epochs)
            warmup_factor = min(1.0, self.current_epoch / warmup_epochs)
            if self.use_rotation_head:
                loss = loss + warmup_factor * self.lambda_geo_normal * loss_geo
            if self.use_normal_head and pred_normal_world is not None:
                loss = loss + warmup_factor * self.lambda_tex_normal * loss_tex
            results['normal_geo_loss'] = loss_geo.item()
            results['normal_tex_loss'] = loss_tex.item()
            results['normal_warmup_factor'] = warmup_factor
            # Forward rendered normal images for visualisation in main.py
            if 'normal_images_pred' in normal_losses:
                results['normal_images_pred'] = normal_losses['normal_images_pred']
                results['normal_images_gt'] = normal_losses['normal_images_gt']
        else:
            results['normal_geo_loss'] = 0.0
            results['normal_tex_loss'] = 0.0

        loss_lpips = torch.zeros(1, device=gaussians.device)
        if self.opt.lambda_lpips > 0:
            # Use pred_alphas (detached) as mask – consistent with alpha-weighted MSE.
            # Only compute perceptual loss where Gaussians actually render;
            # coverage pressure comes solely from mask_loss.
            BV = gt_images.shape[0] * gt_images.shape[1]
            alpha_mask = alpha_w.view(BV, 1, self.opt.output_size, self.opt.output_size)
            gt_resized = F.interpolate(
                gt_images.view(BV, 3, self.opt.output_size, self.opt.output_size) * alpha_mask * 2 - 1,
                (256, 256), mode='bilinear', align_corners=False
            )
            pred_resized = F.interpolate(
                pred_images.view(BV, 3, self.opt.output_size, self.opt.output_size) * alpha_mask * 2 - 1,
                (256, 256), mode='bilinear', align_corners=False
            )
            loss_lpips = self.lpips_loss(gt_resized, pred_resized).mean() * self.opt.lambda_lpips
            loss = loss + loss_lpips

        results['lpips_loss'] = self.opt.lambda_lpips * loss_lpips.item()

        results['loss'] = loss

        # metric – masked PSNR (only foreground pixels) to be consistent with loss
        with torch.no_grad():
            fg_count = gt_masks.sum().clamp(min=1)
            psnr = -10 * torch.log10(((pred_images.detach() - gt_images) ** 2 * gt_masks).sum() / (fg_count * 3) + 1e-8)
            results['psnr'] = psnr

            if self.opt.use_material:
                rough_psnr = -10 * torch.log10(((rough_pred.detach() - rough_gt) ** 2 * gt_masks).sum() / (fg_count + 1e-6) + 1e-8)
                metal_psnr = -10 * torch.log10(((metal_pred.detach() - metal_gt) ** 2 * gt_masks).sum() / (fg_count + 1e-6) + 1e-8)
                results['roughness_psnr'] = rough_psnr
                results['metallic_psnr'] = metal_psnr

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

    def compute_normal_losses(self, gaussians, pred_normal_world, mesh_normal_prior, data, batch_id):
        device = gaussians.device
        loss_geo = torch.zeros(1, device=device)
        loss_tex = torch.zeros(1, device=device)

        # Geometric regularizer: align Gaussian shortest axis with mesh surface normal.
        # mesh_normal_prior = input_feature[:, :3] = mesh vertex normals from octree.
        # This is a structural prior ("ellipsoids should be flat along the surface"),
        # NOT a GT normal supervision.
        if self.use_rotation_head and mesh_normal_prior is not None:
            scales = gaussians[:, 4:7]
            quats = gaussians[:, 7:11]
            axis = self.get_shortest_axis(quats, scales)
            target = F.normalize(mesh_normal_prior, dim=-1, eps=1e-6)
            loss_geo = (1 - F.cosine_similarity(axis, target, dim=-1)).mean()

        if self.use_normal_head and pred_normal_world is not None and 'gt_normal_map' in data:
            normal_colors = (pred_normal_world * 0.5 + 0.5).clamp(0, 1)
            normal_gaussians = torch.cat([gaussians[:, :11], normal_colors], dim=-1)
            # Use black background to match GT normal maps (world-space normals have no
            # canonical "neutral" direction, so [0.5,0.5,1.0] is inappropriate).
            bg_color = torch.zeros(3, device=device, dtype=torch.float32)
            normal_results = self.gs.render(
                normal_gaussians,
                batch_id,
                data['cam_view'],
                data['cam_view_proj'],
                data['cam_pos'],
                bg_color=bg_color,
            )
            pred_normal_img = normal_results['image']
            normal_alpha = normal_results['alpha'].detach()  # [B, V, 1, H, W]
            gt_normal_img = data['gt_normal_map']

            # Use rendered alpha as weight – holes (alpha≈0) don't contribute
            l1 = torch.abs(pred_normal_img - gt_normal_img)
            l1 = (l1 * normal_alpha).sum() / (normal_alpha.sum() * 3 + 1e-6)

            pred_unit = F.normalize(pred_normal_img * 2 - 1, dim=2, eps=1e-6)
            gt_unit = F.normalize(gt_normal_img * 2 - 1, dim=2, eps=1e-6)
            cos = 1 - (pred_unit * gt_unit).sum(dim=2, keepdim=True)
            cos = (cos * normal_alpha).sum() / (normal_alpha.sum() + 1e-6)
            loss_tex = l1 + cos

        result = {'normal_geo_loss': loss_geo, 'normal_tex_loss': loss_tex}
        # Expose rendered normal images for visualisation (detached, no grad)
        if self.use_normal_head and pred_normal_world is not None and 'gt_normal_map' in data:
            result['normal_images_pred'] = pred_normal_img.detach()
            result['normal_images_gt'] = gt_normal_img.detach()
        return result
