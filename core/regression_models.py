import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import copy

import ocnn

from kiui.lpips import LPIPS
import open3d as o3d
from external.clip import clip as clip_module

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
        
        All layers use Xavier init for proper gradient flow.
        The final layer uses a small gain (0.1) so the adapter output
        starts near zero, keeping the residual connection near-identity.
        Combined with residual_scale (0.1), initial perturbation is ~1%.
        
        Previous zero-init on the final layer was a gradient deadlock bug:
        W_2=0 makes ∂L/∂W_1 = scale * ∂L/∂out * W_2^T * ... = 0,
        blocking the intermediate layer from learning until W_2 grows.
        """
        linear_layers = [m for m in self.adapter if isinstance(m, nn.Linear)]
        for i, module in enumerate(linear_layers):
            if i < len(linear_layers) - 1:
                # Intermediate: Xavier init (standard gain)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # Final layer: Xavier with small gain (NOT zero-init!)
                # Allows gradient to flow back through all layers from step 1.
                # Safety: residual_scale (0.1) * small_output ≈ 1% perturbation.
                nn.init.xavier_uniform_(module.weight, gain=0.1)
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

        self.input_depth = self.opt.input_depth
        self.full_depth = self.opt.full_depth

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            if self.opt.use_local_pretrained_ckpt:
                self.lpips_loss = LPIPS(net='vgg', use_url=False, checkpoint_dir = './lpips_checkpoints')
            else:
                self.lpips_loss = LPIPS(net ='vgg')
            self.lpips_loss.requires_grad_(False)

        # CLIP semantic losses (feature-space supervision for CLIP/FID-family metrics)
        self.use_clip_semantic_loss = _as_bool(getattr(self.opt, "use_clip_semantic_loss", False))
        self.clip_loss_random_views = _as_bool(getattr(self.opt, "clip_loss_random_views", True))
        self.clip_loss_use_gt_mask = _as_bool(getattr(self.opt, "clip_loss_use_gt_mask", True))
        self.clip_loss_num_views = max(1, int(getattr(self.opt, "clip_loss_num_views", 2)))
        self.clip_loss_img_size = int(getattr(self.opt, "clip_loss_img_size", 224))
        self.lambda_clip_image = float(getattr(self.opt, "lambda_clip_image", 0.0))
        self.lambda_clip_text = float(getattr(self.opt, "lambda_clip_text", 0.0))
        self.lambda_color_stats = float(getattr(self.opt, "lambda_color_stats", 0.0))
        self.alpha_gt_blend = float(getattr(self.opt, "alpha_gt_blend", 0.0))
        self.alpha_gt_blend = min(1.0, max(0.0, self.alpha_gt_blend))

        self.register_buffer(
            "clip_image_mean",
            torch.tensor([0.48145466, 0.45782750, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_image_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.clip_loss_model = None
        self.clip_tokenize = None
        if self.use_clip_semantic_loss and (self.lambda_clip_image > 0 or self.lambda_clip_text > 0):
            clip_loss_model_name = getattr(self.opt, "clip_loss_model", "ViT-B/32")
            try:
                self.clip_loss_model, _ = clip_module.load(name=clip_loss_model_name, device=self.device, jit=False)
                self.clip_loss_model = self.clip_loss_model.float()
                self.clip_loss_model.eval()
                for p in self.clip_loss_model.parameters():
                    p.requires_grad = False
                self.clip_tokenize = clip_module.tokenize
                print(
                    f"[INFO] CLIP semantic loss enabled: model={clip_loss_model_name}, "
                    f"lambda_img={self.lambda_clip_image}, lambda_text={self.lambda_clip_text}, "
                    f"views={self.clip_loss_num_views}"
                )
            except Exception as e:
                print(f"[WARN] Failed to load CLIP semantic model; disabling CLIP losses. Error: {e}")
                self.clip_loss_model = None
                self.clip_tokenize = None
                self.use_clip_semantic_loss = False

    def _clip_encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images with CLIP loss model, handling full-sequence VisionTransformer output."""
        feat = self.clip_loss_model.encode_image(images)
        if feat.ndim == 3:
            # VisionTransformer returns [N, seq_len, width]; manually pool CLS + project
            feat = self.clip_loss_model.visual.ln_post(feat[:, 0, :])
            if self.clip_loss_model.visual.proj is not None:
                feat = feat @ self.clip_loss_model.visual.proj
        return feat

    def _prepare_clip_image_batch(self, images_01: torch.Tensor) -> torch.Tensor:
        x = images_01.clamp(0, 1).float()
        x = F.interpolate(x, size=(self.clip_loss_img_size, self.clip_loss_img_size), mode='bicubic', align_corners=False)
        x = (x - self.clip_image_mean) / self.clip_image_std
        return x

    @staticmethod
    def _masked_channel_moments(images: torch.Tensor, masks: torch.Tensor):
        # images: [B, V, 3, H, W], masks: [B, V, 1, H, W]
        denom = masks.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        mean = (images * masks).sum(dim=(-1, -2), keepdim=True) / denom
        var = (((images - mean) ** 2) * masks).sum(dim=(-1, -2), keepdim=True) / denom
        std = torch.sqrt(var + 1e-6)
        return mean, std

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())
        if hasattr(self, 'ema_text_adapter') and hasattr(self, 'text_adapter') and self.text_adapter is not None:
            self.ema_text_adapter.load_state_dict(self.text_adapter.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)
        if hasattr(self, 'ema_text_adapter') and hasattr(self, 'text_adapter') and self.text_adapter is not None:
            update_moving_average(self.ema_text_adapter, self.text_adapter, self.ema_updater)

    def state_dict(self, **kwargs):
        # Remove auxiliary frozen loss networks from checkpoints.
        # They are not part of inference-time parameters.
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k or 'clip_loss_model' in k:
                del state_dict[k]
        return state_dict

    def forward_gaussians(self, x, octree, condition = None, data = None, ema = False, condition_ggca = None):
        # x: [N, 4]
        # return: Gaussians: [N, dim_t]
        # condition: adapted text embeddings for all CAs (single embedding stream)
        # condition_ggca: same adapted embeddings for GGCA (unified with condition)
        input_feature = x
        mesh_normal_prior = None
        if input_feature.shape[1] >= 3:
            mesh_normal_prior = F.normalize(input_feature[:, :3], dim=-1, eps=1e-6)

        # Pass normals to GGCA geometry gate if enabled
        normals_for_ggca = mesh_normal_prior if self.use_ggca else None
        
        if ema:
            base_out = self.ema_model(input_feature, octree, condition, normals=normals_for_ggca, condition_ggca=condition_ggca)
        else:
            base_out = self.model(input_feature, octree, condition, normals=normals_for_ggca, condition_ggca=condition_ggca) # [N, out_channels]

        if self.opt.gaussian_loss:
            gaussian_loss = F.mse_loss(base_out, data['gaussian'])
            zeros = base_out.new_zeros([base_out.shape[0], 4])
            gaussian_params = torch.cat([base_out[:, :4], zeros, base_out[:, 4:]], dim=1)
            gaussian_params = gaussian_params * self.gaussian_std + self.gaussian_mean
        else:
            gaussian_loss = torch.zeros(1, device = self.device)
            zeros = base_out.new_zeros([base_out.shape[0], 4])
            gaussian_params = torch.cat([base_out[:, :4], zeros, base_out[:, 4:]], dim=1)

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
            return gaussian_loss, gaussians, material_gaussians

        else:
            return gaussian_loss, gaussians, None


    def set_input(self, input=None, ema=False):
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
            # Unified text stream: TextAdapter adapts embeddings for ALL consumers
            # (both frozen CrossAttention and GGCA) simultaneously.
            #
            # Why NOT split streams (raw→CA, adapted→GGCA):
            #   The pretrained CA was trained with 77-token CLIP, but LongCLIP raw
            #   embeddings are 248-token with different distribution — neither raw
            #   nor adapted matches the original CLIP distribution. Splitting streams
            #   causes CA and GGCA to pull in conflicting directions (v8 regression).
            #   A unified adapted stream lets TextAdapter learn a SINGLE adaptation
            #   that benefits both paths jointly (v7 validated this works better).
            if hasattr(self, 'text_adapter') and self.text_adapter is not None:
                # Use EMA text adapter during eval for consistency with ema_model
                if ema and hasattr(self, 'ema_text_adapter') and self.ema_text_adapter is not None:
                    adapted = self.ema_text_adapter(text_embeds)
                else:
                    adapted = self.text_adapter(text_embeds)
            else:
                adapted = text_embeds
            input['text_embedding'] = adapted       # [bs, seq_len, 768]
            input['text_embedding_ggca'] = adapted   # same adapted for GGCA

    def forward(self, data, ema = False):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        self.set_input(data, ema=ema)

        octree = data['octree']

        condition = None
        condition_ggca = None

        if self.opt.use_text:
            condition = data['text_embedding']  # [bs, seq_len, 768] adapted embeddings
            condition_ggca = data.get('text_embedding_ggca', condition)  # same adapted for GGCA

        input_feature = octree.get_input_feature(feature = self.opt.input_feature, nempty = True)
        gaussian_loss, gaussians, material_gaussians = self.forward_gaussians(
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
        if self.alpha_gt_blend > 0:
            alpha_w = (1.0 - self.alpha_gt_blend) * alpha_w + self.alpha_gt_blend * gt_masks
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

        # CLIP feature matching loss (image-image + image-text) and color moment matching.
        clip_image_loss = torch.zeros(1, device=gaussians.device)
        clip_text_loss = torch.zeros(1, device=gaussians.device)
        color_stats_loss = torch.zeros(1, device=gaussians.device)

        if self.lambda_color_stats > 0:
            pred_mean, pred_std = self._masked_channel_moments(pred_images, gt_masks)
            gt_mean, gt_std = self._masked_channel_moments(gt_images, gt_masks)
            color_stats_loss = F.l1_loss(pred_mean, gt_mean) + F.l1_loss(pred_std, gt_std)
            loss = loss + self.lambda_color_stats * color_stats_loss

        if (
            self.use_clip_semantic_loss
            and (not ema)
            and self.clip_loss_model is not None
            and (self.lambda_clip_image > 0 or self.lambda_clip_text > 0)
        ):
            bsz, nview = pred_images.shape[:2]
            view_count = min(self.clip_loss_num_views, nview)
            if view_count < nview:
                if self.training and self.clip_loss_random_views:
                    view_idx = torch.randperm(nview, device=pred_images.device)[:view_count]
                else:
                    view_idx = torch.arange(view_count, device=pred_images.device)
                pred_for_clip = pred_images[:, view_idx, ...]
                gt_for_clip = gt_images[:, view_idx, ...]
                mask_for_clip = gt_masks[:, view_idx, ...]
            else:
                pred_for_clip = pred_images
                gt_for_clip = gt_images
                mask_for_clip = gt_masks

            if self.clip_loss_use_gt_mask:
                pred_for_clip = pred_for_clip * mask_for_clip
                gt_for_clip = gt_for_clip * mask_for_clip

            pred_clip = self._prepare_clip_image_batch(
                pred_for_clip.reshape(-1, 3, pred_for_clip.shape[-2], pred_for_clip.shape[-1])
            )
            pred_feat = F.normalize(self._clip_encode_image(pred_clip), dim=-1)

            if self.lambda_clip_image > 0:
                gt_clip = self._prepare_clip_image_batch(
                    gt_for_clip.reshape(-1, 3, gt_for_clip.shape[-2], gt_for_clip.shape[-1])
                )
                gt_feat = F.normalize(self._clip_encode_image(gt_clip), dim=-1)
                clip_image_loss = (1.0 - (pred_feat * gt_feat).sum(dim=-1)).mean()
                loss = loss + self.lambda_clip_image * clip_image_loss

            if self.lambda_clip_text > 0 and self.clip_tokenize is not None and ('text' in data):
                text_list = data['text']
                if isinstance(text_list, str):
                    text_list = [text_list]
                text_list = [
                    t if isinstance(t, str) and len(t.strip()) > 0 else "an object with realistic texture"
                    for t in text_list
                ]
                try:
                    text_tokens = self.clip_tokenize(text_list, truncate=True).to(self.device)
                except TypeError:
                    text_tokens = self.clip_tokenize(text_list).to(self.device)
                text_feat = F.normalize(self.clip_loss_model.encode_text(text_tokens), dim=-1)
                text_feat = text_feat.repeat_interleave(view_count, dim=0)
                clip_text_loss = (1.0 - (pred_feat * text_feat).sum(dim=-1)).mean()
                loss = loss + self.lambda_clip_text * clip_text_loss

        results['lpips_loss'] = loss_lpips.item()
        results['clip_image_loss'] = clip_image_loss.item()
        results['clip_text_loss'] = clip_text_loss.item()
        results['color_stats_loss'] = color_stats_loss.item()

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
