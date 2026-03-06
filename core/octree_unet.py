import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple, Literal, Optional
from torch.utils.checkpoint import checkpoint

import ocnn

def ckpt_conv_wrapper(conv_op, x, *args):
    def conv_wrapper(x, dummy_tensor, *args):
        return conv_op(x, *args)

    # The dummy tensor is a workaround when the checkpoint is used for the first conv layer:
    # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
    dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

    return checkpoint(
        conv_wrapper, x, dummy, *args, use_reentrant=False)


class GeometryGatedCrossAttention(nn.Module):
    """
    Geometry-Gated Cross-Attention (GGCA) module with inner-dim projection and FFN.
    
    Uses mesh normals to modulate the cross-attention between text features and 
    geometry features. The gate learns to selectively apply text conditioning 
    based on local geometric properties (e.g., flat vs detailed regions).
    
    Key design choices:
    - inner_dim projection: lifts low-dim features (e.g. 64) to a higher-dim space
      (e.g. 256) for attention, giving each head a reasonable head_dim (e.g. 32).
    - FFN block after attention: standard transformer practice for capacity.
    - Geometry gate: sigmoid gate modulated by normals controls text influence.
    
    Args:
        dim: Feature dimension (input/output)
        num_heads: Number of attention heads
        context_dim: Text embedding dimension (e.g., 768 for CLIP/LongCLIP)
        inner_dim: Dimension for attention computation (default: 4*dim).
                   Set None to use 4*dim automatically.
        ffn_mult: FFN hidden dim multiplier relative to inner_dim (default: 4.0)
        normal_dim: Dimension of normal vectors (default: 3)
        gate_bias: Initial bias for the gate (default: -2.0, sigmoid(-2)≈0.12)
        output_scale: Initial scale for the gated output (default: 0.1)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: int,
        inner_dim: Optional[int] = None,
        ffn_mult: float = 4.0,
        normal_dim: int = 3,
        gate_bias: float = -2.0,
        output_scale: float = 0.1,
        attn_drop: float = 0.0,
        use_geometry_gate: bool = True,  # When False, use learnable scalar gate (no normals needed)
    ):
        super().__init__()
        self.dim = dim
        self.normal_dim = normal_dim
        self.use_geometry_gate = use_geometry_gate
        self.inner_dim = inner_dim if inner_dim is not None else dim * 4
        
        # --- Input projection (dim -> inner_dim) ---
        self.use_proj = (self.inner_dim != dim)
        if self.use_proj:
            self.proj_in = nn.Linear(dim, self.inner_dim)
            self.proj_out = nn.Linear(self.inner_dim, dim)
            # Xavier init for proj_out: allows gradient flow from step 1.
            # Safety is ensured by output_scale * gate (starts at ~0.1 * 0.12 ≈ 0.012)
            nn.init.xavier_uniform_(self.proj_out.weight, gain=0.1)
            nn.init.zeros_(self.proj_out.bias)
        
        # --- Cross-attention in inner_dim space ---
        self.cross_norm = nn.LayerNorm(self.inner_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.inner_dim, 
            num_heads=num_heads, 
            dropout=attn_drop,
            kdim=context_dim, 
            vdim=context_dim, 
            batch_first=True
        )
        
        # --- FFN after attention (standard transformer block) ---
        ffn_hidden = int(self.inner_dim * ffn_mult)
        self.ffn_norm = nn.LayerNorm(self.inner_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.inner_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, self.inner_dim),
        )
        # Xavier init for FFN output: allows gradient flow through FFN.
        # Safety ensured by output_scale * gate at the module output.
        nn.init.xavier_uniform_(self.ffn[-1].weight, gain=0.1)
        nn.init.zeros_(self.ffn[-1].bias)
        
        # Learnable output scale (replaces zero-init as the safety mechanism)
        # Starts small so GGCA contribution is initially ~output_scale * sigmoid(gate_bias)
        self.output_scale = nn.Parameter(torch.tensor(output_scale))
        
        # --- Gate mechanism ---
        if use_geometry_gate:
            # Geometry gate: modulated by normals + features
            # Input: concatenation of features (dim) and normals (normal_dim)
            # Output: per-point gate value in [0, 1]
            gate_hidden = max(dim // 2, 16)
            self.gate_net = nn.Sequential(
                nn.Linear(dim + normal_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, gate_hidden // 2),
                nn.SiLU(),
                nn.Linear(gate_hidden // 2, 1),
            )
            self._init_gate(gate_bias)
        else:
            # Learnable scalar gate (no normals needed, e.g. for coarse octree levels)
            # Initialized to gate_bias so sigmoid(gate_bias) ≈ 0.12
            self.gate_net = None
            self.gate_scalar = nn.Parameter(torch.tensor(gate_bias))
        
    def _init_gate(self, bias: float):
        """
        Initialize gate network: Kaiming for intermediate layers (break symmetry),
        small-scale Xavier for output layer with negative bias so gate starts small.
        
        With bias=-2: sigmoid(-2) ≈ 0.12, giving a small initial text contribution.
        Using Xavier (not zero) for the output layer weight allows gradients to flow
        back through all gate_net layers from step 1.
        """
        for module in self.gate_net:
            if isinstance(module, nn.Linear):
                # Kaiming init so gradients flow from the start
                nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Small-scale Xavier for the final layer (not zero!) so gradient flows
        final_linear = self.gate_net[-1]
        nn.init.xavier_uniform_(final_linear.weight, gain=0.01)
        if final_linear.bias is not None:
            nn.init.constant_(final_linear.bias, bias)
    
    def forward(
        self, 
        data: torch.Tensor, 
        octree, 
        depth: int, 
        context: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            data: Feature tensor [N, dim]
            octree: Octree structure
            depth: Current octree depth
            context: Text embeddings [B, seq_len, context_dim]
            normals: Per-point normal vectors [N, 3]
            
        Returns:
            Gated output tensor [N, dim]
        """
        if context is None:
            return data
        
        # Project to inner_dim if needed
        if self.use_proj:
            h = self.proj_in(data)  # [N, inner_dim]
        else:
            h = data
            
        # Cross-attention (pre-norm)
        h_norm = self.cross_norm(h)
        
        batch_id = octree.batch_id(depth=depth, nempty=True)
        batch_size = octree.batch_size
        
        cross_attn_results = []
        for i in range(batch_size):
            mask = (batch_id == i)
            cross_attn_i = h_norm[mask]
            cross_attn_i, _ = self.cross_attn(
                query=cross_attn_i.unsqueeze(0),
                key=context[i:i+1],
                value=context[i:i+1]
            )
            cross_attn_results.append(cross_attn_i.squeeze(0))
        
        attn_out = torch.cat(cross_attn_results, dim=0)
        h = h + attn_out  # Residual
        
        # FFN (pre-norm)
        h = h + self.ffn(self.ffn_norm(h))  # Residual
        
        # Project back to dim
        if self.use_proj:
            h = self.proj_out(h)  # [N, dim]
        
        # Compute gate
        if self.use_geometry_gate:
            if normals is not None:
                normals = F.normalize(normals, dim=-1, eps=1e-6)
                gate_input = torch.cat([data, normals], dim=-1)
                gate = torch.sigmoid(self.gate_net(gate_input))  # [N, 1]
            else:
                # Fallback: apply gate_net with zero normals so params get gradients
                zero_normals = torch.zeros(data.shape[0], self.normal_dim, device=data.device, dtype=data.dtype)
                gate_input = torch.cat([data, zero_normals], dim=-1)
                gate = torch.sigmoid(self.gate_net(gate_input))  # [N, 1]
        else:
            # Learnable scalar gate (broadcast to all points)
            gate = torch.sigmoid(self.gate_scalar)  # scalar
        
        # Gated residual connection with learnable scale
        # output_scale * gate controls GGCA contribution:
        #   - Starts at ~0.1 * sigmoid(-2) ≈ 0.012 (safe, near-identity)
        #   - Can grow as training progresses
        # gate ~ 1: trust text-conditioned attention
        # gate ~ 0: trust original features (geometry-driven)
        out = data + self.output_scale * gate * h
        
        return out

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float,
        context_dim: int,
        batch_first: bool = True,
    ):
        super().__init__()
        self.cross_norm = torch.nn.LayerNorm(dim)
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_drop, 
            kdim = context_dim, vdim = context_dim, 
            batch_first=batch_first
        )
    
    def forward(self, data, octree, depth, context = None):
        cross_attn_data = self.cross_norm(data)

        batch_id = octree.batch_id(depth = depth, nempty = True)
        batch_size = octree.batch_size

        cross_attn_results = []

        for i in range(batch_size):
            cross_attn_i = cross_attn_data[batch_id == i]
            cross_attn_i, _ = self.cross_attn(
                query = cross_attn_i.unsqueeze(0), key = context[i:i+1], value = context[i:i+1])
            
            cross_attn_results.append(cross_attn_i.squeeze(0))
        
        cross_attn_results = torch.cat(cross_attn_results, dim = 0)

        data = data + cross_attn_results

        return data
    

class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
        use_checkpoint: bool = True, 
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.use_checkpoint = use_checkpoint

        self.norm1 = ocnn.nn.OctreeGroupNorm(in_channels = in_channels, group = groups, nempty = True)
        self.conv1 = ocnn.nn.OctreeConv(in_channels, out_channels, kernel_size=[3], nempty=True, use_bias=True)

        self.norm2 = ocnn.nn.OctreeGroupNorm(in_channels=out_channels, group=groups, nempty=True)
        self.conv2 = ocnn.nn.OctreeConv(out_channels, out_channels, kernel_size=[3], nempty=True, use_bias=True)

        self.act = F.silu
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = ocnn.modules.Conv1x1(in_channels, out_channels, use_bias = True)


    def forward(self, x, octree, depth):
        res = x

        x = self.norm1(x, octree, depth)

        x = self.act(x)
        
        if self.use_checkpoint:
            x = ckpt_conv_wrapper(self.conv1, x, octree, depth)
        else:
            x = self.conv1(x, octree, depth)

        x = self.norm2(x, octree, depth)
        x = self.act(x)

        if self.use_checkpoint:
            x = ckpt_conv_wrapper(self.conv2, x, octree, depth)
        else:
            x = self.conv2(x, octree, depth)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample: bool = True,
        attention: bool = True,
        num_heads: int = 16,
        context_dim: int = 768,
        skip_scale: float = 1,
        use_checkpoint: bool = True, 
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
 
        nets = []
        attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(in_channels, out_channels, skip_scale=skip_scale, use_checkpoint=use_checkpoint))
            if attention:
                attns.append(CrossAttention(dim = out_channels, num_heads = num_heads, attn_drop = 0.0, context_dim = context_dim, batch_first = True))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.downsample = None
        if downsample:
            self.downsample = ocnn.nn.OctreeConv(out_channels, out_channels, kernel_size=[3], stride=2, nempty = True, use_bias=True)

    def forward(self, x, octree, depth, condition = None):
        xs = []

        for attn, net in zip(self.attns, self.nets):
            x = net(x, octree, depth)
            if attn:
                if self.use_checkpoint:
                    x = checkpoint(attn, x, octree, depth, condition, use_reentrant=False)
                else:
                    x = attn(x, octree, depth, condition)
            xs.append(x)

        if self.downsample:
            if self.use_checkpoint:
                x = ckpt_conv_wrapper(self.downsample, x, octree, depth)
            else:
                x = self.downsample(x, octree, depth)
            xs.append(x)
  
        return x, xs


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        num_heads: int = 16,
        context_dim: int = 768,
        skip_scale: float = 1,
        use_checkpoint: bool = True, 
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        nets = []
        attns = []
        # first layer
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale, use_checkpoint=use_checkpoint))
        # more layers
        for i in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale, use_checkpoint=use_checkpoint))
            if attention:
                attns.append(CrossAttention(dim = in_channels, num_heads = num_heads, attn_drop = 0.0, context_dim = context_dim, batch_first = True))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        
    def forward(self, x, octree, depth, condition = None):
        x = self.nets[0](x, octree, depth)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn:
                if self.use_checkpoint:
                    x = checkpoint(attn, x, octree, depth, condition, use_reentrant=False)
                else:
                    x = attn(x, octree, depth, condition)
            x = net(x, octree, depth)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = True,
        attention: bool = True,
        num_heads: int = 16,
        context_dim: int = 768, 
        skip_scale: float = 1,
        use_checkpoint: bool = True, 
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        nets = []
        attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels
            nets.append(ResnetBlock(cin + cskip, out_channels, skip_scale=skip_scale, use_checkpoint=use_checkpoint))
            if attention:
                attns.append(CrossAttention(dim = out_channels, num_heads = num_heads, attn_drop = 0.0, context_dim = context_dim, batch_first = True))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.upsample = None
        if upsample:
            self.up_pool = ocnn.nn.OctreeUpsample(method="nearest", nempty=True)
            self.upsample = ocnn.nn.OctreeConv(out_channels, out_channels, kernel_size=[3], nempty=True, use_bias=True)

    def forward(self, x, xs, octree, depth, condition = None):

        for attn, net in zip(self.attns, self.nets):
            res_x = xs[-1]
            xs = xs[:-1]
            x = torch.cat([x, res_x], dim=1)
            x = net(x, octree, depth)
            if attn:
                if self.use_checkpoint:
                    x = checkpoint(attn, x, octree, depth, condition, use_reentrant=False)
                else:
                    x = attn(x, octree, depth, condition)
            
        if self.upsample:
            if self.use_checkpoint:
                x = ckpt_conv_wrapper(self.up_pool, x, octree, depth)
                x = ckpt_conv_wrapper(self.upsample, x, octree, depth + 1)
            else:
                x = self.up_pool(x, octree, depth)
                x = self.upsample(x, octree, depth + 1)

        return x


class OctreeUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 32, 
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8, 8),
        down_attention: Tuple[bool, ...] = (False, False, False, False, False),
        mid_attention: bool = False,
        up_attention: Tuple[bool, ...] = (False, False, False, False, False),
        num_heads: int = 16,
        context_dim: int = 768,
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
        use_checkpoint: bool = True,
        use_ggca: bool = False,  # Enable Geometry-Gated Cross-Attention
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.use_ggca = use_ggca

        down_channels = [model_channels * m for m in channel_mult]
        up_channels = [model_channels * m for m in channel_mult[::-1]]

        # first
        self.conv_in = ocnn.nn.OctreeConv(in_channels, down_channels[0], kernel_size=[3], nempty=True, use_bias=True)

        # down
        down_blocks = []
        cout = down_channels[0]
        self.downsample = []

        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            downsample = (i != len(down_channels) - 1)

            self.downsample.append(downsample)

            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample = downsample, # not final layer
                attention=down_attention[i],
                num_heads = num_heads,
                context_dim = context_dim,
                skip_scale=skip_scale,
                use_checkpoint=use_checkpoint,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(
            in_channels = down_channels[-1],
            num_layers = layers_per_block,
            attention=mid_attention,
            num_heads = num_heads,
            context_dim = context_dim,
            skip_scale=skip_scale, 
            use_checkpoint=use_checkpoint,
        )

        # up
        up_blocks = []
        cout = up_channels[0]

        self.upsample = []

        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric

            upsample = (i != len(up_channels) - 1)
            self.upsample.append(upsample)

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=upsample, # not final layer
                attention=up_attention[i],
                skip_scale=skip_scale,
                use_checkpoint = use_checkpoint,
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = ocnn.nn.OctreeGroupNorm(in_channels = up_channels[-1], group = 32, nempty = True)
        self.conv_out = ocnn.nn.OctreeConv(up_channels[-1], out_channels, kernel_size=[3], nempty=True, use_bias=True)
        self.conv = ocnn.nn.OctreeConv(out_channels, out_channels, kernel_size=[3], nempty=True, use_bias=True)

        # GGCA: Geometry-Gated Cross-Attention (optional)
        # Uses inner_dim projection so attention runs in a higher-dim space
        # with reasonable head_dim (e.g. 32), then projects back to feature dim.
        self.ggca = None
        self.ggca_mid = None
        if use_ggca:
            ggca_inner = up_channels[-1] * 4   # 64 * 4 = 256
            ggca_heads = max(ggca_inner // 32, 1)  # head_dim=32 -> 8 heads
            self.ggca = GeometryGatedCrossAttention(
                dim=up_channels[-1],
                num_heads=ggca_heads,
                context_dim=context_dim,
                inner_dim=ggca_inner,
                ffn_mult=4.0,
                normal_dim=3,
                gate_bias=-2.0,    # sigmoid(-2)≈0.12, starts with small text contribution
                output_scale=0.1,  # learnable scale, replaces zero-init for safe startup
            )

            # GGCA@512: high-dimensional text fusion at decoder's last 512-dim layer
            # Placed after up_blocks.1 (dim=512), where CrossAttention text features
            # are richest. At this depth normals are unavailable (coarser octree level),
            # so gate defaults to 0.5 (purely attention-based gating).
            mid_dim = up_channels[1]  # 512 for standard architecture
            if mid_dim >= 256:
                ggca_mid_heads = max(mid_dim // 32, 1)  # 512/32 = 16 heads
                self.ggca_mid = GeometryGatedCrossAttention(
                    dim=mid_dim,
                    num_heads=ggca_mid_heads,
                    context_dim=context_dim,
                    inner_dim=mid_dim,      # already high-dim, no need to project up
                    ffn_mult=4.0,
                    normal_dim=3,
                    gate_bias=-2.0,
                    output_scale=0.1,
                    use_geometry_gate=False,  # no normals at coarse octree level
                )

    def forward(self, x, octree, condition = None, normals: Optional[torch.Tensor] = None, condition_ggca: Optional[torch.Tensor] = None):
        # x: [N, Cin] where N is number of non-empty octree nodes
        # normals: [N, 3] mesh normals for GGCA geometry gate (optional)
        # condition_ggca: adapted text embeddings for GGCA (optional;
        #                 when None, falls back to condition for backward compat)

        input_data = x

        depth = octree.depth

        # first
        if self.use_checkpoint:
            x = ckpt_conv_wrapper(self.conv_in, x, octree, depth)
        else:
            x = self.conv_in(x, octree, depth)
        
        # down
        xss = [x]
        for block, downsample in zip(self.down_blocks, self.downsample):
            x, xs = block(x, octree, depth, condition)
            xss.extend(xs)

            if downsample:
                depth -= 1
        
        # mid
        x = self.mid_block(x, octree, depth, condition)

        # up
        ggca_cond = condition_ggca if condition_ggca is not None else condition
        for i, (block, upsample) in enumerate(zip(self.up_blocks, self.upsample)):
            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]
            x = block(x, xs, octree, depth, condition)

            if upsample:
                depth += 1

            # GGCA@512: apply high-dim text fusion after up_blocks.1
            # At this point x is [N, 512] at the upsampled depth.
            # normals=None because we're at a coarser octree level (gate→0.5).
            if i == 1 and self.ggca_mid is not None and ggca_cond is not None:
                if self.use_checkpoint:
                    x = checkpoint(self.ggca_mid, x, octree, depth, ggca_cond, None, use_reentrant=False)
                else:
                    x = self.ggca_mid(x, octree, depth, ggca_cond, None)

        # last
        x = self.norm_out(x, octree, depth)
        x = F.silu(x)
        
        # GGCA@64: enrich ALL features with text+geometry conditioning at conv_out entrance.
        # This allows text-conditioned information to flow into all output heads
        # (albedo, roughness, metallic, opacity, scale) through the shared conv_out.
        if self.ggca is not None and ggca_cond is not None:
            if self.use_checkpoint:
                x = checkpoint(self.ggca, x, octree, depth, ggca_cond, normals, use_reentrant=False)
            else:
                x = self.ggca(x, octree, depth, ggca_cond, normals)
        
        if self.use_checkpoint:
            x = ckpt_conv_wrapper(self.conv_out, x, octree, depth)
        else:
            x = self.conv_out(x, octree, depth) # [N, Cout]
        
        if self.use_checkpoint:
            x = ckpt_conv_wrapper(self.conv, x, octree, depth)
        else:
            x = self.conv(x, octree, depth)

        assert depth == octree.depth
        assert x.shape[0] == input_data.shape[0]

        return x
