#!/usr/bin/env python3
"""
Visualize TextAdapter and GGCA module effectiveness.

Generates publication-ready figures for the paper:
1. TextAdapter analysis: scale parameter, embedding shift, cosine similarity
2. GGCA gate heatmap: per-point gate values projected onto mesh surface
3. GGCA attention visualization: which text tokens each point attends to
4. Ablation comparison: with vs without text conditioning

Usage:
    python scripts/visualize_modules.py \
        --ckpt_path ../experiments/texverse_stage1_new_modules_v6/2026.02.25-02:22:08_lr_0.0004_num_views_8/best_ckpt/model.safetensors \
        --mesh_path ../experiments/texverse_stage1_new_modules_v6/textures/2386a34a0db44d26acd060079e19ce54/mesh.obj \
        --text_prompt "A retro-style video game cartridge with a rectangular black plastic body" \
        --output_dir ../experiments/texverse_stage1_new_modules_v6/visualizations \
        --pointcloud_dir ../datasets/texverse_pointcloud_npz
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize TextAdapter and GGCA modules')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mesh_path', type=str, default=None, help='Path to mesh OBJ file (optional, for gate heatmap)')
    parser.add_argument('--text_prompt', type=str, default='A retro-style video game cartridge', help='Text prompt for visualization')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for visualization images')
    parser.add_argument('--pointcloud_dir', type=str, default='../datasets/texverse_pointcloud_npz', help='Point cloud directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for saved figures')
    return parser.parse_args()


# ============================================================
# 1. TextAdapter Analysis (no model forward pass needed)
# ============================================================

def visualize_text_adapter(model, text_encoder, longclip_model, longclip_tokenize, 
                           text_prompt, device, output_dir, dpi=200):
    """
    Visualize TextAdapter effects:
    - Learned scale parameter value
    - Embedding shift (L2 norm of adapter output)  
    - Cosine similarity between original and adapted embeddings (per token)
    - Token-wise adaptation magnitude
    """
    print("\n=== Visualizing TextAdapter ===")
    
    # Get text adapter
    text_adapter = None
    ema_text_adapter = None
    if hasattr(model, 'text_adapter') and model.text_adapter is not None:
        text_adapter = model.text_adapter
    if hasattr(model, 'ema_text_adapter') and model.ema_text_adapter is not None:
        ema_text_adapter = model.ema_text_adapter
    
    adapter = ema_text_adapter if ema_text_adapter is not None else text_adapter
    if adapter is None:
        print("[WARN] No TextAdapter found, skipping")
        return {}
    
    # Print adapter parameters
    scale_val = adapter.scale.item()
    print(f"  TextAdapter learned scale: {scale_val:.6f} (initial: 0.1)")
    
    # Count parameters
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable:,} trainable")
    
    # Encode text
    if longclip_model is not None and longclip_tokenize is not None:
        context_length = 248
        token = longclip_tokenize(text_prompt, context_length=context_length, truncate=True)
        token = token.to(device)
        with torch.no_grad():
            x = longclip_model.token_embedding(token)
            x = x + longclip_model.positional_embedding[:x.shape[1]]
            x = x.permute(1, 0, 2)
            x = longclip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = longclip_model.ln_final(x)
            raw_embeds = x.float()  # [1, seq_len, 768]
    else:
        from external.clip import tokenize
        token = tokenize(text_prompt).to(device)
        raw_embeds = text_encoder.encode(token).float()
    
    # Apply adapter
    with torch.no_grad():
        adapted_embeds = adapter(raw_embeds)
    
    # Compute metrics per token
    raw = raw_embeds[0]       # [seq_len, 768]
    adapted = adapted_embeds[0]  # [seq_len, 768]
    
    # Find actual token length (non-padding)
    # For simplicity, use the norm to detect active tokens
    token_norms = raw.norm(dim=-1)  # [seq_len]
    active_mask = token_norms > 0.1  # tokens with non-trivial norms
    n_active = active_mask.sum().item()
    
    # Per-token cosine similarity
    cos_sim = F.cosine_similarity(raw, adapted, dim=-1).cpu().numpy()
    
    # Per-token L2 shift
    shift = (adapted - raw).norm(dim=-1).cpu().numpy()
    
    # Adapter output magnitude (before scale)
    with torch.no_grad():
        adapter_out = adapter.adapter(raw_embeds[0:1])
    adapter_mag = adapter_out[0].norm(dim=-1).cpu().numpy()
    
    # Per-dimension analysis
    delta = (adapted - raw).cpu().numpy()  # [seq_len, 768]
    dim_mean_shift = np.mean(np.abs(delta[:n_active]), axis=0)  # [768]
    
    # ---- Figure 1: TextAdapter Summary (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'TextAdapter Analysis\n(learned scale = {scale_val:.4f}, initial = 0.1)', 
                 fontsize=14, fontweight='bold')
    
    # 1a: Cosine similarity per token
    ax = axes[0, 0]
    x_range = np.arange(min(n_active + 5, len(cos_sim)))
    ax.bar(x_range, cos_sim[x_range], color='steelblue', alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Identity (no change)')
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(a) Cosine Similarity: Raw vs Adapted')
    ax.set_ylim(min(cos_sim[x_range].min() - 0.01, 0.95), 1.005)
    ax.legend()
    
    # 1b: L2 shift per token
    ax = axes[0, 1]
    ax.bar(x_range, shift[x_range], color='coral', alpha=0.8)
    ax.set_xlabel('Token Index')
    ax.set_ylabel('L2 Distance')
    ax.set_title('(b) Embedding Shift (L2 norm of delta)')
    
    # 1c: Adapter output magnitude (before scale)
    ax = axes[1, 0]
    ax.bar(x_range, adapter_mag[x_range], color='mediumpurple', alpha=0.8)
    ax.set_xlabel('Token Index')
    ax.set_ylabel('L2 Norm')
    ax.set_title(f'(c) Adapter MLP Output Magnitude (scale × this = shift)')
    
    # 1d: Per-dimension mean absolute shift
    ax = axes[1, 1]
    sorted_dims = np.argsort(dim_mean_shift)[::-1]
    top_k = 50
    ax.bar(range(top_k), dim_mean_shift[sorted_dims[:top_k]], color='seagreen', alpha=0.8)
    ax.set_xlabel(f'Dimension Index (sorted by shift, top-{top_k})')
    ax.set_ylabel('Mean |Δ|')
    ax.set_title('(d) Per-Dimension Adaptation Magnitude')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'textadapter_analysis.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
    
    # ---- Stats summary ----
    stats = {
        'scale': scale_val,
        'mean_cosine_sim': float(np.mean(cos_sim[:n_active])),
        'mean_l2_shift': float(np.mean(shift[:n_active])),
        'max_l2_shift': float(np.max(shift[:n_active])),
        'n_active_tokens': n_active,
    }
    print(f"  Mean cosine similarity (active tokens): {stats['mean_cosine_sim']:.6f}")
    print(f"  Mean L2 shift: {stats['mean_l2_shift']:.6f}")
    print(f"  Max L2 shift: {stats['max_l2_shift']:.6f}")
    
    return stats


# ============================================================
# 2. GGCA Gate Heatmap (needs forward pass with hook)
# ============================================================

def extract_ggca_intermediates(model, input_data, octree, text_embedding, 
                                text_embedding_ggca, device):
    """
    Run forward pass with hooks to capture GGCA gate values and attention weights.
    Returns dict with gate values, attention weights, normals, positions.
    """
    results = {}
    hooks = []
    
    # Get the GGCA module
    ggca = None
    ema_model = model.ema_model if hasattr(model, 'ema_model') else model.model
    if hasattr(ema_model, 'ggca') and ema_model.ggca is not None:
        ggca = ema_model.ggca
    
    if ggca is None:
        print("[WARN] No GGCA module found")
        return results
    
    # Save original forward to monkey-patch
    original_forward = ggca.forward
    
    def hooked_forward(data, octree, depth, context=None, normals=None):
        """Modified forward that captures intermediates."""
        if context is None:
            results['gate'] = None
            return data
        
        # Project to inner_dim
        if ggca.use_proj:
            h = ggca.proj_in(data)
        else:
            h = data
        
        # Cross-attention (pre-norm)
        h_norm = ggca.cross_norm(h)
        batch_id = octree.batch_id(depth=depth, nempty=True)
        batch_size = octree.batch_size
        
        all_attn_weights = []
        cross_attn_results = []
        for i in range(batch_size):
            mask = (batch_id == i)
            cross_attn_i = h_norm[mask]
            # Get attention weights!
            cross_attn_i, attn_w = ggca.cross_attn(
                query=cross_attn_i.unsqueeze(0),
                key=context[i:i+1],
                value=context[i:i+1],
                need_weights=True,
                average_attn_weights=True,  # average over heads
            )
            cross_attn_results.append(cross_attn_i.squeeze(0))
            all_attn_weights.append(attn_w.squeeze(0))  # [N_i, seq_len]
        
        attn_out = torch.cat(cross_attn_results, dim=0)
        h = h + attn_out
        
        # FFN
        h = h + ggca.ffn(ggca.ffn_norm(h))
        
        # Project back
        if ggca.use_proj:
            h = ggca.proj_out(h)
        
        # Compute geometry gate
        if normals is not None:
            normals_norm = F.normalize(normals, dim=-1, eps=1e-6)
            gate_input = torch.cat([data, normals_norm], dim=-1)
            gate = torch.sigmoid(ggca.gate_net(gate_input))
        else:
            gate = torch.ones(data.shape[0], 1, device=data.device) * 0.5
        
        # Save intermediates
        results['gate'] = gate.detach().cpu().numpy()            # [N, 1]
        results['attn_weights'] = [w.detach().cpu().numpy() for w in all_attn_weights]  # list of [N_i, seq_len]
        results['h_before_gate'] = h.detach().cpu().numpy()      # [N, dim]
        results['gate_times_h'] = (gate * h).detach().cpu().numpy()  # [N, dim]
        if normals is not None:
            results['normals'] = normals.detach().cpu().numpy()  # [N, 3]
        
        out = data + gate * h
        return out
    
    # Monkey-patch the forward
    ggca.forward = hooked_forward
    
    # Run forward pass
    with torch.no_grad():
        mesh_normal_prior = None
        if input_data.shape[1] >= 3:
            mesh_normal_prior = F.normalize(input_data[:, :3], dim=-1, eps=1e-6)
        
        normals_for_ggca = mesh_normal_prior if hasattr(model, 'use_ggca') and model.use_ggca else None
        
        condition_ggca = text_embedding_ggca if text_embedding_ggca is not None else text_embedding
        
        _ = ema_model(input_data, octree, text_embedding, normals=normals_for_ggca, condition_ggca=condition_ggca)
    
    # Restore original forward
    ggca.forward = original_forward
    
    # Get positions
    positions = octree.position.detach().cpu().numpy()  # [N, 3]
    results['positions'] = positions
    
    return results


def visualize_ggca_gate(results, output_dir, dpi=200):
    """
    Visualize GGCA gate values as 3D scatter plots (heatmap on point cloud).
    """
    if 'gate' not in results or results['gate'] is None:
        print("[WARN] No gate values to visualize")
        return
    
    gate = results['gate'].squeeze(-1)  # [N]
    positions = results['positions']     # [N, 3]
    
    print(f"\n=== GGCA Gate Statistics ===")
    print(f"  N points: {len(gate)}")
    print(f"  Gate range: [{gate.min():.4f}, {gate.max():.4f}]")
    print(f"  Gate mean: {gate.mean():.4f}")
    print(f"  Gate std: {gate.std():.4f}")
    print(f"  Gate median: {np.median(gate):.4f}")
    print(f"  Gate > 0.5: {(gate > 0.5).sum()} / {len(gate)} ({(gate > 0.5).mean()*100:.1f}%)")
    print(f"  Gate > 0.6: {(gate > 0.6).sum()} / {len(gate)} ({(gate > 0.6).mean()*100:.1f}%)")
    print(f"  Gate > 0.7: {(gate > 0.7).sum()} / {len(gate)} ({(gate > 0.7).mean()*100:.1f}%)")
    
    # ---- Figure 2: Gate heatmap (multi-view) ----
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('GGCA Geometry Gate Values on Point Cloud\n'
                 f'(gate→1: trust text, gate→0: trust geometry; '
                 f'mean={gate.mean():.3f}, std={gate.std():.3f})',
                 fontsize=14, fontweight='bold')
    
    # 4 views: front, side, top, 3/4 view
    views = [
        ('Front', 0, 0),
        ('Side', 0, 90),
        ('Top', 90, 0),
        ('3/4 View', 30, 45),
    ]
    
    # Subsample for faster rendering if too many points
    max_pts = 50000
    if len(gate) > max_pts:
        idx = np.random.choice(len(gate), max_pts, replace=False)
        gate_sub = gate[idx]
        pos_sub = positions[idx]
    else:
        gate_sub = gate
        pos_sub = positions
    
    for i, (name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        sc = ax.scatter(pos_sub[:, 0], pos_sub[:, 1], pos_sub[:, 2],
                       c=gate_sub, cmap='RdYlBu_r', s=0.5, alpha=0.8,
                       vmin=0.0, vmax=1.0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='RdYlBu_r'), 
                        cax=cbar_ax)
    cbar.set_label('Gate Value (0=geometry, 1=text)', fontsize=12)
    
    save_path = os.path.join(output_dir, 'ggca_gate_heatmap.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
    
    # ---- Figure 3: Gate distribution histogram ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('GGCA Gate Distribution Analysis', fontsize=14, fontweight='bold')
    
    # 3a: Histogram
    ax = axes[0]
    ax.hist(gate, bins=50, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='sigmoid(0)=0.5 (init)')
    ax.axvline(x=gate.mean(), color='green', linestyle='-', alpha=0.7, label=f'mean={gate.mean():.3f}')
    ax.set_xlabel('Gate Value')
    ax.set_ylabel('Count')
    ax.set_title('(a) Gate Value Distribution')
    ax.legend()
    
    # 3b: Gate vs normal direction (z-component)
    if 'normals' in results:
        normals = results['normals']
        ax = axes[1]
        # Use absolute z-component of normal as a proxy for "facing up/down"
        nz = np.abs(normals[:, 2])  # how much the surface faces up/down
        # Subsample for scatter
        if len(gate) > 10000:
            idx = np.random.choice(len(gate), 10000, replace=False)
        else:
            idx = np.arange(len(gate))
        ax.scatter(nz[idx], gate[idx], s=0.5, alpha=0.3, c='steelblue')
        ax.set_xlabel('|Normal Z| (vertical alignment)')
        ax.set_ylabel('Gate Value')
        ax.set_title('(b) Gate vs Surface Orientation')
        
        # Also compute correlation
        from scipy import stats as scipy_stats
        corr, pval = scipy_stats.pearsonr(nz, gate)
        ax.text(0.05, 0.95, f'r={corr:.3f}, p={pval:.2e}', transform=ax.transAxes,
                va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes[1].text(0.5, 0.5, 'No normals available', ha='center', va='center', transform=axes[1].transAxes)
    
    # 3c: Gate magnitude of text influence (||gate * h||)
    if 'gate_times_h' in results:
        influence = np.linalg.norm(results['gate_times_h'], axis=-1)  # [N]
        ax = axes[2]
        ax.hist(influence, bins=50, color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('||gate × h|| (Text Influence Magnitude)')
        ax.set_ylabel('Count')
        ax.set_title(f'(c) Text Influence (mean={influence.mean():.4f})')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ggca_gate_distribution.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def visualize_ggca_attention(results, text_prompt, output_dir, dpi=200):
    """
    Visualize cross-attention weights: which text tokens each spatial point attends to.
    """
    if 'attn_weights' not in results or not results['attn_weights']:
        print("[WARN] No attention weights to visualize")
        return
    
    # Take first batch item
    attn = results['attn_weights'][0]  # [N_points, seq_len]
    gate = results['gate'].squeeze(-1)
    positions = results['positions']
    
    print(f"\n=== GGCA Attention Analysis ===")
    print(f"  Attention shape: {attn.shape}")
    
    # Get text tokens for labeling
    words = text_prompt.split()
    # Approximate: each word ≈ 1-2 tokens. Show first N words.
    n_tokens = attn.shape[1]
    
    # ---- Figure 4: Mean attention across all points ----
    mean_attn = attn.mean(axis=0)  # [seq_len]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GGCA Cross-Attention Analysis', fontsize=14, fontweight='bold')
    
    # 4a: Mean attention over token positions
    ax = axes[0, 0]
    display_len = min(80, len(mean_attn))  # show first 80 tokens
    ax.bar(range(display_len), mean_attn[:display_len], color='steelblue', alpha=0.8)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('(a) Mean Attention per Token Position')
    # Mark approximate word boundaries
    if len(words) <= 20:
        word_positions = list(range(1, min(len(words) + 1, display_len)))
        ax.set_xticks(word_positions)
        ax.set_xticklabels(words[:len(word_positions)], rotation=45, ha='right', fontsize=7)
    
    # 4b: Attention entropy distribution
    ax = axes[0, 1]
    # Entropy of attention distribution per point
    attn_clipped = np.clip(attn, 1e-10, 1.0)
    entropy = -np.sum(attn_clipped * np.log(attn_clipped), axis=-1)  # [N]
    max_entropy = np.log(n_tokens)
    ax.hist(entropy, bins=50, color='mediumpurple', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axvline(x=max_entropy, color='red', linestyle='--', alpha=0.5, label=f'Uniform (max={max_entropy:.1f})')
    ax.axvline(x=entropy.mean(), color='green', linestyle='-', alpha=0.7, label=f'Mean={entropy.mean():.2f}')
    ax.set_xlabel('Attention Entropy')
    ax.set_ylabel('Count')
    ax.set_title('(b) Attention Entropy Distribution')
    ax.legend()
    
    # 4c: Top-attended tokens (by mean attention)
    ax = axes[1, 0]
    top_k = min(20, len(mean_attn))
    top_indices = np.argsort(mean_attn)[::-1][:top_k]
    ax.barh(range(top_k), mean_attn[top_indices], color='coral', alpha=0.8)
    labels = [f'Token {i}' for i in top_indices]
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Attention Weight')
    ax.set_title(f'(c) Top-{top_k} Attended Token Positions')
    
    # 4d: Attention heatmap for a sample of points (sorted by gate value)
    ax = axes[1, 1]
    n_sample = min(100, len(attn))
    # Sort by gate value to see if high-gate points have different attention
    gate_sorted_idx = np.argsort(gate)
    # Take evenly spaced samples from sorted
    sample_idx = gate_sorted_idx[np.linspace(0, len(gate_sorted_idx)-1, n_sample, dtype=int)]
    attn_sample = attn[sample_idx, :display_len]
    
    im = ax.imshow(attn_sample, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Token Position')
    ax.set_ylabel(f'Points (sorted by gate: low→high)')
    ax.set_title('(d) Attention Heatmap (sampled points)')
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ggca_attention_analysis.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# 3. Ablation: With vs Without Text Conditioning
# ============================================================

def visualize_ablation(model, input_data, octree, text_embedding, text_embedding_ggca, 
                       device, output_dir, opt, dpi=200):
    """
    Compare model output with and without text conditioning.
    Shows the difference in predicted Gaussian parameters.
    """
    print("\n=== Ablation: Text Conditioning Effect ===")
    
    ema_model = model.ema_model if hasattr(model, 'ema_model') else model.model
    
    with torch.no_grad():
        mesh_normal_prior = None
        if input_data.shape[1] >= 3:
            mesh_normal_prior = F.normalize(input_data[:, :3], dim=-1, eps=1e-6)
        normals = mesh_normal_prior if hasattr(model, 'use_ggca') and model.use_ggca else None
        
        cond_ggca = text_embedding_ggca if text_embedding_ggca is not None else text_embedding
        
        # With text
        out_with_text = ema_model(input_data, octree, text_embedding, normals=normals, condition_ggca=cond_ggca)
        
        # Without text: use zero embeddings (since CrossAttention doesn't handle None)
        zero_text = torch.zeros_like(text_embedding)
        out_no_text = ema_model(input_data, octree, zero_text, normals=normals, condition_ggca=zero_text)
    
    diff = (out_with_text - out_no_text).cpu().numpy()  # [N, out_channels]
    out_w = out_with_text.cpu().numpy()
    out_n = out_no_text.cpu().numpy()
    positions = octree.position.detach().cpu().numpy()
    
    diff_norm = np.linalg.norm(diff, axis=-1)  # [N]
    
    print(f"  Output channels: {diff.shape[1]}")
    print(f"  Mean output diff (L2): {diff_norm.mean():.6f}")
    print(f"  Max output diff (L2): {diff_norm.max():.6f}")
    print(f"  Std output diff (L2): {diff_norm.std():.6f}")
    
    # Channel-wise analysis
    # out_channels: [opacity(1), scale(3), rotation(skip), rgb(3), material(3)] = 10 or more
    # Based on forward_gaussians: base_out is [N, out_channels]
    # After cat: [opacity(1), scale(3), zeros(4), rgb(3), material(3)]
    # But base_out directly: first 4 = opacity+scale, then 4+rgb(3)=7, then material(3)=10
    channel_names = ['Opacity', 'Scale_x', 'Scale_y', 'Scale_z', 
                     'R', 'G', 'B']
    if diff.shape[1] > 7:
        channel_names += ['Roughness', 'Metallic_G', 'Metallic_B']
    if diff.shape[1] > 10:
        channel_names += [f'Ch_{i}' for i in range(10, diff.shape[1])]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ablation: Effect of Text Conditioning on Model Output\n'
                 f'(mean L2 diff: {diff_norm.mean():.4f})', fontsize=14, fontweight='bold')
    
    # 5a: Per-channel mean absolute difference
    ax = axes[0, 0]
    ch_diff = np.abs(diff).mean(axis=0)  # [out_channels]
    n_ch = min(len(channel_names), len(ch_diff))
    ax.bar(range(n_ch), ch_diff[:n_ch], color='steelblue', alpha=0.8)
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(channel_names[:n_ch], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean |Δ|')
    ax.set_title('(a) Per-Channel Mean Absolute Difference')
    
    # 5b: Spatial distribution of diff (3D scatter)
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    max_pts = 30000
    if len(diff_norm) > max_pts:
        idx = np.random.choice(len(diff_norm), max_pts, replace=False)
    else:
        idx = np.arange(len(diff_norm))
    sc = ax.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2],
                   c=diff_norm[idx], cmap='hot', s=0.5, alpha=0.8)
    ax.set_title('(b) Text Effect Magnitude on Points')
    ax.view_init(elev=30, azim=45)
    plt.colorbar(sc, ax=ax, label='||Δoutput||', shrink=0.6)
    
    # 5c: RGB channel difference specifically 
    ax = axes[1, 0]
    if diff.shape[1] >= 7:
        rgb_diff = np.abs(diff[:, 4:7]).mean(axis=-1)  # mean across RGB
    else:
        rgb_diff = diff_norm
    ax.hist(rgb_diff, bins=50, color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Mean |ΔRGB|')
    ax.set_ylabel('Count')
    ax.set_title('(c) Distribution of RGB Prediction Difference')
    
    # 5d: Material channel difference
    ax = axes[1, 1]
    if diff.shape[1] >= 10:
        mat_diff = np.abs(diff[:, 7:10]).mean(axis=-1)
        ax.hist(mat_diff, bins=50, color='mediumpurple', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Mean |ΔMaterial|')
        ax.set_ylabel('Count')
        ax.set_title('(d) Distribution of Material Prediction Difference')
    else:
        ax.text(0.5, 0.5, 'No material channels', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ablation_text_effect.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# 4. Weight analysis (no forward pass needed)
# ============================================================

def visualize_weight_analysis(model, output_dir, dpi=200):
    """
    Analyze the learned weights of TextAdapter and GGCA to show they have
    meaningfully diverged from initialization.
    """
    print("\n=== Weight Analysis ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Module Weight Analysis: TextAdapter & GGCA', fontsize=14, fontweight='bold')
    
    ema_model = model.ema_model if hasattr(model, 'ema_model') else model.model
    
    # TextAdapter weights
    adapter = None
    if hasattr(model, 'ema_text_adapter') and model.ema_text_adapter is not None:
        adapter = model.ema_text_adapter
    elif hasattr(model, 'text_adapter') and model.text_adapter is not None:
        adapter = model.text_adapter
    
    if adapter is not None:
        # Get all adapter linear weights
        weight_data = {}
        for name, param in adapter.named_parameters():
            weight_data[name] = param.detach().cpu().numpy().flatten()
        
        ax = axes[0, 0]
        for name, w in weight_data.items():
            if 'weight' in name:
                ax.hist(w, bins=50, alpha=0.6, label=name, density=True)
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        ax.set_title('(a) TextAdapter Weight Distribution')
        ax.legend(fontsize=7)
        
        # Scale parameter
        ax = axes[0, 1]
        scale_val = adapter.scale.item()
        ax.barh(['Learned', 'Initial'], [scale_val, 0.1], color=['steelblue', 'lightcoral'])
        ax.set_xlabel('Scale Value')
        ax.set_title(f'(b) TextAdapter Scale: {scale_val:.4f} (init=0.1)')
        for i, v in enumerate([scale_val, 0.1]):
            ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=10)
    
    # GGCA weights
    ggca = ema_model.ggca if hasattr(ema_model, 'ggca') else None
    if ggca is not None:
        # Gate network weights
        gate_weights = {}
        for name, param in ggca.gate_net.named_parameters():
            gate_weights[name] = param.detach().cpu().numpy().flatten()
        
        ax = axes[0, 2]
        for name, w in gate_weights.items():
            if 'weight' in name:
                ax.hist(w, bins=50, alpha=0.6, label=f'gate.{name}', density=True)
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        ax.set_title('(c) GGCA Gate Network Weights')
        ax.legend(fontsize=7)
        
        # proj_out weights (should have diverged from zero-init)
        ax = axes[1, 0]
        if ggca.use_proj:
            proj_out_w = ggca.proj_out.weight.detach().cpu().numpy().flatten()
            ax.hist(proj_out_w, bins=50, color='seagreen', alpha=0.8)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero (init)')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Count')
            ax.set_title(f'(d) GGCA proj_out Weights (init=0, std={proj_out_w.std():.6f})')
            ax.legend()
        
        # FFN last layer weights
        ax = axes[1, 1]
        ffn_last_w = ggca.ffn[-1].weight.detach().cpu().numpy().flatten()
        ax.hist(ffn_last_w, bins=50, color='coral', alpha=0.8)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero (init)')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.set_title(f'(e) GGCA FFN Last Layer (init=0, std={ffn_last_w.std():.6f})')
        ax.legend()
        
        # Gate last layer bias
        ax = axes[1, 2]
        gate_last = ggca.gate_net[-1]
        gate_bias = gate_last.bias.item() if gate_last.bias is not None else 0.0
        gate_last_w = gate_last.weight.detach().cpu().numpy().flatten()
        ax.hist(gate_last_w, bins=30, color='mediumpurple', alpha=0.8, label=f'weights (std={gate_last_w.std():.4f})')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero (init)')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.set_title(f'(f) Gate Final Layer (bias={gate_bias:.4f}, init=0.0)')
        ax.legend()
        
        # Print summary
        print(f"  GGCA proj_out weight std: {proj_out_w.std():.6f} (init: 0.0)")
        print(f"  GGCA FFN last weight std: {ffn_last_w.std():.6f} (init: 0.0)")
        print(f"  GGCA gate final bias: {gate_bias:.4f} (init: 0.0)")
        print(f"  GGCA gate final weight std: {gate_last_w.std():.6f} (init: 0.0)")
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'weight_analysis.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # ---- Build model ----
    print("Loading model...")
    from core.options import Options
    opt = Options()
    
    # Set essential options
    opt.use_text = 'True'
    opt.use_longclip = 'True'
    opt.use_text_adapter = 'True'
    opt.use_ggca = 'True'
    opt.use_material = 'True'
    opt.longclip_model = 'third_party/Long-CLIP/checkpoints/longclip-L.pt'
    opt.longclip_context_length = 248
    opt.context_dim = 768
    opt.force_cuda_rast = True
    opt.pointcloud_dir = args.pointcloud_dir
    opt.use_local_pretrained_ckpt = 'False'
    opt.lambda_lpips = 0  # skip LPIPS for visualization
    
    model = __import__('core.regression_models', fromlist=['TexGaussian']).TexGaussian(opt, device)
    model.to(device)
    model.eval()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt_path}")
    from safetensors.torch import load_file
    if args.ckpt_path.endswith('.safetensors'):
        ckpt = load_file(args.ckpt_path, device='cpu')
    else:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
    
    state_dict = model.state_dict()
    loaded = 0
    for k, v in ckpt.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k].copy_(v)
            loaded += 1
    print(f"  Loaded {loaded}/{len(ckpt)} parameters")
    
    # ---- 1. Weight Analysis (no data needed) ----
    visualize_weight_analysis(model, args.output_dir, dpi=args.dpi)
    
    # ---- 2. TextAdapter Analysis ----
    # Use the text encoder already loaded in the model
    text_encoder = model.text_encoder if hasattr(model, 'text_encoder') else None
    
    # We need LongCLIP tokenizer to encode text properly
    longclip_model = None
    longclip_tokenize = None
    if opt.use_longclip == 'True' or opt.use_longclip is True:
        try:
            from texture import load_longclip_model as _load_lc
            lc_model, _, lc_tok = _load_lc(opt.longclip_model, device)
            if hasattr(lc_model, 'visual'):
                del lc_model.visual
            lc_model.requires_grad_(False)
            lc_model.eval()
            longclip_model = lc_model
            longclip_tokenize = lc_tok
        except Exception as e:
            print(f"[WARN] Failed to load LongCLIP from texture.py: {e}")
            # Fallback: try loading directly
            try:
                from core.longclip_utils import load_longclip_model as _load_lc2
                model_path = opt.longclip_model
                if not os.path.isabs(model_path):
                    model_path = os.path.join(PROJECT_ROOT, model_path)
                lc_model, _ = _load_lc2(model_path, device)
                if hasattr(lc_model, 'visual'):
                    del lc_model.visual
                lc_model.requires_grad_(False)
                lc_model.eval()
                longclip_model = lc_model
                # Get tokenizer
                sys.path.insert(0, os.path.join(PROJECT_ROOT, 'third_party', 'Long-CLIP'))
                from model import longclip as lc_mod
                longclip_tokenize = lc_mod.tokenize
            except Exception as e2:
                print(f"[WARN] Failed to load LongCLIP fallback: {e2}")
    
    adapter_stats = visualize_text_adapter(
        model, text_encoder, longclip_model, longclip_tokenize,
        args.text_prompt, device, args.output_dir, dpi=args.dpi
    )
    
    # ---- 3. GGCA Analysis (needs mesh / point cloud) ----
    if args.mesh_path and os.path.isfile(args.mesh_path):
        print(f"\nLoading mesh: {args.mesh_path}")
        import trimesh
        
        mesh = trimesh.load(args.mesh_path, force='mesh')
        # Normalize
        mesh.vertices = mesh.vertices - mesh.bounding_box.centroid
        distances = np.linalg.norm(mesh.vertices, axis=1)
        mesh.vertices /= np.max(distances)
        
        # Sample point cloud
        num_samples = 200000
        point, idx = trimesh.sample.sample_surface(mesh, num_samples)
        tri_verts = mesh.vertices[mesh.faces[idx]]
        vertex_normals = mesh.vertex_normals[mesh.faces[idx]]
        
        # Interpolate normals
        from texture import Converter
        normals = Converter._interpolate_vertex_normals(point, tri_verts, vertex_normals)
        
        from ocnn.octree import Points as OcPoints
        import ocnn
        
        points_gt = OcPoints(
            points=torch.from_numpy(point).float(),
            normals=torch.from_numpy(normals).float()
        )
        points_gt.clip(min=-1, max=1)
        points_gt = points_gt.cuda(non_blocking=True)
        
        # Build octree same as Converter.load_mesh — use merge_octrees for proper GPU placement
        def _pts2octree(pts):
            oc = ocnn.octree.Octree(depth=opt.input_depth, full_depth=opt.full_depth)
            oc.build_octree(pts)
            return oc
        
        octree = ocnn.octree.merge_octrees([_pts2octree(points_gt)])
        octree.construct_all_neigh()
        
        xyzb = octree.xyzb(depth=octree.depth, nempty=True)
        x, y, z, b = xyzb
        xyz = torch.stack([x, y, z], dim=1)
        octree.position = 2 * xyz / (2 ** octree.depth) - 1
        
        input_data = octree.get_input_feature(feature=opt.input_feature, nempty=True)
        
        # Get text embeddings
        if longclip_model is not None and longclip_tokenize is not None:
            context_length = opt.longclip_context_length
            token = longclip_tokenize(args.text_prompt, context_length=context_length, truncate=True)
            token = token.to(device)
            with torch.no_grad():
                x = longclip_model.token_embedding(token)
                x = x + longclip_model.positional_embedding[:x.shape[1]]
                x = x.permute(1, 0, 2)
                x = longclip_model.transformer(x)
                x = x.permute(1, 0, 2)
                x = longclip_model.ln_final(x)
                raw_text = x.float()
        else:
            print("[WARN] No text encoder available, skipping GGCA visualization")
            raw_text = None
        
        if raw_text is not None:
            # Apply adapter
            adapter = None
            if hasattr(model, 'ema_text_adapter') and model.ema_text_adapter is not None:
                adapter = model.ema_text_adapter
            elif hasattr(model, 'text_adapter') and model.text_adapter is not None:
                adapter = model.text_adapter
            
            if adapter is not None:
                with torch.no_grad():
                    adapted_text = adapter(raw_text)
            else:
                adapted_text = raw_text
            
            text_embedding = adapted_text  # In v6 config, both CA and GGCA use adapted
            text_embedding_ggca = adapted_text
            
            # Extract GGCA intermediates
            ggca_results = extract_ggca_intermediates(
                model, input_data, octree, text_embedding, text_embedding_ggca, device
            )
            
            if ggca_results:
                visualize_ggca_gate(ggca_results, args.output_dir, dpi=args.dpi)
                visualize_ggca_attention(ggca_results, args.text_prompt, args.output_dir, dpi=args.dpi)
            
            # Ablation
            visualize_ablation(model, input_data, octree, text_embedding, text_embedding_ggca,
                             device, args.output_dir, opt, dpi=args.dpi)
    else:
        print(f"\n[INFO] No mesh path provided or file not found, skipping GGCA spatial visualization")
        print(f"  Use --mesh_path to enable gate heatmap and attention visualization")
    
    # ---- Save summary ----
    import json
    summary = {
        'checkpoint': args.ckpt_path,
        'mesh': args.mesh_path,
        'text_prompt': args.text_prompt,
        'adapter_stats': adapter_stats if adapter_stats else {},
    }
    summary_path = os.path.join(args.output_dir, 'visualization_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
