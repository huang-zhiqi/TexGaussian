#!/usr/bin/env python3
"""
Intuitive visualization of TextAdapter & GGCA attention maps.

Given a text prompt + 3D mesh, produces publication-ready figures showing:
1. Which WORDS in the prompt receive the most attention from the 3D model
2. For each keyword, which REGIONS on the 3D surface attend to it
3. TextAdapter: how each word's embedding is modified
4. Combined text-to-surface mapping figure

Usage:
    python scripts/visualize_attention_map.py \
        --ckpt_path  <checkpoint.safetensors> \
        --mesh_path  <mesh.obj> \
        --text_prompt "A retro-style video game cartridge..." \
        --output_dir <output_dir>
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.patches import FancyArrowPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def get_tokenizer():
    """Get the BPE tokenizer (shared by CLIP / LongCLIP)."""
    tp = os.path.join(PROJECT_ROOT, 'third_party', 'Long-CLIP')
    if tp not in sys.path:
        sys.path.insert(0, tp)
    from model.simple_tokenizer import SimpleTokenizer
    return SimpleTokenizer()


def tokens_to_words(tokenizer, token_ids):
    """
    Convert a 1-D tensor of BPE token IDs into a list of (word, [token_indices]).
    Returns list of (word_str, list_of_token_positions).
    """
    sot = tokenizer.encoder.get("<|startoftext|>", None)
    eot = tokenizer.encoder.get("<|endoftext|>", None)

    words = []
    current_word_pieces = []
    current_indices = []

    for idx, tid in enumerate(token_ids):
        tid = int(tid)
        if tid == 0:          # padding
            break
        if tid == sot or tid == eot:
            # store special token as its own "word"
            words.append(("[SOS]" if tid == sot else "[EOS]", [idx]))
            continue

        piece = tokenizer.decoder.get(tid, f"<unk:{tid}>")
        current_word_pieces.append(piece)
        current_indices.append(idx)

        if piece.endswith("</w>"):
            word = "".join(current_word_pieces).replace("</w>", "")
            # Decode byte-level encoding back to readable string
            try:
                word = bytearray([tokenizer.byte_decoder[c] for c in word]).decode('utf-8', errors='replace')
            except Exception:
                pass
            words.append((word, list(current_indices)))
            current_word_pieces = []
            current_indices = []

    # Leftover (shouldn't happen normally)
    if current_word_pieces:
        word = "".join(current_word_pieces).replace("</w>", "")
        try:
            word = bytearray([tokenizer.byte_decoder[c] for c in word]).decode('utf-8', errors='replace')
        except Exception:
            pass
        words.append((word, list(current_indices)))

    return words


def aggregate_attn_to_words(attn_weights, word_groups):
    """
    attn_weights : [N_points, seq_len]
    word_groups  : list of (word_str, [token_indices])
    Returns       : [N_points, n_words],  word_labels : list[str]
    """
    n_points = attn_weights.shape[0]
    n_words = len(word_groups)
    word_attn = np.zeros((n_points, n_words), dtype=np.float32)
    labels = []
    for w_idx, (word, indices) in enumerate(word_groups):
        word_attn[:, w_idx] = attn_weights[:, indices].sum(axis=1)
        labels.append(word)
    return word_attn, labels


# ──────────────────────────────────────────────
# Model loading (same approach as visualize_modules.py)
# ──────────────────────────────────────────────

def load_model_and_data(args):
    """Load checkpoint, build octree from mesh, encode text. Returns dict."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- Options ---
    from core.options import Options
    opt = Options()
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
    opt.lambda_lpips = 0

    # --- Model ---
    from core.regression_models import TexGaussian
    model = TexGaussian(opt, device)
    model.to(device); model.eval()

    from safetensors.torch import load_file
    ckpt = load_file(args.ckpt_path, device='cpu') if args.ckpt_path.endswith('.safetensors') \
        else torch.load(args.ckpt_path, map_location='cpu')
    state = model.state_dict()
    loaded = sum(1 for k, v in ckpt.items() if k in state and state[k].shape == v.shape and not state[k].copy_(v) is None)
    print(f"  Loaded {loaded}/{len(ckpt)} params")

    # --- LongCLIP ---
    from texture import load_longclip_model
    lc_model, _, lc_tok = load_longclip_model(opt.longclip_model, device)
    if hasattr(lc_model, 'visual'): del lc_model.visual
    lc_model.requires_grad_(False); lc_model.eval()

    # --- Tokenize ---
    context_length = opt.longclip_context_length
    token_tensor = lc_tok(args.text_prompt, context_length=context_length, truncate=True).to(device)

    # --- Encode text ---
    with torch.no_grad():
        x = lc_model.token_embedding(token_tensor)
        x = x + lc_model.positional_embedding[:x.shape[1]]
        x = x.permute(1, 0, 2)
        x = lc_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = lc_model.ln_final(x)
        raw_text_emb = x.float()       # [1, 248, 768]

    # --- TextAdapter ---
    adapter = getattr(model, 'ema_text_adapter', None) or getattr(model, 'text_adapter', None)
    if adapter is not None:
        with torch.no_grad():
            adapted_text_emb = adapter(raw_text_emb)
    else:
        adapted_text_emb = raw_text_emb

    # --- Mesh → Octree ---
    import trimesh, ocnn
    from ocnn.octree import Points as OcPoints

    mesh = trimesh.load(args.mesh_path, force='mesh')
    mesh.vertices = mesh.vertices - mesh.bounding_box.centroid
    mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()

    pts, idx = trimesh.sample.sample_surface(mesh, 200000)
    tri_v = mesh.vertices[mesh.faces[idx]]
    tri_n = mesh.vertex_normals[mesh.faces[idx]]
    from texture import Converter
    normals = Converter._interpolate_vertex_normals(pts, tri_v, tri_n)

    pts_oc = OcPoints(points=torch.from_numpy(pts).float(), normals=torch.from_numpy(normals).float())
    pts_oc.clip(min=-1, max=1)
    pts_oc = pts_oc.cuda(non_blocking=True)

    def _p2o(p):
        o = ocnn.octree.Octree(depth=opt.input_depth, full_depth=opt.full_depth); o.build_octree(p); return o
    octree = ocnn.octree.merge_octrees([_p2o(pts_oc)])
    octree.construct_all_neigh()
    xyzb = octree.xyzb(depth=octree.depth, nempty=True)
    xyz = torch.stack(xyzb[:3], dim=1)
    octree.position = 2 * xyz / (2 ** octree.depth) - 1

    input_data = octree.get_input_feature(feature=opt.input_feature, nempty=True)

    return dict(model=model, opt=opt, device=device,
                octree=octree, input_data=input_data,
                raw_text_emb=raw_text_emb, adapted_text_emb=adapted_text_emb,
                token_tensor=token_tensor, adapter=adapter)


# ──────────────────────────────────────────────
# Extract GGCA intermediates (hook)
# ──────────────────────────────────────────────

def extract_ggca(model, input_data, octree, text_emb, device):
    """Run forward with hook, return gate [N,1], attn [N, seq_len], positions [N,3]."""
    ema = model.ema_model if hasattr(model, 'ema_model') else model.model
    ggca = getattr(ema, 'ggca', None)
    if ggca is None:
        raise RuntimeError("GGCA module not found in model")

    results = {}
    orig = ggca.forward

    def _hooked(data, octree, depth, context=None, normals=None):
        if context is None:
            return data
        h = ggca.proj_in(data) if ggca.use_proj else data
        h_norm = ggca.cross_norm(h)
        bid = octree.batch_id(depth=depth, nempty=True)
        parts, ws = [], []
        for i in range(octree.batch_size):
            m = (bid == i)
            o, w = ggca.cross_attn(h_norm[m].unsqueeze(0), context[i:i+1], context[i:i+1],
                                    need_weights=True, average_attn_weights=True)
            parts.append(o.squeeze(0)); ws.append(w.squeeze(0))
        attn_out = torch.cat(parts, 0)
        h = h + attn_out
        h = h + ggca.ffn(ggca.ffn_norm(h))
        if ggca.use_proj:
            h = ggca.proj_out(h)
        if normals is not None:
            n = F.normalize(normals, dim=-1, eps=1e-6)
            gate = torch.sigmoid(ggca.gate_net(torch.cat([data, n], -1)))
        else:
            gate = torch.full((data.shape[0], 1), 0.5, device=data.device)
        results['gate'] = gate.cpu().numpy()
        results['attn'] = torch.cat(ws, 0).cpu().numpy()   # [N, seq]
        return data + gate * h

    ggca.forward = _hooked
    with torch.no_grad():
        nrm = F.normalize(input_data[:, :3], dim=-1, eps=1e-6) if input_data.shape[1] >= 3 else None
        ema(input_data, octree, text_emb, normals=nrm, condition_ggca=text_emb)
    ggca.forward = orig
    results['positions'] = octree.position.cpu().numpy()
    return results


# ──────────────────────────────────────────────
# Figure 1: Word-level Attention Summary
# ──────────────────────────────────────────────

def fig_word_attention(word_attn, word_labels, output_dir, dpi=200):
    """Bar chart: which words the entire 3D model attends to most."""
    mean_attn = word_attn.mean(axis=0)   # [n_words]

    # Remove special tokens for cleaner display
    keep = [(i, l, a) for i, (l, a) in enumerate(zip(word_labels, mean_attn))
            if l not in ('[SOS]', '[EOS]')]
    if not keep:
        return
    indices, labels, values = zip(*keep)
    values = np.array(values)

    # Normalize for color
    v_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(max(len(labels) * 0.55, 8), 4))
    cmap = cm.get_cmap('YlOrRd')
    bars = ax.bar(range(len(labels)), values, color=[cmap(v) for v in v_norm],
                  edgecolor='#333', linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=9, fontweight='bold')
    ax.set_ylabel('Mean Cross-Attention Weight', fontsize=11)
    ax.set_title('Which words does the 3D model attend to?', fontsize=13, fontweight='bold')

    # Annotate top-3
    top3 = np.argsort(values)[-3:][::-1]
    for rank, idx in enumerate(top3):
        ax.annotate(f'#{rank+1}', (idx, values[idx]),
                    textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=8, fontweight='bold', color='darkred')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_word_attention.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Fig 1] Saved: {path}")


# ──────────────────────────────────────────────
# Figure 2: Highlighted Text Prompt
# ──────────────────────────────────────────────

def fig_text_highlight(word_attn, word_labels, output_dir, dpi=200):
    """Render the text prompt with background colour proportional to attention."""
    mean_attn = word_attn.mean(axis=0)

    # Filter special tokens
    keep = [(l, a) for l, a in zip(word_labels, mean_attn) if l not in ('[SOS]', '[EOS]')]
    if not keep:
        return
    labels, vals = zip(*keep)
    vals = np.array(vals)
    v_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)

    cmap = cm.get_cmap('YlOrRd')

    fig, ax = plt.subplots(figsize=(14, 3.0))
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Layout words as wrapped text with colored backgrounds
    x, y = 0.02, 0.75
    line_h = 0.22
    font_size = 13

    for i, (word, v) in enumerate(zip(labels, v_norm)):
        color = cmap(v)
        txt = ax.text(x, y, f' {word} ', fontsize=font_size,
                      fontfamily='monospace', fontweight='bold',
                      va='center', ha='left',
                      bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.85, edgecolor='#666'))

        fig.canvas.draw()
        bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
        bb_data = bb.transformed(ax.transData.inverted())
        x = bb_data.x1 + 0.005
        if x > 0.92:
            x = 0.02
            y -= line_h

    ax.set_title('Text Prompt Attention Heatmap (yellow→red = more attended)',
                 fontsize=12, fontweight='bold', pad=15)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vals.min(), vals.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.15,
                        aspect=40)
    cbar.set_label('Attention Weight', fontsize=10)

    path = os.path.join(output_dir, 'fig2_text_highlight.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Fig 2] Saved: {path}")


# ──────────────────────────────────────────────
# Figure 3: Per-Keyword 3D Heatmap
# ──────────────────────────────────────────────

def fig_keyword_3d(word_attn, word_labels, positions, gate,
                   output_dir, dpi=200, n_keywords=6):
    """For top keywords, show 3D point cloud colored by attention to that word."""
    mean_attn = word_attn.mean(axis=0)
    # Pick top keywords (skip specials)
    scored = [(i, l, a) for i, (l, a) in enumerate(zip(word_labels, mean_attn))
              if l not in ('[SOS]', '[EOS]') and len(l) > 1]
    scored.sort(key=lambda x: -x[2])
    top = scored[:n_keywords]

    n = len(top)
    if n == 0:
        return
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(6 * ncols, 5.5 * nrows))
    fig.suptitle('3D Surface Attention per Keyword\n(brighter = stronger attention to that word)',
                 fontsize=14, fontweight='bold', y=1.02)

    # Subsample points for speed
    max_pts = 40000
    if len(positions) > max_pts:
        idx = np.random.RandomState(42).choice(len(positions), max_pts, replace=False)
    else:
        idx = np.arange(len(positions))
    pos = positions[idx]

    for k, (w_idx, word, _) in enumerate(top):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection='3d')
        vals = word_attn[idx, w_idx]
        vmin, vmax = np.percentile(vals, [2, 98])
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                        c=vals, cmap='hot', s=0.4, alpha=0.85,
                        vmin=vmin, vmax=vmax)
        ax.set_title(f'"{word}"', fontsize=13, fontweight='bold', color='darkred')
        ax.view_init(elev=25, azim=135)
        ax.set_xlabel('X', fontsize=8); ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.tick_params(labelsize=6)
        plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.08, label='Attention')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_keyword_3d_heatmap.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Fig 3] Saved: {path}")


# ──────────────────────────────────────────────
# Figure 4: Dominant-Word Map on 3D Surface
# ──────────────────────────────────────────────

def fig_dominant_word_3d(word_attn, word_labels, positions, output_dir, dpi=200):
    """Color each point by the word it attends to most → semantic region map."""
    # Skip special tokens
    real = [(i, l) for i, l in enumerate(word_labels) if l not in ('[SOS]', '[EOS]') and len(l) > 1]
    if not real:
        return
    real_indices, real_labels = zip(*real)
    real_indices = list(real_indices)

    # [N, n_real_words]
    attn_real = word_attn[:, real_indices]
    dominant = attn_real.argmax(axis=1)   # [N] index into real_labels

    # Use only top-K unique dominant words for clearer coloring
    from collections import Counter
    counts = Counter(dominant)
    top_k = min(10, len(counts))
    top_dominant = [w for w, _ in counts.most_common(top_k)]
    # Map: top_dominant index → color index, rest → -1 ("other")
    color_map = {w: i for i, w in enumerate(top_dominant)}

    colors = np.array([color_map.get(d, -1) for d in dominant])
    n_colors = len(top_dominant) + 1  # +1 for "other"

    # Build color palette
    base_cmap = cm.get_cmap('tab10' if n_colors <= 10 else 'tab20')
    palette = [base_cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]
    # Last color for "other"
    palette_arr = np.array(palette)

    # Map colors
    c_idx = np.where(colors >= 0, colors, n_colors - 1)
    pt_colors = palette_arr[c_idx]

    # Subsample
    max_pts = 40000
    rng = np.random.RandomState(42)
    if len(positions) > max_pts:
        idx = rng.choice(len(positions), max_pts, replace=False)
    else:
        idx = np.arange(len(positions))

    fig = plt.figure(figsize=(14, 6))

    for vi, (elev, azim, title) in enumerate([(25, 135, 'View 1'), (25, 45, 'View 2'), (90, 0, 'Top')]):
        ax = fig.add_subplot(1, 3, vi + 1, projection='3d')
        ax.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2],
                   c=pt_colors[idx], s=0.4, alpha=0.85)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=5)

    # Legend
    import matplotlib.patches as mpatches
    handles = []
    for i, wi in enumerate(top_dominant):
        handles.append(mpatches.Patch(color=palette_arr[i], label=real_labels[wi]))
    handles.append(mpatches.Patch(color=palette_arr[-1], label='other'))
    fig.legend(handles=handles, loc='lower center', ncol=min(6, len(handles)),
               fontsize=9, framealpha=0.9, title='Dominant Word', title_fontsize=10)

    fig.suptitle('3D Surface colored by Dominant Attended Word\n(each point colored by the word it attends to most)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    path = os.path.join(output_dir, 'fig4_dominant_word_map.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Fig 4] Saved: {path}")


# ──────────────────────────────────────────────
# Figure 5: GGCA Gate × Attention Combined
# ──────────────────────────────────────────────

def fig_gate_weighted_attention(word_attn, word_labels, positions, gate,
                                output_dir, dpi=200):
    """
    Show gate-weighted attention: which words actually INFLUENCE the output.
    Effective influence = gate × attention (per-point, per-word).
    """
    # gate: [N, 1], word_attn: [N, n_words]
    gate_sq = gate.squeeze(-1)  # [N]
    effective = word_attn * gate_sq[:, None]  # [N, n_words]

    mean_raw = word_attn.mean(0)
    mean_eff = effective.mean(0)

    # Skip specials
    keep = [(i, l) for i, l in enumerate(word_labels) if l not in ('[SOS]', '[EOS]') and len(l) > 1]
    if not keep:
        return
    ki, kl = zip(*keep)
    ki = list(ki)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Raw Attention vs Gate-Weighted Effective Influence',
                 fontsize=13, fontweight='bold')

    for ax, vals, title in [(axes[0], mean_raw[ki], 'Raw Cross-Attention'),
                             (axes[1], mean_eff[ki], 'Gate × Attention (Effective)')]:
        v_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        cmap = cm.get_cmap('YlOrRd')
        ax.barh(range(len(kl)), vals, color=[cmap(v) for v in v_norm],
                edgecolor='#444', linewidth=0.5)
        ax.set_yticks(range(len(kl)))
        ax.set_yticklabels(kl, fontsize=9, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel('Weight')
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig5_gate_weighted_attention.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Fig 5] Saved: {path}")


# ──────────────────────────────────────────────
# Figure 6: TextAdapter Per-Word Embedding Shift
# ──────────────────────────────────────────────

def fig_adapter_per_word(raw_emb, adapted_emb, word_groups, output_dir, dpi=200):
    """Show how much TextAdapter changes each word's embedding."""
    raw = raw_emb[0].cpu().numpy()        # [seq_len, 768]
    adp = adapted_emb[0].cpu().numpy()

    delta_norm = np.linalg.norm(adp - raw, axis=-1)       # [seq_len]
    cos_sim = np.sum(raw * adp, axis=-1) / (np.linalg.norm(raw, axis=-1) * np.linalg.norm(adp, axis=-1) + 1e-8)

    # Aggregate to words
    words, d_vals, c_vals = [], [], []
    for word, indices in word_groups:
        if word in ('[SOS]', '[EOS]'):
            continue
        words.append(word)
        d_vals.append(np.mean(delta_norm[indices]))
        c_vals.append(np.mean(cos_sim[indices]))
    d_vals = np.array(d_vals)
    c_vals = np.array(c_vals)

    fig, axes = plt.subplots(2, 1, figsize=(max(len(words) * 0.55, 8), 8))
    fig.suptitle('TextAdapter: Per-Word Embedding Modification',
                 fontsize=13, fontweight='bold')

    # Embedding shift
    ax = axes[0]
    v_norm = (d_vals - d_vals.min()) / (d_vals.max() - d_vals.min() + 1e-8)
    cmap = cm.get_cmap('Blues')
    ax.bar(range(len(words)), d_vals, color=[cmap(0.3 + 0.7 * v) for v in v_norm],
           edgecolor='#333', linewidth=0.5)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=50, ha='right', fontsize=9, fontweight='bold')
    ax.set_ylabel('L2 Embedding Shift')
    ax.set_title('(a) How much each word is modified by TextAdapter', fontsize=11)

    # Cosine similarity
    ax = axes[1]
    v_norm2 = (c_vals - c_vals.min()) / (c_vals.max() - c_vals.min() + 1e-8)
    cmap2 = cm.get_cmap('RdYlGn')
    ax.bar(range(len(words)), c_vals, color=[cmap2(v) for v in v_norm2],
           edgecolor='#333', linewidth=0.5)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=50, ha='right', fontsize=9, fontweight='bold')
    ax.set_ylabel('Cosine Similarity (raw vs adapted)')
    ax.set_title('(b) Cosine similarity before/after adaptation (lower = more changed)', fontsize=11)
    ax.set_ylim(min(c_vals.min() - 0.005, 0.96), 1.005)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig6_adapter_per_word.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Fig 6] Saved: {path}")


# ──────────────────────────────────────────────
# Figure 7: Combined Overview (paper figure)
# ──────────────────────────────────────────────

def fig_combined_overview(word_attn, word_labels, positions, gate,
                          raw_emb, adapted_emb, word_groups,
                          output_dir, dpi=250):
    """
    A single combined figure suitable for a paper:
    Top row: text highlight + word bar chart
    Bottom row: 3D keyword heatmaps for top-3 words + gate heatmap
    """
    mean_attn = word_attn.mean(axis=0)
    keep = [(i, l, a) for i, (l, a) in enumerate(zip(word_labels, mean_attn))
            if l not in ('[SOS]', '[EOS]') and len(l) > 1]
    keep.sort(key=lambda x: -x[2])

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3,
                           height_ratios=[1, 1.3])

    # ─── Top-left: Text highlight (horizontal colored text) ───
    ax_txt = fig.add_subplot(gs[0, :2])
    ax_txt.axis('off')
    cmap = cm.get_cmap('YlOrRd')
    real = [(l, a) for l, a in zip(word_labels, mean_attn) if l not in ('[SOS]', '[EOS]')]
    if real:
        rl, rv = zip(*real)
        rv = np.array(rv)
        vn = (rv - rv.min()) / (rv.max() - rv.min() + 1e-8)
        x, y = 0.01, 0.78; lh = 0.25
        for w, v in zip(rl, vn):
            t = ax_txt.text(x, y, f' {w} ', fontsize=11, fontfamily='monospace', fontweight='bold',
                            va='center', ha='left',
                            bbox=dict(boxstyle='round,pad=0.12', facecolor=cmap(v), alpha=0.85, edgecolor='#777'))
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
            bb_d = bb.transformed(ax_txt.transData.inverted())
            x = bb_d.x1 + 0.004
            if x > 0.94:
                x = 0.01; y -= lh
    ax_txt.set_title('Text Attention Heatmap', fontsize=12, fontweight='bold')

    # ─── Top-right: Bar chart ───
    ax_bar = fig.add_subplot(gs[0, 2:])
    if keep:
        top_n = min(15, len(keep))
        top = keep[:top_n]
        ti, tl, tv = zip(*top)
        tv = np.array(tv)
        vn = (tv - tv.min()) / (tv.max() - tv.min() + 1e-8)
        ax_bar.barh(range(len(tl)), tv, color=[cmap(v) for v in vn],
                    edgecolor='#444', linewidth=0.5)
        ax_bar.set_yticks(range(len(tl)))
        ax_bar.set_yticklabels(tl, fontsize=10, fontweight='bold')
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel('Attention Weight')
        ax_bar.set_title(f'Top-{top_n} Attended Words', fontsize=12, fontweight='bold')

    # ─── Bottom: 3D heatmaps for top-3 keywords + gate ───
    max_pts = 35000
    rng = np.random.RandomState(42)
    idx = rng.choice(len(positions), min(max_pts, len(positions)), replace=False)
    pos = positions[idx]

    top3 = keep[:3]
    for k, (w_idx, word, _) in enumerate(top3):
        ax3 = fig.add_subplot(gs[1, k], projection='3d')
        vals = word_attn[idx, w_idx]
        vmin, vmax = np.percentile(vals, [2, 98])
        sc = ax3.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                         c=vals, cmap='hot', s=0.3, alpha=0.85,
                         vmin=vmin, vmax=vmax)
        ax3.set_title(f'"{word}"', fontsize=12, fontweight='bold', color='darkred')
        ax3.view_init(elev=25, azim=135)
        ax3.tick_params(labelsize=5)
        plt.colorbar(sc, ax=ax3, shrink=0.5, pad=0.06, label='Attn')

    # Gate heatmap
    ax_g = fig.add_subplot(gs[1, 3], projection='3d')
    g = gate[idx].squeeze(-1)
    sc = ax_g.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=g, cmap='RdYlBu_r', s=0.3, alpha=0.85,
                      vmin=g.min(), vmax=g.max())
    ax_g.set_title('GGCA Gate\n(text trust)', fontsize=12, fontweight='bold', color='navy')
    ax_g.view_init(elev=25, azim=135)
    ax_g.tick_params(labelsize=5)
    plt.colorbar(sc, ax=ax_g, shrink=0.5, pad=0.06, label='Gate')

    fig.suptitle('TextAdapter & GGCA Attention Analysis', fontsize=15, fontweight='bold', y=0.98)
    path = os.path.join(output_dir, 'fig7_combined_overview.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Fig 7] Saved: {path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', required=True)
    # Single-mesh mode
    p.add_argument('--mesh_path', default=None, help='Path to a single .obj mesh')
    p.add_argument('--text_prompt', default=None, help='Text prompt for single mesh')
    # Batch (TSV) mode
    p.add_argument('--tsv_path', default=None, help='Path to TSV file for batch mode')
    p.add_argument('--caption_field', default='caption_long',
                   choices=['caption_short', 'caption_long'],
                   help='Which caption column to use from TSV')
    p.add_argument('--max_samples', type=int, default=-1,
                   help='Max samples to process from TSV (-1 = all)')
    # Common
    p.add_argument('--output_dir', required=True)
    p.add_argument('--pointcloud_dir', default='../datasets/texverse_pointcloud_npz')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--n_keywords', type=int, default=6, help='Number of keyword 3D heatmaps')
    args = p.parse_args()

    # Validate: must provide either (mesh_path + text_prompt) or tsv_path
    if args.tsv_path is None and (args.mesh_path is None or args.text_prompt is None):
        p.error("Provide either --tsv_path for batch mode, or both --mesh_path and --text_prompt for single mode.")
    return args


def process_single(args, mesh_path, text_prompt, output_dir):
    """Process a single mesh+text pair and save figures to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # Override args for load_model_and_data
    args.mesh_path = mesh_path
    args.text_prompt = text_prompt

    # ── Load ──
    print("[1/4] Loading model, mesh, text...")
    ctx = load_model_and_data(args)

    # ── Tokenize → word mapping ──
    print("[2/4] Building word ↔ token mapping...")
    tokenizer = get_tokenizer()
    token_ids = ctx['token_tensor'][0]  # [248]
    word_groups = tokens_to_words(tokenizer, token_ids)
    print(f"  {len(word_groups)} word groups from {int((token_ids > 0).sum())} tokens")
    for w, indices in word_groups[:10]:
        print(f"    '{w}' → tokens {indices}")
    if len(word_groups) > 10:
        print(f"    ... and {len(word_groups) - 10} more")

    # ── Extract GGCA ──
    print("[3/4] Extracting GGCA attention & gate (forward pass)...")
    ggca_res = extract_ggca(ctx['model'], ctx['input_data'], ctx['octree'],
                            ctx['adapted_text_emb'], ctx['device'])
    attn = ggca_res['attn']          # [N, 248]
    gate = ggca_res['gate']          # [N, 1]
    positions = ggca_res['positions']  # [N, 3]
    print(f"  Points: {len(positions)}, Tokens: {attn.shape[1]}")
    print(f"  Gate: mean={gate.mean():.4f}, std={gate.std():.4f}")

    # Aggregate attention to word level
    word_attn, word_labels = aggregate_attn_to_words(attn, word_groups)  # [N, n_words]

    # ── Generate figures ──
    print("[4/4] Generating figures...")
    fig_word_attention(word_attn, word_labels, output_dir, args.dpi)
    fig_text_highlight(word_attn, word_labels, output_dir, args.dpi)
    fig_keyword_3d(word_attn, word_labels, positions, gate,
                   output_dir, args.dpi, args.n_keywords)
    fig_dominant_word_3d(word_attn, word_labels, positions, output_dir, args.dpi)
    fig_gate_weighted_attention(word_attn, word_labels, positions, gate,
                                output_dir, args.dpi)
    fig_adapter_per_word(ctx['raw_text_emb'], ctx['adapted_text_emb'],
                         word_groups, output_dir, args.dpi)
    fig_combined_overview(word_attn, word_labels, positions, gate,
                          ctx['raw_text_emb'], ctx['adapted_text_emb'],
                          word_groups, output_dir, args.dpi)

    # ── Summary ──
    summary = {
        'text_prompt': text_prompt,
        'mesh': mesh_path,
        'checkpoint': args.ckpt_path,
        'n_points': len(positions),
        'n_tokens': int((token_ids > 0).sum()),
        'n_words': len(word_groups),
        'gate_mean': float(gate.mean()),
        'gate_std': float(gate.std()),
        'top_attended_words': [],
    }
    mean_wa = word_attn.mean(0)
    scored = sorted([(l, float(mean_wa[i])) for i, l in enumerate(word_labels)
                     if l not in ('[SOS]', '[EOS]')], key=lambda x: -x[1])
    summary['top_attended_words'] = scored[:10]

    with open(os.path.join(output_dir, 'attention_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Saved 7 figures to: {output_dir}")
    print(f"  Top-3: {', '.join(w for w, _ in scored[:3])}")
    return summary


def load_tsv(tsv_path, caption_field, max_samples):
    """Load TSV and return list of (obj_id, mesh_path, text_prompt)."""
    import csv
    samples = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            obj_id = row['obj_id']
            mesh_path = row['mesh']
            text = row.get(caption_field, row.get('caption_short', ''))
            if not text or not mesh_path:
                continue
            samples.append((obj_id, mesh_path, text))
            if 0 < max_samples <= len(samples):
                break
    return samples


def main():
    args = parse_args()

    if args.tsv_path is not None:
        # ── Batch mode ──
        print("=" * 60)
        print("  Attention Map Visualization (Batch Mode)")
        print("=" * 60)
        print(f"  TSV:     {args.tsv_path}")
        print(f"  Caption: {args.caption_field}")
        print(f"  Max:     {args.max_samples}")
        print()

        samples = load_tsv(args.tsv_path, args.caption_field, args.max_samples)
        print(f"  Loaded {len(samples)} samples from TSV")
        print()

        all_summaries = []
        for i, (obj_id, mesh_path, text_prompt) in enumerate(samples):
            print(f"\n{'─' * 60}")
            print(f"  [{i+1}/{len(samples)}] {obj_id}")
            print(f"  Mesh: {mesh_path}")
            print(f"  Text: {text_prompt[:80]}...")
            print(f"{'─' * 60}")

            if not os.path.isfile(mesh_path):
                print(f"  WARNING: Mesh not found, skipping: {mesh_path}")
                continue

            sample_out = os.path.join(args.output_dir, obj_id)
            try:
                summary = process_single(args, mesh_path, text_prompt, sample_out)
                summary['obj_id'] = obj_id
                all_summaries.append(summary)
            except Exception as e:
                print(f"  ERROR processing {obj_id}: {e}")
                import traceback; traceback.print_exc()
                continue

        # Save batch summary
        batch_summary_path = os.path.join(args.output_dir, 'batch_summary.json')
        with open(batch_summary_path, 'w') as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)

        print()
        print("=" * 60)
        print(f"  Batch complete: {len(all_summaries)}/{len(samples)} succeeded")
        print(f"  Results: {args.output_dir}")
        print(f"  Summary: {batch_summary_path}")
        print("=" * 60)

    else:
        # ── Single mode ──
        print("=" * 60)
        print("  Attention Map Visualization")
        print("=" * 60)
        print(f"  Text: {args.text_prompt[:80]}...")
        print(f"  Mesh: {args.mesh_path}")
        print()

        summary = process_single(args, args.mesh_path, args.text_prompt, args.output_dir)

        print()
        print("=" * 60)
        print(f"  All figures saved to: {args.output_dir}")
        print(f"  Generated 7 figures (fig1–fig7)")
        print("=" * 60)
        print()
        print("  Top-5 attended words:")
        for w, v in summary['top_attended_words'][:5]:
            print(f"    {w:20s}  {v:.6f}")


if __name__ == '__main__':
    main()
