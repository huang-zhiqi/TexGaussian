"""
FID 诊断脚本：分解 v10 vs baseline 的 FID 差异根源

分析维度：
1. FID mean/covariance 分解 —— 是均值偏移还是方差差异
2. 逐样本 Inception 距离 —— 找到拖后腿的样本
3. 图像统计特征对比 —— 亮度、饱和度、对比度分布差异
4. Per-sample CLIP scores —— 哪些样本 v10 更好/更差
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from scipy import linalg
import json
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ======================== Config ========================
V10_ROOT = Path('../experiments/texverse_stage1_new_modules_v10/texverse_gen_renders')
BASE_ROOT = Path('../experiments/texgaussian/texverse_gen_renders')
GT_ROOT = Path('../datasets/texverse_rendered_test')
ENVMAPS = ['shanghai_bund_2k', 'studio_small_09_2k', 'sunflowers_puresky_2k']
VIEWS_PER_SAMPLE = 16  # 16 views per envmap


# ======================== Inception Feature Extractor ========================
class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval()
        # Remove final FC to get 2048-dim pool features
        self.blocks = torch.nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
            inception.maxpool1,
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            inception.maxpool2,
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            inception.avgpool,
        )
    
    def forward(self, x):
        # x: [B, 3, 299, 299] in [0, 1]
        # Inception expects specific normalization
        x = x * 2 - 1  # [0,1] -> [-1,1]
        return self.blocks(x).squeeze(-1).squeeze(-1)  # [B, 2048]


def collect_lit_paths(root, obj_ids, envmaps=ENVMAPS, max_views=VIEWS_PER_SAMPLE):
    """Collect lit image paths grouped by obj_id."""
    paths_by_obj = {}
    for obj_id in obj_ids:
        paths = []
        for env in envmaps:
            env_dir = root / obj_id / 'lit' / env
            if env_dir.exists():
                imgs = sorted(env_dir.glob('*_lit.png'))[:max_views]
                paths.extend(imgs)
        if paths:
            paths_by_obj[obj_id] = paths
    return paths_by_obj


def load_images_batch(paths, size=299):
    """Load images -> [B, 3, H, W] float32 [0,1]"""
    transform = transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
    ])
    imgs = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        imgs.append(transform(img))
    return torch.stack(imgs)


def extract_features(model, paths, batch_size=32):
    """Extract Inception features for all images."""
    all_feats = []
    total = (len(paths) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(paths), batch_size)):
        if bi % 50 == 0:
            print(f"    batch {bi}/{total}...", flush=True)
        batch_paths = paths[i:i+batch_size]
        imgs = load_images_batch(batch_paths).to(device)
        with torch.no_grad():
            feats = model(imgs)
        all_feats.append(feats.cpu())
        del imgs
    return torch.cat(all_feats, dim=0).numpy()


def compute_fid_components(feats_gen, feats_gt):
    """Compute FID and decompose into mean-shift and covariance components.
    Uses float64 for numerical stability in sqrtm.
    """
    feats_gen = feats_gen.astype(np.float64)
    feats_gt = feats_gt.astype(np.float64)
    
    mu_gen = np.mean(feats_gen, axis=0)
    mu_gt = np.mean(feats_gt, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    sigma_gt = np.cov(feats_gt, rowvar=False)
    
    # Mean component: ||mu_gen - mu_gt||^2
    mean_diff = np.sum((mu_gen - mu_gt) ** 2)
    
    # Covariance component using eigenvalue decomposition (much faster than sqrtm)
    # Tr(S1 + S2 - 2*(S1 @ S2)^0.5)
    # For symmetric PSD S1, S2: eigenvalues of S1 @ S2 are real and non-negative
    # Use SVD for numerical stability: sqrt(S1 @ S2) via sqrt of eigenvalues
    product = sigma_gen @ sigma_gt
    # Use eig (not eigvalsh since product is not symmetric)
    eigvals = np.linalg.eigvals(product).real
    eigvals = np.maximum(eigvals, 0)
    cov_diff = np.trace(sigma_gen) + np.trace(sigma_gt) - 2 * np.sum(np.sqrt(eigvals))
    
    fid = mean_diff + cov_diff
    return fid, mean_diff, cov_diff, mu_gen, mu_gt, sigma_gen, sigma_gt


def compute_per_sample_inception_distance(model, gen_paths_by_obj, gt_paths_by_obj, obj_ids):
    """Compute per-sample mean Inception feature distance."""
    distances = {}
    for obj_id in obj_ids:
        if obj_id not in gen_paths_by_obj or obj_id not in gt_paths_by_obj:
            continue
        gen_feats = extract_features(model, gen_paths_by_obj[obj_id])
        gt_feats = extract_features(model, gt_paths_by_obj[obj_id])
        # L2 distance between mean features
        d = np.linalg.norm(gen_feats.mean(0) - gt_feats.mean(0))
        distances[obj_id] = d
    return distances


def compute_image_stats(paths_by_obj, obj_ids, max_per_obj=3):
    """Compute basic image statistics: brightness, saturation, contrast."""
    stats = {}
    for obj_id in obj_ids:
        if obj_id not in paths_by_obj:
            continue
        brightnesses = []
        saturations = []
        contrasts = []
        for p in paths_by_obj[obj_id][:max_per_obj]:
            img = np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
            # Brightness (mean luminance)
            lum = 0.299 * img[...,0] + 0.587 * img[...,1] + 0.114 * img[...,2]
            brightnesses.append(lum.mean())
            # Contrast (std of luminance)
            contrasts.append(lum.std())
            # Saturation
            maxc = img.max(axis=-1)
            minc = img.min(axis=-1)
            sat = np.where(maxc > 0, (maxc - minc) / (maxc + 1e-8), 0)
            saturations.append(sat.mean())
        stats[obj_id] = {
            'brightness': np.mean(brightnesses),
            'contrast': np.mean(contrasts),
            'saturation': np.mean(saturations),
        }
    return stats


def main():
    print("=" * 60)
    print("FID 诊断: v10 vs Baseline (200 samples)")
    print("=" * 60)
    
    # Collect object IDs
    v10_ids = set(p.name for p in V10_ROOT.iterdir() if p.is_dir())
    base_ids = set(p.name for p in BASE_ROOT.iterdir() if p.is_dir())
    gt_ids = set(p.name for p in GT_ROOT.iterdir() if p.is_dir())
    common_ids = sorted(v10_ids & base_ids & gt_ids)
    print(f"\nCommon objects: {len(common_ids)}")
    
    # Collect paths
    print("\nCollecting image paths...")
    v10_paths = collect_lit_paths(V10_ROOT, common_ids)
    base_paths = collect_lit_paths(BASE_ROOT, common_ids)
    gt_paths = collect_lit_paths(GT_ROOT, common_ids)
    
    v10_all = [p for obj in common_ids for p in v10_paths.get(obj, [])]
    base_all = [p for obj in common_ids for p in base_paths.get(obj, [])]
    gt_all = [p for obj in common_ids for p in gt_paths.get(obj, [])]
    print(f"Total images: v10={len(v10_all)}, base={len(base_all)}, gt={len(gt_all)}")
    
    # ============ Part 1: FID Decomposition ============
    print("\n" + "=" * 60)
    print("Part 1: FID Mean/Covariance 分解")
    print("=" * 60)
    
    print("Loading InceptionV3...")
    model = InceptionFeatureExtractor().to(device).eval()
    
    print("Extracting features (this may take a few minutes)...")
    print("  GT features...")
    gt_feats = extract_features(model, gt_all)
    print(f"  GT: {gt_feats.shape}")
    
    print("  v10 features...")
    v10_feats = extract_features(model, v10_all)
    print(f"  v10: {v10_feats.shape}")
    
    print("  Baseline features...")
    base_feats = extract_features(model, base_all)
    print(f"  Base: {base_feats.shape}")
    
    # FID decomposition
    fid_v10, mean_v10, cov_v10, mu_v10, mu_gt, _, _ = compute_fid_components(v10_feats, gt_feats)
    fid_base, mean_base, cov_base, mu_base, _, _, _ = compute_fid_components(base_feats, gt_feats)
    
    print(f"\n{'':>20s} {'v10':>12s} {'Baseline':>12s} {'Diff':>12s}")
    print(f"{'FID (total)':>20s} {fid_v10:>12.2f} {fid_base:>12.2f} {fid_v10-fid_base:>+12.2f}")
    print(f"{'  Mean shift':>20s} {mean_v10:>12.2f} {mean_base:>12.2f} {mean_v10-mean_base:>+12.2f}")
    print(f"{'  Cov diff':>20s} {cov_v10:>12.2f} {cov_base:>12.2f} {cov_v10-cov_base:>+12.2f}")
    print(f"{'  Mean%':>20s} {100*mean_v10/fid_v10:>11.1f}% {100*mean_base/fid_base:>11.1f}%")
    
    # ============ Part 2: Per-sample Inception distance ============
    print("\n" + "=" * 60)
    print("Part 2: 逐样本 Inception feature 距离")
    print("=" * 60)
    
    # Compute per-obj mean feature for each method
    v10_per_obj_feats = {}
    base_per_obj_feats = {}
    gt_per_obj_feats = {}
    
    idx = 0
    for obj_id in common_ids:
        n_v10 = len(v10_paths.get(obj_id, []))
        n_base = len(base_paths.get(obj_id, []))
        n_gt = len(gt_paths.get(obj_id, []))
        # We need to track indices correctly
        pass
    
    # Simpler: compute per-object distances using pre-extracted features  
    # Re-index features by object
    v10_idx = 0
    base_idx = 0
    gt_idx = 0
    v10_dists = {}
    base_dists = {}
    
    for obj_id in common_ids:
        n_v10 = len(v10_paths.get(obj_id, []))
        n_base = len(base_paths.get(obj_id, []))
        n_gt = len(gt_paths.get(obj_id, []))
        
        v10_obj = v10_feats[v10_idx:v10_idx+n_v10].mean(0)
        base_obj = base_feats[base_idx:base_idx+n_base].mean(0)
        gt_obj = gt_feats[gt_idx:gt_idx+n_gt].mean(0)
        
        v10_dists[obj_id] = np.linalg.norm(v10_obj - gt_obj)
        base_dists[obj_id] = np.linalg.norm(base_obj - gt_obj)
        
        v10_idx += n_v10
        base_idx += n_base
        gt_idx += n_gt
    
    # Sort by v10-baseline difference (worst first)
    diff_dists = {obj: v10_dists[obj] - base_dists[obj] for obj in common_ids}
    sorted_by_diff = sorted(diff_dists.items(), key=lambda x: -x[1])
    
    print(f"\nv10 比 baseline 差距最大的 10 个样本 (Inception distance):")
    print(f"{'Obj ID':>36s} {'v10_dist':>10s} {'base_dist':>10s} {'diff':>10s}")
    for obj_id, diff in sorted_by_diff[:10]:
        print(f"{obj_id:>36s} {v10_dists[obj_id]:>10.2f} {base_dists[obj_id]:>10.2f} {diff:>+10.2f}")
    
    print(f"\nv10 比 baseline 好得最多的 10 个样本:")
    for obj_id, diff in sorted_by_diff[-10:][::-1]:
        print(f"{obj_id:>36s} {v10_dists[obj_id]:>10.2f} {base_dists[obj_id]:>10.2f} {diff:>+10.2f}")
    
    # Statistics
    v10_better = sum(1 for d in diff_dists.values() if d < 0)
    base_better = sum(1 for d in diff_dists.values() if d > 0)
    print(f"\n样本胜负统计:")
    print(f"  v10 更好: {v10_better}/{len(common_ids)} ({100*v10_better/len(common_ids):.1f}%)")
    print(f"  Baseline 更好: {base_better}/{len(common_ids)} ({100*base_better/len(common_ids):.1f}%)")
    print(f"  v10 平均距离: {np.mean(list(v10_dists.values())):.3f}")
    print(f"  Base 平均距离: {np.mean(list(base_dists.values())):.3f}")
    
    # ============ Part 3: Image Statistics ============
    print("\n" + "=" * 60)
    print("Part 3: 图像统计特征对比 (亮度/对比度/饱和度)")
    print("=" * 60)
    
    print("Computing image stats...")
    v10_stats = compute_image_stats(v10_paths, common_ids)
    base_stats = compute_image_stats(base_paths, common_ids)
    gt_stats = compute_image_stats(gt_paths, common_ids)
    
    for metric in ['brightness', 'contrast', 'saturation']:
        v10_vals = [v10_stats[o][metric] for o in common_ids if o in v10_stats]
        base_vals = [base_stats[o][metric] for o in common_ids if o in base_stats]
        gt_vals = [gt_stats[o][metric] for o in common_ids if o in gt_stats]
        
        print(f"\n  {metric}:")
        print(f"    GT:       mean={np.mean(gt_vals):.4f}  std={np.std(gt_vals):.4f}")
        print(f"    v10:      mean={np.mean(v10_vals):.4f}  std={np.std(v10_vals):.4f}  delta_from_gt={np.mean(v10_vals)-np.mean(gt_vals):+.4f}")
        print(f"    Baseline: mean={np.mean(base_vals):.4f}  std={np.std(base_vals):.4f}  delta_from_gt={np.mean(base_vals)-np.mean(gt_vals):+.4f}")
    
    # Per-sample brightness diff from GT
    v10_bright_diffs = [abs(v10_stats[o]['brightness'] - gt_stats[o]['brightness']) for o in common_ids if o in v10_stats and o in gt_stats]
    base_bright_diffs = [abs(base_stats[o]['brightness'] - gt_stats[o]['brightness']) for o in common_ids if o in base_stats and o in gt_stats]
    print(f"\n  Mean absolute brightness delta from GT:")
    print(f"    v10:      {np.mean(v10_bright_diffs):.4f}")
    print(f"    Baseline: {np.mean(base_bright_diffs):.4f}")
    
    # ============ Part 4: Inception feature distribution analysis ============
    print("\n" + "=" * 60)
    print("Part 4: Inception 特征空间分析")
    print("=" * 60)
    
    # Feature magnitude statistics
    v10_norms = np.linalg.norm(v10_feats, axis=1)
    base_norms = np.linalg.norm(base_feats, axis=1)
    gt_norms = np.linalg.norm(gt_feats, axis=1)
    
    print(f"\n  Feature L2 norms:")
    print(f"    GT:       mean={gt_norms.mean():.2f}  std={gt_norms.std():.2f}")
    print(f"    v10:      mean={v10_norms.mean():.2f}  std={v10_norms.std():.2f}")
    print(f"    Baseline: mean={base_norms.mean():.2f}  std={base_norms.std():.2f}")
    
    # Top-K principal component analysis
    from sklearn.decomposition import PCA
    pca = PCA(n_components=20)
    pca.fit(gt_feats)
    
    gt_proj = pca.transform(gt_feats)
    v10_proj = pca.transform(v10_feats)
    base_proj = pca.transform(base_feats)
    
    print(f"\n  Top-20 PCA component mean differences (from GT):")
    print(f"  {'PC':>4s} {'v10_delta':>12s} {'base_delta':>12s} {'v10_worse?':>12s}")
    for i in range(20):
        v10_d = abs(v10_proj[:, i].mean() - gt_proj[:, i].mean())
        base_d = abs(base_proj[:, i].mean() - gt_proj[:, i].mean())
        worse = "<<< v10" if v10_d > base_d * 1.5 else ("base" if base_d > v10_d * 1.5 else "~")
        print(f"  {i:>4d} {v10_d:>12.3f} {base_d:>12.3f} {worse:>12s}")
    
    # Variance comparison in PC space
    print(f"\n  Top-10 PCA component variance ratio (gen/gt):")
    print(f"  {'PC':>4s} {'VarExpl%':>10s} {'v10_ratio':>12s} {'base_ratio':>12s}")
    for i in range(10):
        gt_var = gt_proj[:, i].var()
        v10_var = v10_proj[:, i].var()
        base_var = base_proj[:, i].var()
        print(f"  {i:>4d} {100*pca.explained_variance_ratio_[i]:>9.1f}% {v10_var/gt_var:>12.3f} {base_var/gt_var:>12.3f}")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
