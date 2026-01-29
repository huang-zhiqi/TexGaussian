#!/usr/bin/env python3
"""
Compute evaluation metrics between ground-truth (GT) renders and generated (Gen) renders.

Metrics:
- Masked (unlit, per-channel): Albedo PSNR/SSIM (sRGB), Roughness/Metallic L1, Normal MeanAngularError
- Distribution (lit only): FID, KID
- Semantic (lit only): CLIP image similarity, CLIP text similarity (caption_short), LongCLIP text similarity (caption_long)
- Optional: LPIPS on masked albedo when enabled

Multi-View Consistency Metrics (CVPR 2025 / ECCV 2024-2025 standard):
- CrossView_LPIPS: Perceptual consistency between adjacent views (detects Janus problems)
- CrossView_L1: Pixel-level consistency between adjacent views
- Normal_Consistency: Angular consistency of surface normals across views
- Normal_Distribution_Div: Distribution divergence of normals (detects multi-face artifacts)
- Reproj_L1 / Reproj_LPIPS: Reprojection error when depth maps are available (MEt3R/MVGBench style)

The consistency metrics are critical for detecting:
1. Janus (multi-face) problems common in 3D generation
2. Texture flickering across viewpoints
3. Geometric inconsistencies

Usage:
    python eval_metrics.py --experiment_name my_exp --metrics all
    python eval_metrics.py --experiment_name my_exp --metrics consistency --consistency_channel albedo

The script reads images directly from their source directories without copying them
elsewhere. GT and Gen filenames are assumed to align for each channel.
If lit renders are stored under per-HDRI subfolders (lit/<hdri_name>), the script
computes lit metrics per HDRI and reports their mean under HDRI/Mean/*.
"""

import argparse
import csv
import gc
import glob
import json
import math
import os
import sys
import traceback
import warnings
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Set environment variables BEFORE importing torch to prevent CUDA issues
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
# Disable tokenizers parallelism to prevent deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Lazy imports for heavy modules to improve stability
clip = None
lpips = None
torchmetrics = None


def _lazy_import_clip():
    """Lazy import clip to avoid import-time crashes."""
    global clip
    if clip is None:
        import clip as _clip
        clip = _clip
    return clip


def _lazy_import_lpips():
    """Lazy import lpips to avoid import-time crashes."""
    global lpips
    if lpips is None:
        import lpips as _lpips
        lpips = _lpips
    return lpips


def _lazy_import_torchmetrics():
    """Lazy import torchmetrics to avoid import-time crashes."""
    global torchmetrics
    if torchmetrics is None:
        import torchmetrics as _torchmetrics
        torchmetrics = _torchmetrics
    return torchmetrics

# Silence torchvision deprecated pretrained warnings triggered inside lpips
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Arguments other than a weight enum or `None` for 'weights' are deprecated",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LONGCLIP_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "third_party", "Long-CLIP"))
DEFAULT_LONGCLIP_MODEL = os.path.join(DEFAULT_LONGCLIP_ROOT, "checkpoints", "longclip-L.pt")
DEFAULT_LONGCLIP_CONTEXT_LENGTH = 248


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated renders against GT renders.")
    parser.add_argument(
        "--experiment_name",
        required=True,
        help="Name of the experiment; used to resolve the generated render directory.",
    )
    parser.add_argument(
        "--base_gt_dir",
        default="../datasets/texverse_rendered",
        help="Root directory containing GT renders, organized by object id.",
    )
    parser.add_argument(
        "--base_gen_dir",
        default=None,
        help="Root directory containing generated renders. Defaults to "
        "'../experiments/{experiment_name}/texverse_gen_renders'.",
    )
    parser.add_argument(
        "--lit_subdir",
        default="lit",
        help="Subdirectory under each obj_id that stores lit images.",
    )
    parser.add_argument(
        "--unlit_subdir",
        default="unlit",
        help="Subdirectory under each obj_id that stores unlit images.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for metric computation.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Computation device (e.g., 'cuda', 'cuda:1', or 'cpu'). Falls back to CPU if unavailable.",
    )
    parser.add_argument("--kid_subset_size", type=int, default=50, help="Subset size for KID computation.")
    parser.add_argument("--clip_model", default="ViT-B/32", help="CLIP model variant to load.")
    parser.add_argument(
        "--longclip_model",
        default=DEFAULT_LONGCLIP_MODEL,
        help="Path to LongCLIP checkpoint (.pt) used with caption_long text similarity.",
    )
    parser.add_argument(
        "--longclip_root",
        default=DEFAULT_LONGCLIP_ROOT,
        help="Path to Long-CLIP repo for importing longclip when not installed.",
    )
    parser.add_argument(
        "--longclip_context_length",
        type=int,
        default=DEFAULT_LONGCLIP_CONTEXT_LENGTH,
        help="Context length for LongCLIP tokenization.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug prints and diff map dumps for masked metrics.",
    )
    parser.add_argument(
        "--prompts_file",
        default=None,
        help=(
            "Optional JSON mapping (obj_id -> {caption_short, caption_long}) or TSV manifest path "
            "(expects obj_id + caption_short + caption_long) for CLIP/LongCLIP text-image similarity."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for metrics (JSON or CSV). Defaults to 'metrics_{experiment_name}.json' in CWD.",
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help=(
            "Which metrics to compute. Presets: all | pixel (psnr,ssim,lpips) | "
            "dist (fid,kid) | semantic (clip) | consistency (multi-view). "
            "Pixel metrics are computed per unlit channel. "
            "CLIP text similarity uses caption_short and LongCLIP uses caption_long when --prompts_file is provided. "
            "Multi-view consistency metrics detect Janus/flickering issues. "
            "You can also pass a comma list, e.g., 'psnr,ssim,clip,consistency'."
        ),
    )
    parser.add_argument(
        "--consistency_pairs",
        type=int,
        default=5,
        help="Number of random adjacent view pairs to sample per object for consistency metrics.",
    )
    parser.add_argument(
        "--consistency_channel",
        default="albedo",
        choices=["albedo", "lit", "normal"],
        help="Which channel to use for cross-view consistency: albedo, lit (beauty), or normal.",
    )
    parser.add_argument(
        "--reprojection",
        action="store_true",
        help="Also compute depth-based reprojection metrics (requires depth maps and dense views).",
    )
    return parser.parse_args()


def resolve_gen_dir(args: argparse.Namespace) -> str:
    if args.base_gen_dir:
        return args.base_gen_dir
    return os.path.join("../experiments", args.experiment_name, "texverse_gen_renders")


def list_object_ids(base_gen_dir: str) -> List[str]:
    obj_ids = [
        d
        for d in os.listdir(base_gen_dir)
        if os.path.isdir(os.path.join(base_gen_dir, d))
    ]
    return sorted(obj_ids)

UNLIT_CHANNELS = ["albedo", "rough", "metal", "normal"]
CHANNEL_LABELS = {
    "albedo": "Albedo",
    "rough": "Roughness",
    "metal": "Metallic",
    "normal": "Normal",
}


def collect_lit_paths(
    base_gt_dir: str,
    base_gen_dir: str,
    lit_subdir: str,
    obj_ids: List[str],
) -> Tuple[List[str], List[str]]:
    all_gt_paths: List[str] = []
    all_gen_paths: List[str] = []

    for obj_id in obj_ids:
        gt_dir = os.path.join(base_gt_dir, obj_id, lit_subdir)
        gen_dir = os.path.join(base_gen_dir, obj_id, lit_subdir)
        gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*_lit.png")))
        gen_paths = sorted(glob.glob(os.path.join(gen_dir, "*_lit.png")))

        if not gen_paths:
            raise FileNotFoundError(f"No generated lit images found for {obj_id} in {gen_dir}")
        if not gt_paths:
            raise FileNotFoundError(f"No GT lit images found for {obj_id} in {gt_dir}")
        if len(gen_paths) != len(gt_paths):
            raise ValueError(f"Lit image count mismatch for {obj_id}: {len(gen_paths)} gen vs {len(gt_paths)} gt")

        gt_names = [os.path.basename(p) for p in gt_paths]
        gen_names = [os.path.basename(p) for p in gen_paths]
        if gt_names != gen_names:
            raise ValueError(f"Lit filename mismatch for {obj_id}")

        all_gt_paths.extend(gt_paths)
        all_gen_paths.extend(gen_paths)

    return all_gt_paths, all_gen_paths


def list_subdirs(path: str) -> Set[str]:
    if not os.path.isdir(path):
        return set()
    return {name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))}


def detect_lit_hdris(
    base_gt_dir: str,
    base_gen_dir: str,
    lit_subdir: str,
    obj_ids: List[str],
) -> List[str]:
    candidates: Optional[Set[str]] = None
    union: Set[str] = set()
    had_subdirs = False
    objs_without_subdirs: List[str] = []

    for obj_id in obj_ids:
        gt_dir = os.path.join(base_gt_dir, obj_id, lit_subdir)
        gen_dir = os.path.join(base_gen_dir, obj_id, lit_subdir)
        gt_subdirs = list_subdirs(gt_dir)
        gen_subdirs = list_subdirs(gen_dir)

        if gt_subdirs or gen_subdirs:
            had_subdirs = True
            if not gt_subdirs or not gen_subdirs:
                raise FileNotFoundError(
                    f"Lit HDRI subdir mismatch for {obj_id}: gt={sorted(gt_subdirs)} gen={sorted(gen_subdirs)}"
                )
            obj_subdirs = gt_subdirs & gen_subdirs
            if not obj_subdirs:
                raise FileNotFoundError(f"No common lit HDRI subdirs for {obj_id}")
            union |= obj_subdirs
            candidates = obj_subdirs if candidates is None else candidates & obj_subdirs
        else:
            objs_without_subdirs.append(obj_id)

    if not had_subdirs:
        return []
    if objs_without_subdirs:
        preview = ", ".join(objs_without_subdirs[:5])
        raise FileNotFoundError(
            f"Some objects lack lit HDRI subdirs under '{lit_subdir}' (e.g., {preview})"
        )
    if not candidates:
        raise FileNotFoundError("No common lit HDRI subdirs across objects.")
    if union != candidates:
        missing = sorted(union - candidates)
        if missing:
            print(f"[WARN] HDRI subdirs not shared by all objects; ignoring: {', '.join(missing)}")
    return sorted(candidates)


def collect_unlit_channel_paths(
    base_gt_dir: str,
    base_gen_dir: str,
    unlit_subdir: str,
    channel: str,
    obj_ids: List[str],
) -> Tuple[List[str], List[str]]:
    all_gt_paths: List[str] = []
    all_gen_paths: List[str] = []
    pattern = f"*_{channel}.png"

    for obj_id in obj_ids:
        gt_dir = os.path.join(base_gt_dir, obj_id, unlit_subdir)
        gen_dir = os.path.join(base_gen_dir, obj_id, unlit_subdir)
        gt_paths = sorted(glob.glob(os.path.join(gt_dir, pattern)))
        gen_paths = sorted(glob.glob(os.path.join(gen_dir, pattern)))

        if not gen_paths:
            raise FileNotFoundError(f"No generated {channel} images found for {obj_id} in {gen_dir}")
        if not gt_paths:
            raise FileNotFoundError(f"No GT {channel} images found for {obj_id} in {gt_dir}")
        if len(gen_paths) != len(gt_paths):
            raise ValueError(
                f"{channel} image count mismatch for {obj_id}: {len(gen_paths)} gen vs {len(gt_paths)} gt"
            )

        gt_names = [os.path.basename(p) for p in gt_paths]
        gen_names = [os.path.basename(p) for p in gen_paths]
        if gt_names != gen_names:
            raise ValueError(f"{channel} filename mismatch for {obj_id}")

        all_gt_paths.extend(gt_paths)
        all_gen_paths.extend(gen_paths)

    return all_gt_paths, all_gen_paths


def compute_lit_metrics_for_paths(
    lit_gt_paths: List[str],
    lit_gen_paths: List[str],
    base_gen_dir: str,
    batch_size: int,
    device: torch.device,
    metric_flags: Dict[str, bool],
    clip_model: Optional[torch.nn.Module],
    clip_preprocess: Optional[object],
    longclip_model: Optional[torch.nn.Module],
    longclip_preprocess: Optional[object],
    longclip_module: Optional[object],
    clip_text_feature_map: Optional[Dict[str, torch.Tensor]],
    longclip_text_feature_map: Optional[Dict[str, torch.Tensor]],
    do_clip_image: bool,
    do_clip_text: bool,
    do_longclip_text: bool,
    kid_subset_size_default: int,
) -> Dict[str, float]:
    lit_metrics: Dict[str, float] = {}

    kid_subset_size = None
    if metric_flags["kid"]:
        kid_subset_size = min(kid_subset_size_default, len(lit_gt_paths))
        if kid_subset_size < kid_subset_size_default:
            print(f"KID subset_size clipped to {kid_subset_size} due to limited samples.")

    fid_metric = None
    kid_metric = None
    if metric_flags["fid"] or metric_flags["kid"]:
        torchmetrics_mod = _lazy_import_torchmetrics()
        if metric_flags["fid"]:
            fid_metric = torchmetrics_mod.image.fid.FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        if metric_flags["kid"]:
            kid_metric = torchmetrics_mod.image.kid.KernelInceptionDistance(subset_size=kid_subset_size, normalize=False).to(device)
        _clear_cuda_cache()

    clip_image_scores: List[float] = []
    clip_text_scores: List[float] = []
    longclip_text_scores: List[float] = []
    total_batches = (len(lit_gen_paths) + batch_size - 1) // batch_size
    print(f"Processing {total_batches} batches for lit metrics...")

    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, len(lit_gen_paths), batch_size)):
            batch_gen_paths = lit_gen_paths[start : start + batch_size]
            batch_gt_paths = lit_gt_paths[start : start + batch_size]

            try:
                _, gen_uint8, gen_clip = load_batch(
                    batch_gen_paths,
                    clip_preprocess,
                    device,
                    include_clip=do_clip_image or do_clip_text,
                )
                _, gt_uint8, gt_clip = load_batch(
                    batch_gt_paths,
                    clip_preprocess,
                    device,
                    include_clip=do_clip_image,
                )

                gen_feat = None
                if (do_clip_image or do_clip_text) and clip_model and gen_clip is not None:
                    gen_feat = clip_model.encode_image(gen_clip)
                    gen_feat = F.normalize(gen_feat, dim=-1)

                if do_clip_image and clip_model and gen_feat is not None and gt_clip is not None:
                    gt_feat = clip_model.encode_image(gt_clip)
                    gt_feat = F.normalize(gt_feat, dim=-1)
                    sim = (gen_feat * gt_feat).sum(dim=-1)
                    clip_image_scores.append(sim.mean().item())
                    del gt_feat

                batch_obj_ids = None
                if do_clip_text or do_longclip_text:
                    batch_obj_ids = [obj_id_from_path(path, base_gen_dir) for path in batch_gen_paths]

                if do_clip_text and gen_feat is not None and clip_text_feature_map is not None:
                    text_feat = torch.stack([clip_text_feature_map[obj_id] for obj_id in batch_obj_ids], dim=0)
                    sim = (gen_feat * text_feat).sum(dim=-1)
                    clip_text_scores.append(sim.mean().item())
                    del text_feat

                if gen_clip is not None:
                    del gen_clip
                if gt_clip is not None:
                    del gt_clip
                if gen_feat is not None:
                    del gen_feat
                gen_clip = gt_clip = gen_feat = None
                if batch_idx % 5 == 0:
                    _clear_cuda_cache()

                longclip_gen_feat = None
                if do_longclip_text and longclip_model and longclip_preprocess is not None:
                    _, _, gen_longclip = load_batch(
                        batch_gen_paths,
                        longclip_preprocess,
                        device,
                        include_clip=True,
                    )
                    if gen_longclip is not None:
                        longclip_gen_feat = longclip_model.encode_image(gen_longclip)
                        longclip_gen_feat = F.normalize(longclip_gen_feat, dim=-1)
                        del gen_longclip

                if do_longclip_text and longclip_gen_feat is not None and longclip_text_feature_map is not None:
                    if batch_obj_ids is None:
                        batch_obj_ids = [obj_id_from_path(path, base_gen_dir) for path in batch_gen_paths]
                    long_text_feat = torch.stack(
                        [longclip_text_feature_map[obj_id] for obj_id in batch_obj_ids],
                        dim=0,
                    )
                    sim = (longclip_gen_feat * long_text_feat).sum(dim=-1)
                    longclip_text_scores.append(sim.mean().item())
                    del long_text_feat

                if longclip_gen_feat is not None:
                    del longclip_gen_feat
                longclip_gen_feat = None

                if fid_metric:
                    fid_metric.update(gt_uint8, real=True)
                    fid_metric.update(gen_uint8, real=False)
                if kid_metric:
                    kid_metric.update(gt_uint8, real=True)
                    kid_metric.update(gen_uint8, real=False)

                del gen_uint8, gt_uint8

            except Exception as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "cuda" in error_msg or "memory" in error_msg:
                    print(f"[WARN] CUDA/Memory error at batch {batch_idx}/{total_batches}: {e}")
                    print("[WARN] Clearing cache and skipping this batch...")
                    _clear_cuda_cache()
                    continue
                print(f"[ERROR] Error at batch {batch_idx}: {e}")
                traceback.print_exc()
                raise

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{total_batches} batches")
                _clear_cuda_cache()

    if do_clip_image:
        lit_metrics["CLIP_Image_Score"] = (
            float(np.mean(clip_image_scores)) if clip_image_scores else float("nan")
        )
    if do_clip_text:
        lit_metrics["CLIP_Text_Score"] = (
            float(np.mean(clip_text_scores)) if clip_text_scores else float("nan")
        )
    if do_longclip_text:
        lit_metrics["LongCLIP_Text_Score"] = (
            float(np.mean(longclip_text_scores)) if longclip_text_scores else float("nan")
        )
    if fid_metric:
        lit_metrics["FID"] = fid_metric.compute().item()
        del fid_metric
    if kid_metric:
        try:
            lit_metrics["KID"] = kid_metric.compute()[0].item()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: KID computation failed ({exc}); skipping KID.")
            lit_metrics["KID"] = float("nan")
        del kid_metric
    _clear_cuda_cache()

    return lit_metrics


def compute_mean_metrics(metrics_by_group: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys: Set[str] = set()
    for metrics in metrics_by_group.values():
        keys.update(metrics.keys())

    mean_metrics: Dict[str, float] = {}
    for key in sorted(keys):
        values = [
            val
            for metrics in metrics_by_group.values()
            if key in metrics
            for val in [metrics[key]]
            if isinstance(val, (int, float)) and math.isfinite(val)
        ]
        mean_metrics[key] = float(np.mean(values)) if values else float("nan")
    return mean_metrics


def parse_metrics_arg(metrics_arg: str) -> Dict[str, bool]:
    metrics_arg = metrics_arg.lower().replace(" ", "")
    presets = {
        "all": {"psnr", "ssim", "lpips", "fid", "kid", "clip", "consistency"},
        "pixel": {"psnr", "ssim", "lpips"},
        "structural": {"psnr", "ssim", "lpips"},
        "dist": {"fid", "kid"},
        "distribution": {"fid", "kid"},
        "semantic": {"clip"},
        "consistency": {"consistency"},
        "geometric": {"consistency"},
    }
    supported: Set[str] = {"psnr", "ssim", "lpips", "fid", "kid", "clip", "consistency"}
    if metrics_arg in presets:
        selected = presets[metrics_arg]
    else:
        selected = {m for m in metrics_arg.split(",") if m}
        unknown = selected - supported
        if unknown:
            raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")
    if not selected:
        raise ValueError("No metrics selected.")
    return {name: (name in selected) for name in supported}


def resolve_longclip_module(longclip_root: Optional[str]) -> Tuple[object, str]:
    last_exc = None
    try:
        import longclip as longclip_module
        return longclip_module, "longclip"
    except Exception as exc:
        last_exc = exc

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
            from model import longclip as longclip_module
            return longclip_module, f"model.longclip ({root})"
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        "longclip is not available; install longclip or set --longclip_root "
        f"(default: {DEFAULT_LONGCLIP_ROOT})"
    ) from last_exc


def load_longclip_model(longclip_model_path: str, device: torch.device, longclip_root: Optional[str]):
    longclip_module, import_source = resolve_longclip_module(longclip_root)
    if not longclip_model_path:
        raise ValueError("LongCLIP model path is empty; provide --longclip_model.")
    
    # Handle relative paths - resolve from script directory if not absolute
    if not os.path.isabs(longclip_model_path):
        # Try relative to current working directory first
        if not os.path.isfile(longclip_model_path):
            # Try relative to script directory
            script_relative = os.path.join(SCRIPT_DIR, "..", longclip_model_path.lstrip("../"))
            script_relative = os.path.abspath(script_relative)
            if os.path.isfile(script_relative):
                longclip_model_path = script_relative
            else:
                # Try default location
                default_path = os.path.join(DEFAULT_LONGCLIP_ROOT, "checkpoints", "longclip-L.pt")
                if os.path.isfile(default_path):
                    longclip_model_path = default_path
    
    if not os.path.isfile(longclip_model_path):
        raise FileNotFoundError(f"LongCLIP model not found: {longclip_model_path}")
    
    print(f"Loading LongCLIP from: {longclip_model_path}")
    # Load to CPU first, then move to device to avoid GPU memory fragmentation
    model, preprocess = longclip_module.load(longclip_model_path, device="cpu")
    model = model.to(device)
    model.eval()
    return model, preprocess, longclip_module, import_source


def load_prompt_pairs(prompts_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    ext = os.path.splitext(prompts_path)[1].lower()
    if ext in {".tsv", ".csv"}:
        with open(prompts_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if not reader.fieldnames:
                raise ValueError("Prompts TSV must include a header row.")
            required = {"obj_id", "caption_short", "caption_long"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Prompts TSV missing columns: {', '.join(sorted(missing))}")
            short_map: Dict[str, str] = {}
            long_map: Dict[str, str] = {}
            for row in reader:
                obj_id = (row.get("obj_id") or "").strip()
                if not obj_id:
                    continue
                caption_short = (row.get("caption_short") or "").strip()
                caption_long = (row.get("caption_long") or "").strip()
                if caption_short:
                    short_map[obj_id] = caption_short
                if caption_long:
                    long_map[obj_id] = caption_long
        return short_map, long_map

    with open(prompts_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Prompts file must be a JSON object mapping obj_id to prompt text.")

    short_map = {}
    long_map = {}
    fallback_used = False
    for obj_id, value in data.items():
        obj_id = str(obj_id).strip()
        if not obj_id:
            continue
        if isinstance(value, dict):
            short_val = value.get("caption_short", value.get("caption", ""))
            long_val = value.get("caption_long", value.get("caption", ""))
        else:
            short_val = value
            long_val = value
            fallback_used = True
        caption_short = str(short_val).strip() if short_val is not None else ""
        caption_long = str(long_val).strip() if long_val is not None else ""
        if caption_short:
            short_map[obj_id] = caption_short
        if caption_long:
            long_map[obj_id] = caption_long

    if fallback_used:
        print("[WARN] Prompts JSON uses obj_id -> string; using the same text for caption_short and caption_long.")

    return short_map, long_map


def obj_id_from_path(path: str, base_dir: str) -> str:
    rel = os.path.relpath(path, base_dir)
    return rel.split(os.sep)[0]


def extract_alpha_mask(gen_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> torch.Tensor:
    if gen_tensor.shape[1] < 4 or gt_tensor.shape[1] < 4:
        raise ValueError("Alpha channel required for masked metrics. Ensure keep_alpha=True when loading.")
    gen_alpha = gen_tensor[:, 3:4, ...]
    gt_alpha = gt_tensor[:, 3:4, ...]
    return (gen_alpha > 0.5) & (gt_alpha > 0.5)


MIN_MASK_PIXELS = 10


def compute_masked_metrics(
    gen_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    metric_type: str = "color",
    debug: bool = False,
    obj_ids: Optional[Sequence[str]] = None,
    debug_names: Optional[Sequence[str]] = None,
    debug_dir: Optional[str] = None,
    debug_tag: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Input: gen_tensor, gt_tensor (B, 4, H, W) in [0, 1].
    Logic:
    1. Extract alpha and compute mask intersection.
    2. Compute metrics only over masked pixels.
    3. If mask pixels are extremely few, return NaN.
    4. Branch:
       - metric_type == 'normal_world': return (MeanAngularError, mask_count, aux).
       - metric_type == 'color': return (PSNR, mask_count, MSE) in sRGB space (albedo).
       - metric_type == 'scalar': return (L1, mask_count, aux) for roughness/metallic.
    """
    if gen_tensor.shape != gt_tensor.shape:
        raise ValueError("Gen/GT tensor shapes must match.")

    gen_rgb = gen_tensor[:, :3, ...]
    gt_rgb = gt_tensor[:, :3, ...]
    mask = extract_alpha_mask(gen_tensor, gt_tensor)
    mask_f = mask.float()
    mask_count = mask_f.sum().item()
    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        diff = (gen_tensor - gt_tensor).abs()
        diff_gray = diff.mean(dim=1).clamp(0.0, 1.0)
        for idx in range(diff_gray.shape[0]):
            name = None
            if debug_names and idx < len(debug_names):
                name = debug_names[idx]
            else:
                name = obj_ids[idx] if obj_ids and idx < len(obj_ids) else f"sample_{idx}"
            safe_name = name.replace(os.sep, "_")
            tag = debug_tag or metric_type
            diff_path = os.path.join(debug_dir, f"{safe_name}_{tag}_diff.png")
            diff_img = (diff_gray[idx].detach().cpu().numpy() * 255.0).astype(np.uint8)
            Image.fromarray(diff_img, mode="L").save(diff_path)

    if mask_count < MIN_MASK_PIXELS:
        return float("nan"), 0.0, float("nan")

    if metric_type in {"normal_world", "normal"}:
        if debug:
            for idx in range(gen_rgb.shape[0]):
                oid = obj_ids[idx] if obj_ids and idx < len(obj_ids) else f"sample_{idx}"
                mask_i = mask[idx].squeeze(0)
                if mask_i.any():
                    values = gen_rgb[idx][:, mask_i]
                    var = values.var(unbiased=False).item()
                else:
                    var = float("nan")
                print(f"[Debug {oid}] Normal Variance: {var:.6f}")

        gen_vec = gen_rgb * 2.0 - 1.0
        gt_vec = gt_rgb * 2.0 - 1.0
        gen_vec = F.normalize(gen_vec, dim=1, eps=1e-6)
        gt_vec = F.normalize(gt_vec, dim=1, eps=1e-6)
        dot = (gen_vec * gt_vec).sum(dim=1).clamp(-1.0, 1.0)
        ang_err = torch.acos(dot) * (180.0 / math.pi)
        mean_angular_error = (ang_err * mask.squeeze(1).float()).sum() / mask_count
        return mean_angular_error.item(), mask_count, float("nan")

    if metric_type == "color":
        if debug:
            for idx in range(gen_rgb.shape[0]):
                oid = obj_ids[idx] if obj_ids and idx < len(obj_ids) else f"sample_{idx}"
                mask_i = mask[idx].squeeze(0)
                denom = mask_i.sum().item() * 3.0
                if denom > 0:
                    gen_mean = (gen_rgb[idx] * mask_i).sum().item() / denom
                    gt_mean = (gt_rgb[idx] * mask_i).sum().item() / denom
                else:
                    gen_mean = float("nan")
                    gt_mean = float("nan")
                print(
                    f"[Debug {oid}] Channel: {metric_type} | "
                    f"Gen Mean: {gen_mean:.4f} | GT Mean: {gt_mean:.4f}"
                )
        # Albedo PSNR/SSIM are computed in sRGB space (renderer outputs sRGB PNGs).
        diff = (gen_rgb - gt_rgb) * mask_f
        mse = diff.pow(2).sum() / (mask_count * 3.0)
        mse_val = mse.item()
        psnr = float("inf") if mse_val == 0 else -10.0 * math.log10(mse_val)
        return psnr, mask_count, mse_val

    if metric_type == "scalar":
        diff = (gen_rgb - gt_rgb) * mask_f
        l1 = diff.abs().sum() / (mask_count * 3.0)
        return l1.item(), mask_count, float("nan")

    raise ValueError(f"Unknown metric_type: {metric_type}")


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    kernel_2d = gaussian[:, None] @ gaussian[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def compute_masked_ssim(
    gen_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tuple[float, float]:
    gen_rgb = gen_tensor[:, :3, ...]
    gt_rgb = gt_tensor[:, :3, ...]
    mask = extract_alpha_mask(gen_tensor, gt_tensor).float()
    mask_count = mask.sum().item()
    if mask_count < MIN_MASK_PIXELS:
        return float("nan"), 0.0

    device = gen_rgb.device
    dtype = gen_rgb.dtype
    kernel_2d = _gaussian_kernel(window_size, sigma, device, dtype)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(gen_rgb.shape[1], 1, 1, 1)
    pad = window_size // 2

    masked_gen = gen_rgb * mask
    masked_gt = gt_rgb * mask
    mask_sum = F.conv2d(mask, kernel[:1], padding=pad).clamp_min(1e-6)

    mu_gen = F.conv2d(masked_gen, kernel, padding=pad, groups=gen_rgb.shape[1]) / mask_sum
    mu_gt = F.conv2d(masked_gt, kernel, padding=pad, groups=gt_rgb.shape[1]) / mask_sum

    sigma_gen = F.conv2d(masked_gen * masked_gen, kernel, padding=pad, groups=gen_rgb.shape[1]) / mask_sum - mu_gen.pow(2)
    sigma_gt = F.conv2d(masked_gt * masked_gt, kernel, padding=pad, groups=gt_rgb.shape[1]) / mask_sum - mu_gt.pow(2)
    sigma_gen_gt = (
        F.conv2d(masked_gen * masked_gt, kernel, padding=pad, groups=gen_rgb.shape[1]) / mask_sum - mu_gen * mu_gt
    )

    sigma_gen = torch.clamp(sigma_gen, min=0.0)
    sigma_gt = torch.clamp(sigma_gt, min=0.0)

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    ssim_map = ((2 * mu_gen * mu_gt + c1) * (2 * sigma_gen_gt + c2)) / (
        (mu_gen.pow(2) + mu_gt.pow(2) + c1) * (sigma_gen + sigma_gt + c2)
    )
    ssim_map = ssim_map.mean(dim=1, keepdim=True)
    ssim_val = (ssim_map * mask).sum() / mask_count
    return ssim_val.item(), mask_count


# ===================== Multi-View Consistency Metrics =====================
# These metrics detect Janus (multi-face) problems and texture flickering
# by measuring consistency across adjacent views.


def load_transforms_json(transforms_path: str) -> Dict:
    """Load camera transforms from transforms.json."""
    with open(transforms_path, "r") as f:
        return json.load(f)


def get_camera_intrinsics(transforms: Dict) -> Tuple[float, float, float, float, int, int]:
    """Extract camera intrinsics from transforms dict."""
    intrinsics = transforms.get("intrinsics", {})
    fx = intrinsics.get("fx", 711.111)
    fy = intrinsics.get("fy", 711.111)
    cx = intrinsics.get("cx", 256.0)
    cy = intrinsics.get("cy", 256.0)
    w = intrinsics.get("w", 512)
    h = intrinsics.get("h", 512)
    return fx, fy, cx, cy, w, h


def world_to_camera_matrix(frame: Dict) -> np.ndarray:
    """Get 4x4 world-to-camera matrix from frame dict."""
    return np.array(frame["world_to_camera"], dtype=np.float32)


def camera_to_world_matrix(frame: Dict) -> np.ndarray:
    """Get 4x4 camera-to-world matrix (inverse of world-to-camera)."""
    w2c = world_to_camera_matrix(frame)
    return np.linalg.inv(w2c)


def compute_relative_pose(frame1: Dict, frame2: Dict) -> np.ndarray:
    """Compute relative pose from camera1 to camera2 coordinate system.
    
    Returns T_2_1: transforms points from camera1 coords to camera2 coords.
    """
    c2w_1 = camera_to_world_matrix(frame1)  # cam1 -> world
    w2c_2 = world_to_camera_matrix(frame2)   # world -> cam2
    return w2c_2 @ c2w_1  # cam1 -> world -> cam2


def select_adjacent_view_pairs(
    frames: List[Dict],
    num_pairs: int,
    seed: int = 42,
    max_angle_deg: float = 90.0,
) -> List[Tuple[int, int]]:
    """Select pairs of views for consistency evaluation.
    
    Selects view pairs with the smallest angular separation for meaningful
    reprojection metrics. Always returns the pairs with smallest angles
    available in the dataset.
    
    Args:
        frames: List of frame dicts containing 'world_to_camera' matrices
        num_pairs: Maximum number of pairs to return
        seed: Random seed for reproducibility
        max_angle_deg: Unused (kept for API compatibility)
        
    Returns:
        List of (frame_idx1, frame_idx2) pairs sorted by angular separation.
    """
    n_frames = len(frames)
    if n_frames < 2:
        return []
    
    # Extract camera positions from world_to_camera matrices
    # world_to_camera: camera_point = M @ world_point
    # Camera position in world = inverse(M)[:3, 3]
    # For 4x4 matrix M = [R|t], camera position = -R^T @ t
    positions = []
    for f in frames:
        m = np.array(f["world_to_camera"], dtype=np.float64)
        R = m[:3, :3]
        t = m[:3, 3]
        cam_pos = -R.T @ t
        positions.append(cam_pos)
    positions = np.array(positions)
    
    # Compute angles between all pairs of camera view directions
    # Assuming cameras look toward origin, view direction ~ -cam_pos (normalized)
    view_dirs = -positions / (np.linalg.norm(positions, axis=1, keepdims=True) + 1e-8)
    
    # Compute all pair angles
    all_pairs_with_angles = []
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            cos_angle = np.clip(np.dot(view_dirs[i], view_dirs[j]), -1.0, 1.0)
            angle = np.rad2deg(np.arccos(cos_angle))
            all_pairs_with_angles.append((i, j, angle))
    
    # Sort by angle (prefer smaller angles)
    all_pairs_with_angles.sort(key=lambda x: x[2])
    
    rng = np.random.default_rng(seed)
    
    # Take pairs with smallest angles, with some randomness
    top_candidates = all_pairs_with_angles[:min(len(all_pairs_with_angles), num_pairs * 2)]
    
    if len(top_candidates) > num_pairs:
        indices = rng.choice(len(top_candidates), size=num_pairs, replace=False)
        selected = [top_candidates[i] for i in sorted(indices)]
    else:
        selected = top_candidates
    
    return [(p[0], p[1]) for p in selected]


def load_image_rgba(path: str, device: torch.device) -> torch.Tensor:
    """Load image as RGBA float tensor [1, 4, H, W] in [0, 1]."""
    with Image.open(path) as img:
        img = img.convert("RGBA")
        np_img = np.array(img, dtype=np.uint8)
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).float().div(255.0)
    return tensor.unsqueeze(0).to(device)


def compute_cross_view_lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    lpips_model,
) -> float:
    """Compute LPIPS between two views using intersection of alpha masks.
    
    Args:
        img1, img2: (1, 4, H, W) RGBA tensors in [0, 1]
        lpips_model: LPIPS model instance
    
    Returns:
        LPIPS value (lower = more similar)
    """
    # Extract alpha and compute mask intersection
    alpha1 = img1[:, 3:4, ...]
    alpha2 = img2[:, 3:4, ...]
    mask = ((alpha1 > 0.5) & (alpha2 > 0.5)).float()
    
    mask_count = mask.sum().item()
    if mask_count < MIN_MASK_PIXELS:
        return float("nan")
    
    # Apply mask to RGB channels
    rgb1 = img1[:, :3, ...] * mask
    rgb2 = img2[:, :3, ...] * mask
    
    # LPIPS expects [-1, 1] input
    with torch.no_grad():
        lpips_val = lpips_model(rgb1 * 2.0 - 1.0, rgb2 * 2.0 - 1.0)
    
    return lpips_val.mean().item()


def compute_cross_view_l1(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> float:
    """Compute masked L1 between two views.
    
    Args:
        img1, img2: (1, 4, H, W) RGBA tensors in [0, 1]
    
    Returns:
        Mean L1 error over masked pixels
    """
    alpha1 = img1[:, 3:4, ...]
    alpha2 = img2[:, 3:4, ...]
    mask = ((alpha1 > 0.5) & (alpha2 > 0.5)).float()
    
    mask_count = mask.sum().item()
    if mask_count < MIN_MASK_PIXELS:
        return float("nan")
    
    rgb1 = img1[:, :3, ...]
    rgb2 = img2[:, :3, ...]
    
    diff = (rgb1 - rgb2).abs() * mask
    l1 = diff.sum() / (mask_count * 3.0)
    
    return l1.item()


def compute_normal_angular_consistency(
    normal1: torch.Tensor,
    normal2: torch.Tensor,
    relative_pose: np.ndarray,
) -> float:
    """Compute angular consistency of normals across views.
    
    For true multi-view consistency, normals from view1 should match normals from view2
    when transformed to the same coordinate system. This detects Janus problems where
    different views show completely different surface orientations.
    
    Args:
        normal1, normal2: (1, 4, H, W) normal maps in [0, 1] (will be converted to [-1, 1])
        relative_pose: 4x4 matrix transforming from camera1 to camera2 coords
    
    Returns:
        Mean angular error in degrees (lower = more consistent)
    """
    alpha1 = normal1[:, 3:4, ...]
    alpha2 = normal2[:, 3:4, ...]
    mask = ((alpha1 > 0.5) & (alpha2 > 0.5)).float()
    
    mask_count = mask.sum().item()
    if mask_count < MIN_MASK_PIXELS:
        return float("nan")
    
    # Convert normals from [0, 1] to [-1, 1] and normalize
    n1 = normal1[:, :3, ...] * 2.0 - 1.0
    n2 = normal2[:, :3, ...] * 2.0 - 1.0
    
    n1 = F.normalize(n1, dim=1, eps=1e-6)
    n2 = F.normalize(n2, dim=1, eps=1e-6)
    
    # For Janus detection, we compare the distribution of normal directions
    # rather than per-pixel correspondence (which would require depth for reprojection)
    # A consistent surface should have similar normal distributions across adjacent views
    
    # Compute cosine similarity at each pixel position
    # Note: This is a simplified metric; full reprojection would require depth maps
    dot = (n1 * n2).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    ang_err = torch.acos(dot) * (180.0 / math.pi)
    
    # Compute mean angular error over masked pixels
    mean_error = (ang_err * mask).sum() / mask_count
    
    return mean_error.item()


def compute_normal_distribution_divergence(
    normal1: torch.Tensor,
    normal2: torch.Tensor,
) -> float:
    """Compute divergence of normal distributions between two views.
    
    This metric quantifies whether the distribution of surface normals is consistent
    across views. Janus problems typically show very different normal distributions
    (e.g., one view faces left, another faces right).
    
    Args:
        normal1, normal2: (1, 4, H, W) normal maps in [0, 1]
    
    Returns:
        KL divergence approximation (lower = more consistent distributions)
    """
    alpha1 = normal1[:, 3:4, ...]
    alpha2 = normal2[:, 3:4, ...]
    
    mask1 = (alpha1 > 0.5).squeeze()
    mask2 = (alpha2 > 0.5).squeeze()
    
    if mask1.sum() < MIN_MASK_PIXELS or mask2.sum() < MIN_MASK_PIXELS:
        return float("nan")
    
    # Convert normals and extract valid pixels
    n1 = normal1[:, :3, ...].squeeze().permute(1, 2, 0)  # (H, W, 3)
    n2 = normal2[:, :3, ...].squeeze().permute(1, 2, 0)
    
    n1 = n1 * 2.0 - 1.0
    n2 = n2 * 2.0 - 1.0
    
    n1_valid = n1[mask1]  # (N1, 3)
    n2_valid = n2[mask2]  # (N2, 3)
    
    n1_valid = F.normalize(n1_valid, dim=1, eps=1e-6)
    n2_valid = F.normalize(n2_valid, dim=1, eps=1e-6)
    
    # Compute mean direction for each view
    mean1 = n1_valid.mean(dim=0)
    mean2 = n2_valid.mean(dim=0)
    mean1 = F.normalize(mean1.unsqueeze(0), dim=1).squeeze()
    mean2 = F.normalize(mean2.unsqueeze(0), dim=1).squeeze()
    
    # Angular difference between mean directions
    cos_sim = (mean1 * mean2).sum().clamp(-1.0, 1.0)
    mean_ang_diff = torch.acos(cos_sim) * (180.0 / math.pi)
    
    # Variance of normals in each view (spread around mean)
    var1 = (1.0 - (n1_valid @ mean1.unsqueeze(1)).squeeze().clamp(-1.0, 1.0)).mean()
    var2 = (1.0 - (n2_valid @ mean2.unsqueeze(1)).squeeze().clamp(-1.0, 1.0)).mean()
    
    # Combined divergence score: mean direction difference + variance difference
    var_diff = (var1 - var2).abs()
    
    return (mean_ang_diff.item() + var_diff.item() * 90.0)  # Scale variance to similar range


def compute_multiview_consistency_metrics(
    base_gt_dir: str,
    base_gen_dir: str,
    unlit_subdir: str,
    lit_subdir: str,
    obj_ids: List[str],
    device: torch.device,
    lpips_model,
    num_pairs: int = 5,
    channel: str = "albedo",
    debug: bool = False,
) -> Dict[str, float]:
    """Compute multi-view consistency metrics across all objects.
    
    These metrics compare image content directly between view pairs without
    requiring depth maps or 3D reprojection. They work well even with sparse
    view datasets (e.g., fibonacci_sphere with 16 views).
    
    Metrics computed:
    - CrossView_LPIPS: Perceptual consistency between view pairs (lower = better)
    - CrossView_L1: Pixel-level consistency (lower = better)
    - Normal_Consistency: Angular consistency of surface normals (lower = better)
    - Normal_Distribution_Div: Divergence of normal distributions (lower = better)
    - GT_CrossView_LPIPS/L1: Same metrics for GT (for comparison)
    - View_Pair_Angle: Average angle between selected view pairs (informational)
    
    Args:
        base_gt_dir: Root GT render directory
        base_gen_dir: Root generated render directory
        unlit_subdir: Subdirectory for unlit renders
        lit_subdir: Subdirectory for lit renders
        obj_ids: List of object IDs to evaluate
        device: Torch device
        lpips_model: LPIPS model instance (or None)
        num_pairs: Number of view pairs to sample per object
        channel: Which channel to use ("albedo", "lit", or "normal")
        debug: Whether to print debug info
    
    Returns:
        Dictionary of metric name -> value
    """
    gen_lpips_scores: List[float] = []
    gen_l1_scores: List[float] = []
    gt_lpips_scores: List[float] = []
    gt_l1_scores: List[float] = []
    normal_consistency_scores: List[float] = []
    normal_dist_div_scores: List[float] = []
    view_pair_angles: List[float] = []  # Track angles for diagnostics
    mask_overlap_ratios: List[float] = []  # Track overlap for diagnostics
    
    print(f"Computing multi-view consistency metrics ({channel} channel)...")
    
    for obj_idx, obj_id in enumerate(obj_ids):
        # Load transforms.json
        gen_transforms_path = os.path.join(base_gen_dir, obj_id, "transforms.json")
        gt_transforms_path = os.path.join(base_gt_dir, obj_id, "transforms.json")
        
        if not os.path.exists(gen_transforms_path):
            # Try GT transforms if gen doesn't have its own
            if os.path.exists(gt_transforms_path):
                gen_transforms_path = gt_transforms_path
            else:
                print(f"[WARN] No transforms.json for {obj_id}, skipping consistency metrics.")
                continue
        
        try:
            transforms = load_transforms_json(gen_transforms_path)
        except Exception as e:
            print(f"[WARN] Failed to load transforms for {obj_id}: {e}")
            continue
        
        frames = transforms.get("frames", [])
        if len(frames) < 2:
            print(f"[WARN] Not enough frames for {obj_id}, skipping.")
            continue
        
        # Select view pairs (prefer pairs with smaller angular separation)
        pairs = select_adjacent_view_pairs(frames, num_pairs, seed=42 + obj_idx)
        
        # Compute view directions for angle calculation
        positions = []
        for f in frames:
            m = np.array(f["world_to_camera"], dtype=np.float64)
            R = m[:3, :3]
            t = m[:3, 3]
            cam_pos = -R.T @ t
            positions.append(cam_pos)
        positions_arr = np.array(positions)
        view_dirs = -positions_arr / (np.linalg.norm(positions_arr, axis=1, keepdims=True) + 1e-8)
        
        for idx1, idx2 in pairs:
            frame1 = frames[idx1]
            frame2 = frames[idx2]
            prefix1 = frame1.get("file_prefix", f"{idx1:03d}")
            prefix2 = frame2.get("file_prefix", f"{idx2:03d}")
            
            # Record angle between this view pair
            cos_angle = np.clip(np.dot(view_dirs[idx1], view_dirs[idx2]), -1.0, 1.0)
            pair_angle = np.rad2deg(np.arccos(cos_angle))
            view_pair_angles.append(pair_angle)
            
            # Determine image paths based on channel
            if channel == "lit":
                gen_dir = os.path.join(base_gen_dir, obj_id, lit_subdir)
                gt_dir = os.path.join(base_gt_dir, obj_id, lit_subdir)
                
                # Handle HDRI subdirectories
                if os.path.isdir(gen_dir):
                    subdirs = [d for d in os.listdir(gen_dir) if os.path.isdir(os.path.join(gen_dir, d))]
                    if subdirs:
                        gen_dir = os.path.join(gen_dir, subdirs[0])
                        gt_dir = os.path.join(gt_dir, subdirs[0])
                
                gen_path1 = os.path.join(gen_dir, f"{prefix1}_lit.png")
                gen_path2 = os.path.join(gen_dir, f"{prefix2}_lit.png")
                gt_path1 = os.path.join(gt_dir, f"{prefix1}_lit.png")
                gt_path2 = os.path.join(gt_dir, f"{prefix2}_lit.png")
            else:
                # albedo or normal
                suffix = channel
                gen_dir = os.path.join(base_gen_dir, obj_id, unlit_subdir)
                gt_dir = os.path.join(base_gt_dir, obj_id, unlit_subdir)
                gen_path1 = os.path.join(gen_dir, f"{prefix1}_{suffix}.png")
                gen_path2 = os.path.join(gen_dir, f"{prefix2}_{suffix}.png")
                gt_path1 = os.path.join(gt_dir, f"{prefix1}_{suffix}.png")
                gt_path2 = os.path.join(gt_dir, f"{prefix2}_{suffix}.png")
            
            # Check if files exist
            if not all(os.path.exists(p) for p in [gen_path1, gen_path2]):
                if debug:
                    print(f"[DEBUG] Missing gen files for {obj_id} pair ({idx1}, {idx2})")
                continue
            
            try:
                gen_img1 = load_image_rgba(gen_path1, device)
                gen_img2 = load_image_rgba(gen_path2, device)
                
                # Record mask overlap ratio
                alpha1 = gen_img1[:, 3:4, ...] > 0.5
                alpha2 = gen_img2[:, 3:4, ...] > 0.5
                mask_overlap = (alpha1 & alpha2).float().mean().item()
                if mask_overlap > 0:
                    mask_overlap_ratios.append(mask_overlap)
                
                # Compute generated metrics
                gen_l1 = compute_cross_view_l1(gen_img1, gen_img2)
                if math.isfinite(gen_l1):
                    gen_l1_scores.append(gen_l1)
                
                if lpips_model is not None:
                    gen_lpips = compute_cross_view_lpips(gen_img1, gen_img2, lpips_model)
                    if math.isfinite(gen_lpips):
                        gen_lpips_scores.append(gen_lpips)
                
                # Load and compute GT metrics for comparison
                if all(os.path.exists(p) for p in [gt_path1, gt_path2]):
                    gt_img1 = load_image_rgba(gt_path1, device)
                    gt_img2 = load_image_rgba(gt_path2, device)
                    
                    gt_l1 = compute_cross_view_l1(gt_img1, gt_img2)
                    if math.isfinite(gt_l1):
                        gt_l1_scores.append(gt_l1)
                    
                    if lpips_model is not None:
                        gt_lpips = compute_cross_view_lpips(gt_img1, gt_img2, lpips_model)
                        if math.isfinite(gt_lpips):
                            gt_lpips_scores.append(gt_lpips)
                    
                    del gt_img1, gt_img2
                
                del gen_img1, gen_img2
                
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Error processing {obj_id} pair ({idx1}, {idx2}): {e}")
                continue
            
            # Also compute normal consistency if we're not already using normals
            if channel != "normal":
                normal_gen_path1 = os.path.join(base_gen_dir, obj_id, unlit_subdir, f"{prefix1}_normal.png")
                normal_gen_path2 = os.path.join(base_gen_dir, obj_id, unlit_subdir, f"{prefix2}_normal.png")
                
                if os.path.exists(normal_gen_path1) and os.path.exists(normal_gen_path2):
                    try:
                        normal1 = load_image_rgba(normal_gen_path1, device)
                        normal2 = load_image_rgba(normal_gen_path2, device)
                        
                        rel_pose = compute_relative_pose(frame1, frame2)
                        
                        ang_consistency = compute_normal_angular_consistency(normal1, normal2, rel_pose)
                        if math.isfinite(ang_consistency):
                            normal_consistency_scores.append(ang_consistency)
                        
                        dist_div = compute_normal_distribution_divergence(normal1, normal2)
                        if math.isfinite(dist_div):
                            normal_dist_div_scores.append(dist_div)
                        
                        del normal1, normal2
                    except Exception as e:
                        if debug:
                            print(f"[DEBUG] Error processing normals for {obj_id}: {e}")
        
        if (obj_idx + 1) % 10 == 0:
            print(f"  Processed {obj_idx + 1}/{len(obj_ids)} objects for consistency...")
            _clear_cuda_cache()
    
    # Compute final metrics
    metrics: Dict[str, float] = {}
    
    if gen_lpips_scores:
        metrics["CrossView_LPIPS"] = float(np.mean(gen_lpips_scores))
    if gen_l1_scores:
        metrics["CrossView_L1"] = float(np.mean(gen_l1_scores))
    if gt_lpips_scores:
        metrics["GT_CrossView_LPIPS"] = float(np.mean(gt_lpips_scores))
    if gt_l1_scores:
        metrics["GT_CrossView_L1"] = float(np.mean(gt_l1_scores))
    if normal_consistency_scores:
        metrics["Normal_Consistency"] = float(np.mean(normal_consistency_scores))
    if normal_dist_div_scores:
        metrics["Normal_Distribution_Div"] = float(np.mean(normal_dist_div_scores))
    
    # Compute relative metrics (Gen vs GT ratio) - useful for comparing methods
    if "CrossView_LPIPS" in metrics and "GT_CrossView_LPIPS" in metrics:
        gt_val = metrics["GT_CrossView_LPIPS"]
        if gt_val > 1e-6:
            metrics["CrossView_LPIPS_Ratio"] = metrics["CrossView_LPIPS"] / gt_val
    
    if "CrossView_L1" in metrics and "GT_CrossView_L1" in metrics:
        gt_val = metrics["GT_CrossView_L1"]
        if gt_val > 1e-6:
            metrics["CrossView_L1_Ratio"] = metrics["CrossView_L1"] / gt_val
    
    # Add diagnostic info about view pairs
    if view_pair_angles:
        metrics["View_Pair_Angle_Mean"] = float(np.mean(view_pair_angles))
        metrics["View_Pair_Angle_Min"] = float(np.min(view_pair_angles))
    if mask_overlap_ratios:
        metrics["Mask_Overlap_Ratio"] = float(np.mean(mask_overlap_ratios))
    
    return metrics


# ===================== Depth-based Reprojection Error (when depth maps available) =====================


def load_depth_image(path: str, device: torch.device, near: float = 0.1, far: float = 4.0) -> torch.Tensor:
    """Load normalized depth image and convert back to metric depth.
    
    Args:
        path: Path to depth image (grayscale PNG, values in [0, 1])
        device: Torch device
        near: Near plane used during rendering
        far: Far plane used during rendering
    
    Returns:
        Depth tensor (1, 1, H, W) with metric depth values
    """
    with Image.open(path) as img:
        img = img.convert("L")  # Grayscale
        np_depth = np.array(img, dtype=np.float32) / 255.0
    
    # Convert normalized [0, 1] back to metric depth
    metric_depth = np_depth * (far - near) + near
    
    tensor = torch.from_numpy(metric_depth).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def compute_reprojection_error(
    img1: torch.Tensor,
    img2: torch.Tensor,
    depth1: torch.Tensor,
    intrinsics: Dict[str, float],
    relative_pose: np.ndarray,
    lpips_model=None,
) -> Tuple[float, float, float]:
    """Compute reprojection error by warping img1 to img2's viewpoint.
    
    This is the gold standard for multi-view consistency evaluation (MEt3R, MVGBench).
    
    Given:
    - Image I1 at viewpoint V1 with depth D1
    - Image I2 at viewpoint V2
    - Relative pose T_{2,1} (transforms V1 coords to V2 coords)
    - Camera intrinsics K
    
    Process:
    1. Unproject I1 pixels to 3D using D1 and K^-1
    2. Transform 3D points by T_{2,1}
    3. Project to V2 image plane using K
    4. Sample I2 at projected coordinates -> I1_warped
    5. Compute error between I1 and I1_warped
    
    Args:
        img1: Source image (1, 4, H, W) RGBA in [0, 1]
        img2: Target image (1, 4, H, W) RGBA in [0, 1]
        depth1: Depth map for img1 (1, 1, H, W) metric depth
        intrinsics: Camera intrinsics dict with fx, fy, cx, cy
        relative_pose: 4x4 matrix T_{2,1}
        lpips_model: Optional LPIPS model for perceptual error
    
    Returns:
        Tuple of (L1_error, LPIPS_error, valid_ratio)
    """
    device = img1.device
    _, _, H, W = img1.shape
    
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    
    # Create pixel grid
    u = torch.arange(W, device=device, dtype=torch.float32)
    v = torch.arange(H, device=device, dtype=torch.float32)
    uu, vv = torch.meshgrid(u, v, indexing='xy')  # (H, W)
    
    # Get depth values
    d1 = depth1.squeeze()  # (H, W)
    
    # Unproject to camera 1 coordinates
    # x = (u - cx) * d / fx
    # y = (v - cy) * d / fy
    # z = d
    x1 = (uu - cx) * d1 / fx
    y1 = (vv - cy) * d1 / fy
    z1 = d1
    
    # Stack to 3D points (H, W, 3)
    pts1 = torch.stack([x1, y1, z1], dim=-1)
    
    # Add homogeneous coordinate
    ones = torch.ones_like(z1)
    pts1_h = torch.stack([x1, y1, z1, ones], dim=-1)  # (H, W, 4)
    
    # Transform to camera 2 coordinates
    T = torch.from_numpy(relative_pose).float().to(device)  # (4, 4)
    pts1_flat = pts1_h.reshape(-1, 4)  # (H*W, 4)
    pts2_flat = (T @ pts1_flat.T).T  # (H*W, 4)
    pts2 = pts2_flat[:, :3].reshape(H, W, 3)  # (H, W, 3)
    
    # Project to image 2
    x2 = pts2[..., 0]
    y2 = pts2[..., 1]
    z2 = pts2[..., 2]
    
    # Avoid division by zero
    z2_safe = z2.clamp(min=1e-6)
    
    u2 = fx * x2 / z2_safe + cx
    v2 = fy * y2 / z2_safe + cy
    
    # Normalize to [-1, 1] for grid_sample
    u2_norm = 2.0 * u2 / (W - 1) - 1.0
    v2_norm = 2.0 * v2 / (H - 1) - 1.0
    grid = torch.stack([u2_norm, v2_norm], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    
    # Sample img2 at projected coordinates
    # This gives us the colors from img2 at the locations where img1 pixels project to
    img2_sampled = F.grid_sample(
        img2,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )  # (1, 4, H, W)
    
    # Create validity mask:
    # 1. Projected coords within image bounds
    # 2. Positive depth in both views
    # 3. Alpha valid in both images
    in_bounds = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
    depth_valid = (z2 > 0) & (d1 > 0)
    alpha_valid = (img1[:, 3, :, :] > 0.5) & (img2_sampled[:, 3, :, :] > 0.5)
    
    valid_mask = (in_bounds & depth_valid & alpha_valid.squeeze(0)).float().unsqueeze(0).unsqueeze(0)
    valid_count = valid_mask.sum().item()
    
    if valid_count < MIN_MASK_PIXELS:
        return float("nan"), float("nan"), 0.0
    
    # Compute L1 error on RGB
    # Compare img1 with img2 sampled at projected locations
    # If consistent, img1[p] should equal img2[project(p)]
    rgb1 = img1[:, :3, :, :]
    rgb2_sampled = img2_sampled[:, :3, :, :]
    l1_diff = (rgb1 - rgb2_sampled).abs() * valid_mask
    l1_error = l1_diff.sum() / (valid_count * 3.0)
    
    # Compute LPIPS error if model provided
    lpips_error = float("nan")
    if lpips_model is not None:
        rgb1_masked = rgb1 * valid_mask
        rgb2_sampled_masked = rgb2_sampled * valid_mask
        with torch.no_grad():
            lpips_val = lpips_model(rgb1_masked * 2.0 - 1.0, rgb2_sampled_masked * 2.0 - 1.0)
            lpips_error = lpips_val.mean().item()
    
    valid_ratio = valid_count / (H * W)
    
    return l1_error.item(), lpips_error, valid_ratio


def compute_reprojection_metrics(
    base_gt_dir: str,
    base_gen_dir: str,
    unlit_subdir: str,
    obj_ids: List[str],
    device: torch.device,
    lpips_model,
    num_pairs: int = 5,
    debug: bool = False,
) -> Dict[str, float]:
    """Compute reprojection-based multi-view consistency metrics.
    
    This requires depth maps to be rendered. If depth maps are not available,
    returns empty dict.
    
    Args:
        base_gt_dir: Root GT render directory
        base_gen_dir: Root generated render directory
        unlit_subdir: Subdirectory for unlit renders
        obj_ids: List of object IDs
        device: Torch device
        lpips_model: LPIPS model instance
        num_pairs: Number of view pairs per object
        debug: Enable debug output
    
    Returns:
        Dictionary of reprojection metrics
    """
    gen_reproj_l1: List[float] = []
    gen_reproj_lpips: List[float] = []
    gt_reproj_l1: List[float] = []
    gt_reproj_lpips: List[float] = []
    valid_ratios: List[float] = []
    skipped_pairs = 0
    total_pairs = 0
    
    # Check if depth maps are available
    sample_obj = obj_ids[0] if obj_ids else None
    if sample_obj:
        sample_depth = os.path.join(base_gen_dir, sample_obj, unlit_subdir, "000_depth.png")
        if not os.path.exists(sample_depth):
            print("[INFO] Depth maps not found in gen dir. Skipping reprojection metrics.")
            print("       Run render_gen_aligned.py with depth enabled to generate depth maps.")
            return {}
    
    print("Computing reprojection-based consistency metrics...")
    
    for obj_idx, obj_id in enumerate(obj_ids):
        # Load transforms
        transforms_path = os.path.join(base_gt_dir, obj_id, "transforms.json")
        if not os.path.exists(transforms_path):
            transforms_path = os.path.join(base_gen_dir, obj_id, "transforms.json")
        
        if not os.path.exists(transforms_path):
            continue
        
        try:
            transforms = load_transforms_json(transforms_path)
        except Exception:
            continue
        
        frames = transforms.get("frames", [])
        intrinsics = transforms.get("intrinsics", {})
        
        # Get depth normalization params
        depth_meta = transforms.get("meta", {}).get("depth", {})
        depth_near = depth_meta.get("near", 0.1)
        depth_far = depth_meta.get("far", 4.0)
        
        if len(frames) < 2:
            continue
        
        pairs = select_adjacent_view_pairs(frames, num_pairs, seed=42 + obj_idx)
        
        for idx1, idx2 in pairs:
            frame1 = frames[idx1]
            frame2 = frames[idx2]
            prefix1 = frame1.get("file_prefix", f"{idx1:03d}")
            prefix2 = frame2.get("file_prefix", f"{idx2:03d}")
            
            gen_dir = os.path.join(base_gen_dir, obj_id, unlit_subdir)
            gt_dir = os.path.join(base_gt_dir, obj_id, unlit_subdir)
            
            # Paths for albedo and depth
            gen_albedo1 = os.path.join(gen_dir, f"{prefix1}_albedo.png")
            gen_albedo2 = os.path.join(gen_dir, f"{prefix2}_albedo.png")
            gen_depth1 = os.path.join(gen_dir, f"{prefix1}_depth.png")
            
            gt_albedo1 = os.path.join(gt_dir, f"{prefix1}_albedo.png")
            gt_albedo2 = os.path.join(gt_dir, f"{prefix2}_albedo.png")
            gt_depth1 = os.path.join(gt_dir, f"{prefix1}_depth.png")
            
            # Check files exist
            if not all(os.path.exists(p) for p in [gen_albedo1, gen_albedo2, gen_depth1]):
                continue
            
            try:
                total_pairs += 1
                # Compute relative pose
                rel_pose = compute_relative_pose(frame1, frame2)
                
                # Load images
                gen_img1 = load_image_rgba(gen_albedo1, device)
                gen_img2 = load_image_rgba(gen_albedo2, device)
                gen_d1 = load_depth_image(gen_depth1, device, depth_near, depth_far)
                
                # Compute reprojection error
                l1, lpips_err, valid_ratio = compute_reprojection_error(
                    gen_img1, gen_img2, gen_d1, intrinsics, rel_pose, lpips_model
                )
                
                if valid_ratio < 0.01:  # Less than 1% valid pixels
                    skipped_pairs += 1
                    if debug:
                        print(f"[DEBUG] {obj_id} pair ({idx1},{idx2}): valid_ratio={valid_ratio:.4f}, skipping")
                    del gen_img1, gen_img2, gen_d1
                    continue
                
                if math.isfinite(l1):
                    gen_reproj_l1.append(l1)
                if math.isfinite(lpips_err):
                    gen_reproj_lpips.append(lpips_err)
                if valid_ratio > 0:
                    valid_ratios.append(valid_ratio)
                
                del gen_img1, gen_img2, gen_d1
                
                # GT reprojection (for reference)
                if all(os.path.exists(p) for p in [gt_albedo1, gt_albedo2, gt_depth1]):
                    gt_img1 = load_image_rgba(gt_albedo1, device)
                    gt_img2 = load_image_rgba(gt_albedo2, device)
                    gt_d1 = load_depth_image(gt_depth1, device, depth_near, depth_far)
                    
                    gt_l1, gt_lpips_err, _ = compute_reprojection_error(
                        gt_img1, gt_img2, gt_d1, intrinsics, rel_pose, lpips_model
                    )
                    
                    if math.isfinite(gt_l1):
                        gt_reproj_l1.append(gt_l1)
                    if math.isfinite(gt_lpips_err):
                        gt_reproj_lpips.append(gt_lpips_err)
                    
                    del gt_img1, gt_img2, gt_d1
                
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Reprojection error for {obj_id}: {e}")
                continue
        
        if (obj_idx + 1) % 10 == 0:
            print(f"  Processed {obj_idx + 1}/{len(obj_ids)} objects for reprojection...")
            _clear_cuda_cache()
    
    # Report statistics
    if total_pairs > 0:
        skip_rate = 100 * skipped_pairs / total_pairs
        print(f"  Total pairs: {total_pairs}, Skipped (low overlap): {skipped_pairs} ({skip_rate:.1f}%)")
        
        if skipped_pairs == total_pairs:
            print("  [INFO] All view pairs skipped due to insufficient overlap (<1% valid pixels).")
            print("         This typically happens with sparse view datasets (e.g., fibonacci_sphere with <24 views)")
            print("         where view angles between any pair exceed ~30 degrees.")
            print("         Consider using CrossView_LPIPS/L1 metrics instead for such datasets.")
    
    # Compile metrics
    metrics: Dict[str, float] = {}
    
    if gen_reproj_l1:
        metrics["Reproj_L1"] = float(np.mean(gen_reproj_l1))
    if gen_reproj_lpips:
        metrics["Reproj_LPIPS"] = float(np.mean(gen_reproj_lpips))
    if gt_reproj_l1:
        metrics["GT_Reproj_L1"] = float(np.mean(gt_reproj_l1))
    if gt_reproj_lpips:
        metrics["GT_Reproj_LPIPS"] = float(np.mean(gt_reproj_lpips))
    if valid_ratios:
        metrics["Reproj_Valid_Ratio"] = float(np.mean(valid_ratios))
    
    # Ratio metrics
    if "Reproj_L1" in metrics and "GT_Reproj_L1" in metrics:
        gt_val = metrics["GT_Reproj_L1"]
        if gt_val > 1e-6:
            metrics["Reproj_L1_Ratio"] = metrics["Reproj_L1"] / gt_val
    
    return metrics


def _clear_cuda_cache():
    """Clear CUDA cache and run garbage collection to prevent OOM."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass  # Ignore errors during cleanup
    gc.collect()


def encode_clip_texts(
    model,
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    clip_module = _lazy_import_clip()
    features: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            try:
                tokens = clip_module.tokenize(batch_prompts, truncate=True).to(device)
                text_feat = model.encode_text(tokens)
                text_feat = F.normalize(text_feat, dim=-1)
                # Move to CPU to reduce GPU memory pressure
                features.append(text_feat.cpu())
                del tokens, text_feat
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARN] CUDA OOM during text encoding at batch {start}, clearing cache and retrying...")
                    _clear_cuda_cache()
                    tokens = clip_module.tokenize(batch_prompts, truncate=True).to(device)
                    text_feat = model.encode_text(tokens)
                    text_feat = F.normalize(text_feat, dim=-1)
                    features.append(text_feat.cpu())
                    del tokens, text_feat
                else:
                    raise
            # Periodically clear cache
            if (start // batch_size) % 10 == 0:
                _clear_cuda_cache()
    result = torch.cat(features, dim=0).to(device)
    _clear_cuda_cache()
    return result


def encode_longclip_texts(
    model,
    longclip_module,
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
    context_length: int,
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            try:
                tokens = longclip_module.tokenize(
                    batch_prompts,
                    truncate=True,
                    context_length=context_length,
                ).to(device)
                text_feat = model.encode_text(tokens)
                text_feat = F.normalize(text_feat, dim=-1)
                # Move to CPU to reduce GPU memory pressure
                features.append(text_feat.cpu())
                del tokens, text_feat
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARN] CUDA OOM during LongCLIP text encoding at batch {start}, clearing cache and retrying...")
                    _clear_cuda_cache()
                    tokens = longclip_module.tokenize(
                        batch_prompts,
                        truncate=True,
                        context_length=context_length,
                    ).to(device)
                    text_feat = model.encode_text(tokens)
                    text_feat = F.normalize(text_feat, dim=-1)
                    features.append(text_feat.cpu())
                    del tokens, text_feat
                else:
                    raise
            # Periodically clear cache
            if (start // batch_size) % 10 == 0:
                _clear_cuda_cache()
    result = torch.cat(features, dim=0).to(device)
    _clear_cuda_cache()
    return result


def load_batch(
    paths: Sequence[str],
    clip_preprocess,
    device: torch.device,
    include_clip: bool,
    keep_alpha: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load images as float [0,1], uint8 [0,255], and optionally CLIP-preprocessed batches."""
    float_batch: List[torch.Tensor] = []
    uint8_batch: List[torch.Tensor] = []
    clip_batch: List[torch.Tensor] = []

    for path in paths:
        try:
            with Image.open(path) as img:
                if include_clip:
                    clip_img = img.convert("RGB")
                    if clip_preprocess is None:
                        raise ValueError("CLIP preprocess is required when include_clip=True.")
                    clip_tensor = clip_preprocess(clip_img).unsqueeze(0)  # type: ignore[arg-type]
                    clip_batch.append(clip_tensor)

                img = img.convert("RGBA" if keep_alpha else "RGB")
                np_img = np.array(img, dtype=np.uint8)

            uint8_tensor = torch.from_numpy(np_img).permute(2, 0, 1)
            float_tensor = uint8_tensor.float().div(255.0)

            uint8_batch.append(uint8_tensor)
            float_batch.append(float_tensor)
        except Exception as e:
            print(f"[WARN] Failed to load image {path}: {e}")
            raise

    # Use non_blocking=True for async transfer to reduce blocking
    float_tensor_batch = torch.stack(float_batch, dim=0).to(device, non_blocking=True)
    uint8_tensor_batch = torch.stack(uint8_batch, dim=0).to(device, non_blocking=True)
    clip_tensor_batch = torch.cat(clip_batch, dim=0).to(device, non_blocking=True) if include_clip and clip_batch else None
    return float_tensor_batch, uint8_tensor_batch, clip_tensor_batch


def save_metrics(final_metrics: Dict[str, float], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.lower().endswith(".csv"):
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in final_metrics.items():
                writer.writerow([k, v])
    else:
        with open(output_path, "w") as f:
            json.dump(final_metrics, f, indent=4)


def main() -> None:
    args = parse_args()
    metric_flags = parse_metrics_arg(args.metrics)
    base_gen_dir = resolve_gen_dir(args)
    base_gt_dir = args.base_gt_dir

    if not os.path.isdir(base_gen_dir):
        raise FileNotFoundError(f"Generated render directory not found: {base_gen_dir}")
    if not os.path.isdir(base_gt_dir):
        raise FileNotFoundError(f"GT render directory not found: {base_gt_dir}")

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cpu" and args.device.startswith("cuda"):
        print("CUDA requested but not available; falling back to CPU.")

    batch_size = max(1, args.batch_size)
    final_metrics: Dict[str, float] = {}
    debug_dir = os.path.abspath("debug_output") if args.debug else None

    obj_ids = list_object_ids(base_gen_dir)
    if not obj_ids:
        raise FileNotFoundError(f"No object directories found under {base_gen_dir}")

    do_clip_image = metric_flags["clip"]
    do_clip_text = metric_flags["clip"] and args.prompts_file is not None
    do_longclip_text = metric_flags["clip"] and args.prompts_file is not None
    do_lit_metrics = metric_flags["fid"] or metric_flags["kid"] or do_clip_image or do_clip_text or do_longclip_text
    do_unlit_metrics = metric_flags["psnr"] or metric_flags["ssim"] or metric_flags["lpips"]

    if do_lit_metrics:
        lit_hdris = detect_lit_hdris(base_gt_dir, base_gen_dir, args.lit_subdir, obj_ids)
        if lit_hdris:
            print(f"Detected {len(lit_hdris)} HDRI subdirs under '{args.lit_subdir}': {', '.join(lit_hdris)}")

        clip_model = None
        clip_preprocess = None
        if do_clip_image or do_clip_text:
            print("Loading CLIP model...")
            try:
                _clear_cuda_cache()
                clip_module = _lazy_import_clip()
                clip_model, clip_preprocess = clip_module.load(args.clip_model, device="cpu")
                clip_model = clip_model.to(device)
                clip_model.eval()
                for param in clip_model.parameters():
                    param.requires_grad = False
                print(f"CLIP model loaded: {args.clip_model}")
                _clear_cuda_cache()
            except Exception as e:
                print(f"[ERROR] Failed to load CLIP model: {e}")
                traceback.print_exc()
                raise RuntimeError(f"CLIP model loading failed: {e}") from e

        longclip_model = None
        longclip_preprocess = None
        longclip_module = None
        if do_longclip_text:
            print("Loading LongCLIP model...")
            try:
                _clear_cuda_cache()
                longclip_model, longclip_preprocess, longclip_module, import_source = load_longclip_model(
                    args.longclip_model,
                    device,
                    args.longclip_root,
                )
                for param in longclip_model.parameters():
                    param.requires_grad = False
                print(f"LongCLIP model loaded from: {import_source}")
                _clear_cuda_cache()
            except Exception as e:
                print(f"[ERROR] Failed to load LongCLIP model: {e}")
                traceback.print_exc()
                raise RuntimeError(f"LongCLIP model loading failed: {e}") from e

        prompt_short_map = None
        prompt_long_map = None
        clip_text_feature_map = None
        longclip_text_feature_map = None
        if do_clip_text or do_longclip_text:
            if not os.path.isfile(args.prompts_file):
                raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")
            prompt_short_map, prompt_long_map = load_prompt_pairs(args.prompts_file)

            if do_clip_text:
                if clip_model is None:
                    raise RuntimeError("CLIP model must be loaded to compute text-image similarity.")
                if not prompt_short_map:
                    raise ValueError("Prompts file contains no usable (obj_id, caption_short) pairs.")
                missing_short = [obj_id for obj_id in obj_ids if obj_id not in prompt_short_map]
                if missing_short:
                    preview = ", ".join(missing_short[:5])
                    raise ValueError(f"Missing caption_short for {len(missing_short)} obj_ids (e.g., {preview}).")
                prompt_texts = [prompt_short_map[obj_id] for obj_id in obj_ids]
                text_features = encode_clip_texts(clip_model, prompt_texts, device, batch_size)
                clip_text_feature_map = {obj_id: text_features[i] for i, obj_id in enumerate(obj_ids)}

            if do_longclip_text:
                if longclip_model is None or longclip_module is None:
                    raise RuntimeError("LongCLIP model must be loaded to compute text-image similarity.")
                if not prompt_long_map:
                    raise ValueError("Prompts file contains no usable (obj_id, caption_long) pairs.")
                missing_long = [obj_id for obj_id in obj_ids if obj_id not in prompt_long_map]
                if missing_long:
                    preview = ", ".join(missing_long[:5])
                    raise ValueError(f"Missing caption_long for {len(missing_long)} obj_ids (e.g., {preview}).")
                long_prompt_texts = [prompt_long_map[obj_id] for obj_id in obj_ids]
                long_text_features = encode_longclip_texts(
                    longclip_model,
                    longclip_module,
                    long_prompt_texts,
                    device,
                    batch_size,
                    args.longclip_context_length,
                )
                longclip_text_feature_map = {
                    obj_id: long_text_features[i] for i, obj_id in enumerate(obj_ids)
                }

        def run_lit_for_subdir(label: str, lit_subdir: str) -> Dict[str, float]:
            lit_gt_paths, lit_gen_paths = collect_lit_paths(
                base_gt_dir,
                base_gen_dir,
                lit_subdir,
                obj_ids,
            )
            if not lit_gt_paths:
                raise RuntimeError(f"No lit image pairs found for {label}.")
            print(f"Lit pairs ({label}): {len(lit_gt_paths)} images across {len(obj_ids)} objects.")
            return compute_lit_metrics_for_paths(
                lit_gt_paths,
                lit_gen_paths,
                base_gen_dir,
                batch_size,
                device,
                metric_flags,
                clip_model,
                clip_preprocess,
                longclip_model,
                longclip_preprocess,
                longclip_module,
                clip_text_feature_map,
                longclip_text_feature_map,
                do_clip_image,
                do_clip_text,
                do_longclip_text,
                args.kid_subset_size,
            )

        lit_metrics_by_hdri: Dict[str, Dict[str, float]] = {}
        if lit_hdris:
            for hdri in lit_hdris:
                lit_subdir = os.path.join(args.lit_subdir, hdri)
                print(f"== HDRI: {hdri} ==")
                lit_metrics = run_lit_for_subdir(hdri, lit_subdir)
                lit_metrics_by_hdri[hdri] = lit_metrics
                for metric_name, value in lit_metrics.items():
                    final_metrics[f"HDRI/{hdri}/{metric_name}"] = value

            mean_metrics = compute_mean_metrics(lit_metrics_by_hdri)
            for metric_name, value in mean_metrics.items():
                final_metrics[f"HDRI/Mean/{metric_name}"] = value
        else:
            lit_metrics = run_lit_for_subdir(args.lit_subdir, args.lit_subdir)
            for metric_name, value in lit_metrics.items():
                final_metrics[f"HDRI/Mean/{metric_name}"] = value

        print("Cleaning up CLIP/LongCLIP models...")
        if clip_model is not None:
            del clip_model
        if clip_preprocess is not None:
            del clip_preprocess
        if longclip_model is not None:
            del longclip_model
        if longclip_preprocess is not None:
            del longclip_preprocess
        if longclip_module is not None:
            del longclip_module
        if clip_text_feature_map is not None:
            del clip_text_feature_map
        if longclip_text_feature_map is not None:
            del longclip_text_feature_map
        clip_model = clip_preprocess = longclip_model = longclip_preprocess = None
        longclip_module = clip_text_feature_map = longclip_text_feature_map = None
        _clear_cuda_cache()

    if do_unlit_metrics:
        lpips_mod = _lazy_import_lpips()
        lpips_model = lpips_mod.LPIPS(net="vgg").to(device) if metric_flags["lpips"] else None
        if lpips_model:
            lpips_model.eval()
            _clear_cuda_cache()

        for channel in UNLIT_CHANNELS:
            gt_paths, gen_paths = collect_unlit_channel_paths(
                base_gt_dir,
                base_gen_dir,
                args.unlit_subdir,
                channel,
                obj_ids,
            )
            if not gt_paths:
                raise RuntimeError(f"No {channel} image pairs found.")

            label = CHANNEL_LABELS.get(channel, channel)
            print(f"Unlit {label}: {len(gt_paths)} image pairs.")

            total_mask = 0.0
            total_mse_weighted = 0.0
            total_l1_weighted = 0.0
            total_mean_angular_error_weighted = 0.0
            total_ssim_weighted = 0.0
            total_ssim_mask = 0.0
            lpips_scores: List[torch.Tensor] = []

            with torch.no_grad():
                for start in range(0, len(gen_paths), batch_size):
                    batch_gen_paths = gen_paths[start : start + batch_size]
                    batch_gt_paths = gt_paths[start : start + batch_size]
                    batch_obj_ids = [obj_id_from_path(path, base_gen_dir) for path in batch_gen_paths]
                    batch_debug_names = [
                        f"{obj_id}_{os.path.splitext(os.path.basename(path))[0]}"
                        for obj_id, path in zip(batch_obj_ids, batch_gen_paths)
                    ]

                    gen_float, _, _ = load_batch(
                        batch_gen_paths,
                        None,
                        device,
                        include_clip=False,
                        keep_alpha=True,
                    )
                    gt_float, _, _ = load_batch(
                        batch_gt_paths,
                        None,
                        device,
                        include_clip=False,
                        keep_alpha=True,
                    )

                    if channel == "normal":
                        mean_angular_error, mask_count, _ = compute_masked_metrics(
                            gen_float,
                            gt_float,
                            metric_type="normal_world",
                            debug=args.debug,
                            obj_ids=batch_obj_ids,
                            debug_names=batch_debug_names,
                            debug_dir=debug_dir,
                            debug_tag=channel,
                        )
                        if mask_count > 0:
                            total_mean_angular_error_weighted += mean_angular_error * mask_count
                            total_mask += mask_count
                    elif channel == "albedo":
                        _, mask_count, mse = compute_masked_metrics(
                            gen_float,
                            gt_float,
                            metric_type="color",
                            debug=args.debug,
                            obj_ids=batch_obj_ids,
                            debug_names=batch_debug_names,
                            debug_dir=debug_dir,
                            debug_tag=channel,
                        )
                        if mask_count > 0:
                            total_mask += mask_count
                            total_mse_weighted += mse * mask_count

                        if metric_flags["ssim"]:
                            ssim_val, ssim_mask = compute_masked_ssim(gen_float, gt_float)
                            if ssim_mask > 0:
                                total_ssim_weighted += ssim_val * ssim_mask
                                total_ssim_mask += ssim_mask
                    else:
                        # L1 reflects physical coefficient deviation for roughness/metallic.
                        l1, mask_count, _ = compute_masked_metrics(
                            gen_float,
                            gt_float,
                            metric_type="scalar",
                            debug=args.debug,
                            obj_ids=batch_obj_ids,
                            debug_names=batch_debug_names,
                            debug_dir=debug_dir,
                            debug_tag=channel,
                        )
                        if mask_count > 0:
                            total_mask += mask_count
                            total_l1_weighted += l1 * mask_count

                    if lpips_model and channel == "albedo":
                        mask = extract_alpha_mask(gen_float, gt_float).float()
                        gen_rgb = gen_float[:, :3, ...] * mask
                        gt_rgb = gt_float[:, :3, ...] * mask
                        lpips_val = lpips_model(gen_rgb * 2.0 - 1.0, gt_rgb * 2.0 - 1.0)
                        lpips_scores.append(lpips_val.mean().detach().cpu())

            if channel == "normal":
                # Normal_MeanAngularError is computed over the alpha-intersection mask.
                final_metrics["Normal_MeanAngularError"] = (
                    total_mean_angular_error_weighted / total_mask if total_mask > 0 else float("nan")
                )
                continue

            if channel == "albedo" and metric_flags["psnr"]:
                if total_mask > 0:
                    mse_val = total_mse_weighted / total_mask
                    final_metrics["Albedo_PSNR_Masked"] = (
                        float("inf") if mse_val == 0 else -10.0 * math.log10(mse_val)
                    )
                else:
                    final_metrics["Albedo_PSNR_Masked"] = float("nan")

            if channel == "albedo" and metric_flags["ssim"]:
                final_metrics["Albedo_SSIM_Masked"] = (
                    total_ssim_weighted / total_ssim_mask if total_ssim_mask > 0 else float("nan")
                )

            if channel == "albedo" and lpips_model:
                final_metrics["Albedo_LPIPS_Masked"] = (
                    torch.stack(lpips_scores).mean().item() if lpips_scores else float("nan")
                )

            if channel == "rough":
                final_metrics["Roughness_L1_Masked"] = (
                    total_l1_weighted / total_mask if total_mask > 0 else float("nan")
                )

            if channel == "metal":
                final_metrics["Metallic_L1_Masked"] = (
                    total_l1_weighted / total_mask if total_mask > 0 else float("nan")
                )

    # ===================== Multi-View Consistency Metrics =====================
    # These metrics detect Janus (multi-face) problems and texture flickering
    do_consistency = metric_flags.get("consistency", False)
    
    if do_consistency:
        print("\n=== Computing Multi-View Consistency Metrics ===")
        
        # Load LPIPS model for consistency metrics if not already loaded
        lpips_mod = _lazy_import_lpips()
        consistency_lpips_model = lpips_mod.LPIPS(net="vgg").to(device)
        consistency_lpips_model.eval()
        _clear_cuda_cache()
        
        try:
            # 1. Cross-view consistency metrics (works without depth maps)
            consistency_metrics = compute_multiview_consistency_metrics(
                base_gt_dir=base_gt_dir,
                base_gen_dir=base_gen_dir,
                unlit_subdir=args.unlit_subdir,
                lit_subdir=args.lit_subdir,
                obj_ids=obj_ids,
                device=device,
                lpips_model=consistency_lpips_model,
                num_pairs=args.consistency_pairs,
                channel=args.consistency_channel,
                debug=args.debug,
            )
            
            # Add consistency metrics to final metrics with prefix
            for metric_name, value in consistency_metrics.items():
                final_metrics[f"Consistency/{metric_name}"] = value
            
            print("Cross-view consistency metrics computed.")
            for k, v in consistency_metrics.items():
                print(f"  {k}: {v:.6f}")
            
            # 2. Reprojection-based metrics (optional, requires depth maps and dense views)
            if args.reprojection:
                print("\n--- Computing Reprojection Metrics (--reprojection enabled) ---")
                reproj_metrics = compute_reprojection_metrics(
                    base_gt_dir=base_gt_dir,
                    base_gen_dir=base_gen_dir,
                    unlit_subdir=args.unlit_subdir,
                    obj_ids=obj_ids,
                    device=device,
                    lpips_model=consistency_lpips_model,
                    num_pairs=args.consistency_pairs,
                    debug=args.debug,
                )
                
                if reproj_metrics:
                    for metric_name, value in reproj_metrics.items():
                        final_metrics[f"Consistency/{metric_name}"] = value
                    
                    print("Reprojection metrics computed successfully.")
                    for k, v in reproj_metrics.items():
                        print(f"  {k}: {v:.6f}")
                else:
                    print("Reprojection metrics skipped (no valid results).")
                    print("  Sparse views may have insufficient overlap for reprojection.")
            
        except Exception as e:
            print(f"[ERROR] Failed to compute consistency metrics: {e}")
            traceback.print_exc()
        finally:
            del consistency_lpips_model
            _clear_cuda_cache()

    output_path = args.output or f"metrics_{args.experiment_name}.json"
    save_metrics(final_metrics, output_path)
    print(f"Saved metrics to {output_path}:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
