#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse the same mesh-loading and preprocessing logic as precompute script.
from precompute_pointclouds import (
    load_jobs_from_tsv,
    load_mesh_as_pointcloud,
    preprocess_pointcloud,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class ValidateTarget:
    uid: str
    mesh_path: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate precomputed pointcloud npz files."
    )
    parser.add_argument(
        "--pointcloud_dir",
        default="../datasets/texverse_pointcloud_npz",
        help="Directory containing <obj_id>.npz pointcloud files.",
    )
    parser.add_argument(
        "--tsv",
        nargs="*",
        default=[],
        help="Optional TSV list (obj_id + mesh path) to define validation set and mesh rebuild checks.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of unique samples to validate. <=0 means all.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=200000,
        help="Max points used in rebuild comparison path. Must match precompute setting.",
    )
    parser.add_argument(
        "--check_mesh_rebuild",
        action="store_true",
        help="Rebuild from mesh and compare with saved npz values.",
    )
    parser.add_argument(
        "--strict_float32",
        action="store_true",
        help="Require points/normals dtype to be exactly float32.",
    )
    parser.add_argument(
        "--radius_min",
        type=float,
        default=0.05,
        help="Lower bound for max radius after normalization (sanity check only).",
    )
    parser.add_argument(
        "--radius_max",
        type=float,
        default=1.10,
        help="Upper bound for max radius after normalization (sanity check only).",
    )
    parser.add_argument(
        "--extent_max_min",
        type=float,
        default=0.95,
        help="Lower bound for max bbox extent after normalization.",
    )
    parser.add_argument(
        "--extent_max_max",
        type=float,
        default=1.05,
        help="Upper bound for max bbox extent after normalization.",
    )
    parser.add_argument(
        "--center_tol",
        type=float,
        default=0.10,
        help="Tolerance for bbox-center abs max component, where bbox-center=(min+max)/2.",
    )
    parser.add_argument(
        "--centroid_tol",
        type=float,
        default=-1.0,
        help="Optional tolerance for abs(mean(points)) max component. <=0 disables this check.",
    )
    parser.add_argument(
        "--normal_len_min",
        type=float,
        default=0.50,
        help="Lower bound for normal vector length checks.",
    )
    parser.add_argument(
        "--normal_len_max",
        type=float,
        default=1.50,
        help="Upper bound for normal vector length checks.",
    )
    parser.add_argument(
        "--compare_tol",
        type=float,
        default=1e-5,
        help="Max abs diff tolerance for rebuild comparison.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop on first failed sample.",
    )
    parser.add_argument(
        "--export_viz",
        action="store_true",
        help="Export visualization images for pointcloud-vs-mesh comparison.",
    )
    parser.add_argument(
        "--viz_mode",
        choices=["failures", "all"],
        default="failures",
        help="Which samples to export visualization for.",
    )
    parser.add_argument(
        "--viz_limit",
        type=int,
        default=50,
        help="Maximum number of visualization images to export.",
    )
    parser.add_argument(
        "--viz_max_points",
        type=int,
        default=20000,
        help="Maximum points used per cloud in visualization.",
    )
    parser.add_argument(
        "--viz_dir",
        default="",
        help="Visualization output directory. Defaults to <pointcloud_dir>/validate_viz.",
    )
    return parser.parse_args()


def collect_targets(pointcloud_dir: str, tsv_list: List[str], max_samples: int) -> List[ValidateTarget]:
    targets: List[ValidateTarget] = []

    if tsv_list:
        by_uid: Dict[str, ValidateTarget] = {}
        for tsv in tsv_list:
            tsv_abs = os.path.abspath(tsv)
            if not os.path.isfile(tsv_abs):
                raise FileNotFoundError(f"TSV not found: {tsv_abs}")
            jobs = load_jobs_from_tsv(tsv_abs)
            print(f"[INFO] Loaded {len(jobs)} rows from {tsv_abs}")
            for job in jobs:
                if job.uid not in by_uid:
                    by_uid[job.uid] = ValidateTarget(uid=job.uid, mesh_path=job.mesh_path)
        targets = list(by_uid.values())
    else:
        npz_files = sorted(glob.glob(os.path.join(pointcloud_dir, "*.npz")))
        targets = [ValidateTarget(uid=os.path.basename(p)[:-4], mesh_path="") for p in npz_files]

    if max_samples > 0:
        targets = targets[:max_samples]
    return targets


def validate_npz_structure(
    uid: str,
    npz_path: str,
    strict_float32: bool,
    radius_min: float,
    radius_max: float,
    extent_max_min: float,
    extent_max_max: float,
    center_tol: float,
    centroid_tol: float,
    normal_len_min: float,
    normal_len_max: float,
) -> Tuple[bool, Dict[str, float], List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    reasons: List[str] = []
    stats: Dict[str, float] = {}

    if not os.path.isfile(npz_path):
        return False, stats, [f"missing npz: {npz_path}"], None, None

    try:
        data = np.load(npz_path)
    except Exception as e:
        return False, stats, [f"load npz failed: {e}"], None, None

    if "points" not in data or "normals" not in data:
        return False, stats, ["missing keys (points/normals)"], None, None

    points = data["points"]
    normals = data["normals"]

    if strict_float32:
        if points.dtype != np.float32:
            reasons.append(f"points dtype={points.dtype} != float32")
        if normals.dtype != np.float32:
            reasons.append(f"normals dtype={normals.dtype} != float32")

    if points.ndim != 2 or points.shape[1] != 3:
        reasons.append(f"invalid points shape={points.shape}")
        return False, stats, reasons, None, None
    if normals.shape != points.shape:
        reasons.append(f"normals shape={normals.shape} != points shape={points.shape}")
        return False, stats, reasons, None, None

    if not np.isfinite(points).all():
        reasons.append("points contains NaN/Inf")
    if not np.isfinite(normals).all():
        reasons.append("normals contains NaN/Inf")

    radius = float(np.linalg.norm(points, axis=1).max())
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    bbox_extent = pmax - pmin
    bbox_extent_max = float(np.max(bbox_extent))
    bbox_center = float(np.abs((pmax + pmin) * 0.5).max())
    centroid = float(np.abs(points.mean(axis=0)).max())
    nlen = np.linalg.norm(normals, axis=1)
    nlen_min = float(nlen.min())
    nlen_max = float(nlen.max())
    nlen_mean = float(nlen.mean())

    stats["num_points"] = float(points.shape[0])
    stats["radius"] = radius
    stats["bbox_extent_max"] = bbox_extent_max
    stats["bbox_center_abs_max"] = bbox_center
    stats["centroid_abs_max"] = centroid
    stats["normal_len_min"] = nlen_min
    stats["normal_len_max"] = nlen_max
    stats["normal_len_mean"] = nlen_mean

    if radius < radius_min or radius > radius_max:
        reasons.append(f"radius={radius:.6f} out of [{radius_min}, {radius_max}]")
    if bbox_extent_max < extent_max_min or bbox_extent_max > extent_max_max:
        reasons.append(
            f"bbox_extent_max={bbox_extent_max:.6f} out of [{extent_max_min}, {extent_max_max}]"
        )
    if bbox_center > center_tol:
        reasons.append(f"bbox_center_abs_max={bbox_center:.6f} > {center_tol}")
    if centroid_tol > 0 and centroid > centroid_tol:
        reasons.append(f"centroid_abs_max={centroid:.6f} > {centroid_tol}")
    if nlen_min < normal_len_min or nlen_max > normal_len_max:
        reasons.append(
            f"normal_len_range=({nlen_min:.6f},{nlen_max:.6f}) out of [{normal_len_min}, {normal_len_max}]"
        )

    return len(reasons) == 0, stats, reasons, points, normals


def validate_mesh_rebuild(
    uid: str,
    mesh_path: str,
    points_saved: np.ndarray,
    normals_saved: np.ndarray,
    max_points: int,
    compare_tol: float,
) -> Tuple[bool, List[str], Dict[str, float]]:
    reasons: List[str] = []
    stats: Dict[str, float] = {}

    if not mesh_path:
        return False, ["mesh path missing for rebuild check"], stats
    if not os.path.isfile(mesh_path):
        return False, [f"mesh not found: {mesh_path}"], stats

    try:
        points_ref, normals_ref = load_mesh_as_pointcloud(mesh_path)
        points_ref, normals_ref = preprocess_pointcloud(uid, points_ref, normals_ref, max_points)
    except Exception as e:
        return False, [f"rebuild failed: {e}"], stats

    if points_ref.shape != points_saved.shape or normals_ref.shape != normals_saved.shape:
        reasons.append(
            f"rebuild shape mismatch ref(points={points_ref.shape}, normals={normals_ref.shape}) "
            f"saved(points={points_saved.shape}, normals={normals_saved.shape})"
        )
        return False, reasons, stats

    diff_points = float(np.max(np.abs(points_ref - points_saved)))
    diff_normals = float(np.max(np.abs(normals_ref - normals_saved)))
    stats["rebuild_diff_points"] = diff_points
    stats["rebuild_diff_normals"] = diff_normals

    if diff_points > compare_tol:
        reasons.append(f"rebuild_diff_points={diff_points:.6e} > {compare_tol}")
    if diff_normals > compare_tol:
        reasons.append(f"rebuild_diff_normals={diff_normals:.6e} > {compare_tol}")

    return len(reasons) == 0, reasons, stats


def _subsample_for_viz(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points or max_points <= 0:
        return points
    ids = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
    return points[ids]


def _set_axes_equal(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) * 0.5
    half = float(np.max(maxs - mins) * 0.5)
    if half < 1e-8:
        half = 1.0
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)


def export_visualization(
    uid: str,
    npz_points: np.ndarray,
    npz_normals: np.ndarray,
    mesh_path: str,
    viz_dir: str,
    viz_max_points: int,
) -> Tuple[bool, str]:
    if plt is None:
        return False, "matplotlib not available"

    mesh_points = None
    mesh_reason = ""
    if mesh_path and os.path.isfile(mesh_path):
        try:
            mesh_points, mesh_normals = load_mesh_as_pointcloud(mesh_path)
            mesh_points, _ = preprocess_pointcloud(
                uid=uid,
                points=mesh_points,
                normals=mesh_normals,
                max_points=-1,
            )
        except Exception as e:
            mesh_reason = str(e)
    else:
        mesh_reason = "mesh missing"

    os.makedirs(viz_dir, exist_ok=True)
    out_path = os.path.join(viz_dir, f"{uid}.png")

    pts = _subsample_for_viz(npz_points, viz_max_points)
    nrm = _subsample_for_viz(npz_normals, viz_max_points)
    colors = np.clip((nrm + 1.0) * 0.5, 0.0, 1.0)

    if mesh_points is not None:
        mesh_pts = _subsample_for_viz(mesh_points, viz_max_points)
        all_pts = np.concatenate([pts, mesh_pts], axis=0)
    else:
        mesh_pts = None
        all_pts = pts

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    if mesh_pts is not None:
        ax1.scatter(mesh_pts[:, 0], mesh_pts[:, 1], mesh_pts[:, 2], s=0.3, c="gray", alpha=0.8)
        ax1.set_title("Mesh points")
    else:
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, c="gray", alpha=0.8)
        ax1.set_title(f"Mesh unavailable ({mesh_reason})")

    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, c=colors, alpha=0.9)
    ax2.set_title("NPZ points (normal color)")

    if mesh_pts is not None:
        ax3.scatter(mesh_pts[:, 0], mesh_pts[:, 1], mesh_pts[:, 2], s=0.2, c="gray", alpha=0.25)
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, c=colors, alpha=0.85)
    ax3.set_title("Overlay")

    for ax in (ax1, ax2, ax3):
        _set_axes_equal(ax, all_pts)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    fig.suptitle(uid)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True, out_path


def main() -> int:
    args = parse_args()
    pointcloud_dir = os.path.abspath(args.pointcloud_dir)
    if not os.path.isdir(pointcloud_dir):
        print(f"[ERROR] pointcloud_dir not found: {pointcloud_dir}")
        return 1

    tsv_list = [os.path.abspath(x) for x in args.tsv]
    targets = collect_targets(pointcloud_dir, tsv_list, args.max_samples)
    if not targets:
        print("[ERROR] No validation targets found.")
        return 1

    print(f"[INFO] Pointcloud dir: {pointcloud_dir}")
    print(f"[INFO] Targets: {len(targets)}")
    print(f"[INFO] check_mesh_rebuild: {args.check_mesh_rebuild}")
    print(f"[INFO] strict_float32: {args.strict_float32}")
    print(f"[INFO] export_viz: {args.export_viz} mode={args.viz_mode} limit={args.viz_limit}")
    if args.export_viz and plt is None:
        print("[WARN] matplotlib not available; visualization export disabled.")
        args.export_viz = False

    passed = 0
    failed = 0
    failures: List[Tuple[str, str, str, str]] = []
    viz_count = 0
    viz_dir = args.viz_dir if args.viz_dir else os.path.join(pointcloud_dir, "validate_viz")

    for idx, target in enumerate(targets, start=1):
        uid = target.uid
        npz_path = os.path.join(pointcloud_dir, uid + ".npz")

        ok_struct, stats, reasons, points_saved, normals_saved = validate_npz_structure(
            uid=uid,
            npz_path=npz_path,
            strict_float32=args.strict_float32,
            radius_min=args.radius_min,
            radius_max=args.radius_max,
            extent_max_min=args.extent_max_min,
            extent_max_max=args.extent_max_max,
            center_tol=args.center_tol,
            centroid_tol=args.centroid_tol,
            normal_len_min=args.normal_len_min,
            normal_len_max=args.normal_len_max,
        )

        ok_rebuild = True
        rebuild_reasons: List[str] = []
        rebuild_stats: Dict[str, float] = {}
        if args.check_mesh_rebuild and points_saved is not None and normals_saved is not None:
            ok_rebuild, rebuild_reasons, rebuild_stats = validate_mesh_rebuild(
                uid=uid,
                mesh_path=target.mesh_path,
                points_saved=points_saved,
                normals_saved=normals_saved,
                max_points=args.max_points,
                compare_tol=args.compare_tol,
            )

        ok = ok_struct and ok_rebuild

        do_viz = (
            args.export_viz
            and viz_count < args.viz_limit
            and (
                args.viz_mode == "all"
                or (args.viz_mode == "failures" and not ok)
            )
            and points_saved is not None
            and normals_saved is not None
        )
        if do_viz:
            viz_ok, viz_info = export_visualization(
                uid=uid,
                npz_points=points_saved,
                npz_normals=normals_saved,
                mesh_path=target.mesh_path,
                viz_dir=viz_dir,
                viz_max_points=args.viz_max_points,
            )
            viz_count += 1
            if idx % 20 == 0 or idx == len(targets):
                if viz_ok:
                    print(f"[INFO] viz saved: {viz_info}")
                else:
                    print(f"[WARN] viz failed for {uid}: {viz_info}")

        if ok:
            passed += 1
        else:
            failed += 1
            reason_list = reasons + rebuild_reasons
            reason_msg = " | ".join(reason_list) if reason_list else "unknown failure"
            failures.append((uid, npz_path, target.mesh_path, reason_msg))
            if args.fail_fast:
                print(f"[WARN] {uid}: {reason_msg}")
                break

        if idx % 100 == 0 or idx == len(targets):
            print(
                f"[INFO] {idx}/{len(targets)} passed={passed} failed={failed} "
                f"radius={stats.get('radius', float('nan')):.4f} "
                f"extent_max={stats.get('bbox_extent_max', float('nan')):.4f} "
                f"bbox_center={stats.get('bbox_center_abs_max', float('nan')):.4f} "
                f"centroid={stats.get('centroid_abs_max', float('nan')):.4f} "
                f"nlen_mean={stats.get('normal_len_mean', float('nan')):.4f} "
                f"rebuild_dp={rebuild_stats.get('rebuild_diff_points', float('nan')):.2e} "
                f"rebuild_dn={rebuild_stats.get('rebuild_diff_normals', float('nan')):.2e}"
            )

    fail_path = os.path.join(pointcloud_dir, "validate_failures.tsv")
    with open(fail_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["obj_id", "npz_path", "mesh_path", "reason"])
        writer.writerows(failures)
    if failures:
        print(f"[INFO] Wrote failures: {fail_path}")
    else:
        print(f"[INFO] No failures. Cleared failure file: {fail_path}")

    if args.export_viz:
        print(f"[INFO] Visualization output: {viz_dir} (count={viz_count})")
    print(f"[INFO] Validation done. total={passed + failed}, passed={passed}, failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
