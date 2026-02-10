#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import trimesh
except Exception:
    trimesh = None

try:
    import open3d as o3d
except Exception:
    o3d = None


@dataclass
class Job:
    uid: str
    mesh_path: str
    source_tsv: str
    row_index: int


def resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def normalize_points(points: np.ndarray) -> np.ndarray:
    center = (points.max(axis=0) + points.min(axis=0)) * 0.5
    points = points - center
    bbox_extent = points.max(axis=0) - points.min(axis=0)
    max_dim = float(np.max(bbox_extent))
    if max_dim > 1e-8:
        points = points / max_dim
    return points


def normalize_normals(normals: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return normals / norms


def obj_parse_vertex_and_normal_indices(face_token: str) -> Tuple[int, int]:
    # OBJ token formats: v, v/vt, v//vn, v/vt/vn
    parts = face_token.split("/")
    vi = int(parts[0]) if parts and parts[0] else 0
    ni = int(parts[2]) if len(parts) >= 3 and parts[2] else 0
    return vi, ni


def to_zero_based(idx: int, size: int) -> int:
    if idx > 0:
        return idx - 1
    if idx < 0:
        return size + idx
    return -1


def obj_to_points_normals(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    normals_src: List[List[float]] = []
    faces_v: List[List[int]] = []
    faces_vn: List[List[int]] = []

    with open(mesh_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            if line.startswith("v "):
                vals = line.strip().split()
                if len(vals) >= 4:
                    vertices.append([float(vals[1]), float(vals[2]), float(vals[3])])
            elif line.startswith("vn "):
                vals = line.strip().split()
                if len(vals) >= 4:
                    normals_src.append([float(vals[1]), float(vals[2]), float(vals[3])])
            elif line.startswith("f "):
                toks = line.strip().split()[1:]
                if len(toks) < 3:
                    continue
                # Fan triangulation for ngons.
                for i in range(1, len(toks) - 1):
                    tri = [toks[0], toks[i], toks[i + 1]]
                    tri_v: List[int] = []
                    tri_vn: List[int] = []
                    for tok in tri:
                        vi, ni = obj_parse_vertex_and_normal_indices(tok)
                        tri_v.append(vi)
                        tri_vn.append(ni)
                    faces_v.append(tri_v)
                    faces_vn.append(tri_vn)

    if not vertices:
        raise ValueError(f"OBJ has no vertices: {mesh_path}")

    v = np.asarray(vertices, dtype=np.float32)
    n = np.zeros_like(v, dtype=np.float32)
    f_v = np.asarray(faces_v, dtype=np.int64) if faces_v else np.zeros((0, 3), dtype=np.int64)
    f_vn = np.asarray(faces_vn, dtype=np.int64) if faces_vn else np.zeros((0, 3), dtype=np.int64)
    n_src = np.asarray(normals_src, dtype=np.float32) if normals_src else np.zeros((0, 3), dtype=np.float32)

    if f_v.shape[0] > 0:
        # Prefer author-provided vertex normals when available.
        use_vn = n_src.shape[0] > 0 and np.any(f_vn != 0)
        if use_vn:
            count = np.zeros((v.shape[0], 1), dtype=np.float32)
            for fi in range(f_v.shape[0]):
                for k in range(3):
                    vi = to_zero_based(int(f_v[fi, k]), v.shape[0])
                    ni = to_zero_based(int(f_vn[fi, k]), n_src.shape[0])
                    if vi < 0 or vi >= v.shape[0] or ni < 0 or ni >= n_src.shape[0]:
                        continue
                    n[vi] += n_src[ni]
                    count[vi, 0] += 1.0
            valid = count[:, 0] > 0
            n[valid] = n[valid] / count[valid]

        # Fill missing normals from geometric face normals.
        missing = np.linalg.norm(n, axis=1) < 1e-8
        if np.any(missing):
            n_geo = np.zeros_like(v, dtype=np.float32)
            for fi in range(f_v.shape[0]):
                i0 = to_zero_based(int(f_v[fi, 0]), v.shape[0])
                i1 = to_zero_based(int(f_v[fi, 1]), v.shape[0])
                i2 = to_zero_based(int(f_v[fi, 2]), v.shape[0])
                if (
                    i0 < 0 or i1 < 0 or i2 < 0 or
                    i0 >= v.shape[0] or i1 >= v.shape[0] or i2 >= v.shape[0]
                ):
                    continue
                p0, p1, p2 = v[i0], v[i1], v[i2]
                fn = np.cross(p1 - p0, p2 - p0)
                fn_norm = np.linalg.norm(fn)
                if fn_norm < 1e-12:
                    continue
                fn = fn / fn_norm
                n_geo[i0] += fn
                n_geo[i1] += fn
                n_geo[i2] += fn
            n[missing] = n_geo[missing]

    nlen = np.linalg.norm(n, axis=1)
    if nlen.max(initial=0.0) < 1e-8:
        n[:, 2] = 1.0
    else:
        # Keep conversion stable even when mesh has isolated/unreferenced vertices.
        zero_mask = nlen < 1e-8
        if np.any(zero_mask):
            n[zero_mask, 2] = 1.0

    return v, n


def load_mesh_as_pointcloud(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    ext = os.path.splitext(mesh_path)[1].lower()

    if ext == ".obj":
        return obj_to_points_normals(mesh_path)

    if trimesh is not None:
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            geoms = [g for g in mesh.geometry.values() if hasattr(g, "vertices") and len(g.vertices) > 0]
            if not geoms:
                raise ValueError(f"Empty mesh scene: {mesh_path}")
            mesh = trimesh.util.concatenate(geoms)
        points = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        return points, normals

    if o3d is not None:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh.is_empty():
            raise ValueError(f"Empty mesh: {mesh_path}")
        mesh.compute_vertex_normals()
        points = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        return points, normals

    raise ImportError(
        "Non-OBJ mesh needs trimesh/open3d. Install one, or provide OBJ mesh paths."
    )


def preprocess_pointcloud(
    uid: str,
    points: np.ndarray,
    normals: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Invalid points for uid={uid}, shape={points.shape}")

    if normals.shape != points.shape:
        normals = np.zeros_like(points, dtype=np.float32)
        normals[:, 2] = 1.0

    points = points.astype(np.float32, copy=False)
    normals = normals.astype(np.float32, copy=False)

    if max_points > 0 and points.shape[0] > max_points:
        seed = int(hashlib.md5(uid.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        ids = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[ids]
        normals = normals[ids]

    points = normalize_points(points)
    normals = normalize_normals(normals)
    return np.ascontiguousarray(points), np.ascontiguousarray(normals)


def load_jobs_from_tsv(tsv_path: str) -> List[Job]:
    jobs: List[Job] = []
    tsv_path = os.path.abspath(tsv_path)
    base_dir = os.path.dirname(tsv_path)

    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_index, row in enumerate(reader, start=2):
            uid = (row.get("obj_id") or row.get("uid") or row.get("id") or "").strip()
            if not uid:
                continue
            mesh_raw = (row.get("mesh") or row.get("mesh_path") or "").strip()
            mesh_path = resolve_path(mesh_raw, base_dir)
            jobs.append(Job(uid=uid, mesh_path=mesh_path, source_tsv=tsv_path, row_index=row_index))
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute pointcloud npz files from mesh paths in TSV."
    )
    parser.add_argument(
        "--tsv",
        nargs="+",
        required=True,
        help="One or more TSV files containing obj_id and mesh/mesh_path columns.",
    )
    parser.add_argument(
        "--output_dir",
        default="../datasets/texverse_pointcloud_npz",
        help="Output directory for <obj_id>.npz files.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=200000,
        help="Maximum points per sample after downsampling. <=0 disables cap.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of unique samples to process. <=0 means all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing npz files.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately on first failed sample.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    jobs_by_uid: Dict[str, Job] = {}
    for tsv in args.tsv:
        tsv = os.path.abspath(tsv)
        if not os.path.isfile(tsv):
            print(f"[ERROR] TSV not found: {tsv}")
            return 1
        jobs = load_jobs_from_tsv(tsv)
        print(f"[INFO] Loaded {len(jobs)} rows from {tsv}")
        for job in jobs:
            prev = jobs_by_uid.get(job.uid)
            if prev is None:
                jobs_by_uid[job.uid] = job
            elif prev.mesh_path != job.mesh_path:
                print(
                    f"[WARN] Duplicate uid with different mesh paths: {job.uid}\n"
                    f"       keep: {prev.mesh_path}\n"
                    f"       skip: {job.mesh_path}"
                )

    jobs = list(jobs_by_uid.values())
    total_all = len(jobs)
    if total_all == 0:
        print("[ERROR] No valid rows found in input TSV(s).")
        return 1

    if args.max_samples > 0:
        jobs = jobs[:args.max_samples]

    total = len(jobs)
    print(f"[INFO] Unique samples: {total_all}")
    if args.max_samples > 0:
        print(f"[INFO] Limit samples: {args.max_samples} -> processing {total}")
    else:
        print("[INFO] Limit samples: all")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Max points: {args.max_points}")

    created = 0
    skipped = 0
    failed = 0
    failures: List[Tuple[str, str, str]] = []

    for idx, job in enumerate(jobs, start=1):
        out_path = os.path.join(output_dir, job.uid + ".npz")
        if os.path.isfile(out_path) and not args.overwrite:
            skipped += 1
            if idx % 200 == 0 or idx == total:
                print(f"[INFO] {idx}/{total} created={created} skipped={skipped} failed={failed}")
            continue

        if not job.mesh_path or not os.path.isfile(job.mesh_path):
            failed += 1
            reason = f"mesh missing: {job.mesh_path}"
            failures.append((job.uid, job.mesh_path, reason))
            print(f"[WARN] {job.uid}: {reason}")
            if args.fail_fast:
                break
            continue

        try:
            points, normals = load_mesh_as_pointcloud(job.mesh_path)
            points, normals = preprocess_pointcloud(job.uid, points, normals, args.max_points)
            np.savez_compressed(out_path, points=points, normals=normals)
            created += 1
        except Exception as e:
            failed += 1
            reason = str(e)
            failures.append((job.uid, job.mesh_path, reason))
            print(f"[WARN] {job.uid}: {reason}")
            if args.fail_fast:
                break

        if idx % 200 == 0 or idx == total:
            print(f"[INFO] {idx}/{total} created={created} skipped={skipped} failed={failed}")

    fail_path = os.path.join(output_dir, "precompute_failures.tsv")
    if failures:
        with open(fail_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["obj_id", "mesh_path", "reason"])
            writer.writerows(failures)
        print(f"[INFO] Wrote failures to: {fail_path}")

    print(
        "[INFO] Done. "
        f"total={total}, created={created}, skipped={skipped}, failed={failed}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
