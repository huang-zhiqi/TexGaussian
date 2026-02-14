import os
import csv
import json
import glob
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from external.clip import tokenize

from core.options import Options
from core.longclip_utils import resolve_longclip_module

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("yes", "true", "t", "y", "1")
    return bool(v)

class CollateBatch:

    def __init__(self):
        pass

    def __call__(self, batch: list):
        assert type(batch) == list

        outputs = {}
        for key in batch[0].keys():
            outputs[key] = [b[key] for b in batch]

            if 'points' in key or 'normals' in key:
                outputs[key] = [torch.from_numpy(a) for a in outputs[key]]
        
            elif 'gaussian' in key:
                outputs[key] = torch.cat(outputs[key], dim = 0)
            
            elif 'uid' in key:
                pass
        
            else:
                outputs[key] = torch.stack(outputs[key])

        return outputs

def collate_func(batch):

    collate_batch = CollateBatch()
    output = collate_batch(batch)

    return output


class TexGaussianDataset(Dataset):

    def __init__(self, opt: Options, training=True):

        self.opt = opt
        self.training = training
        self.use_longclip = _to_bool(self.opt.use_longclip)
        self.use_text = _to_bool(self.opt.use_text)
        self.use_material = _to_bool(self.opt.use_material)
        self.use_normal_head = _to_bool(self.opt.use_normal_head)
        self.caption_field = getattr(self.opt, "caption_field", "caption_short")
        self.longclip_tokenize = None
        self.pointcloud_dir = (getattr(self.opt, "pointcloud_dir", "") or "").strip()
        self.train_image_root = (getattr(self.opt, "image_dir", "") or "").strip()
        self.test_image_root = (getattr(self.opt, "test_image_dir", "") or "").strip()
        if self.train_image_root.startswith("path_to_"):
            self.train_image_root = ""
        if self.test_image_root.startswith("path_to_"):
            self.test_image_root = ""
        if not self.test_image_root:
            self.test_image_root = self.train_image_root
        if self.pointcloud_dir.startswith("path_to_"):
            self.pointcloud_dir = ""
        if not self.pointcloud_dir:
            raise ValueError(
                "pointcloud_dir must be set to the precomputed npz root. "
                "Mesh fallback is disabled."
            )
        if not self.train_image_root:
            raise ValueError("image_dir must be set.")
        if not os.path.isdir(self.train_image_root):
            raise FileNotFoundError(f"image_dir does not exist: {self.train_image_root}")
        if not self.test_image_root:
            raise ValueError("test_image_dir resolved empty; set image_dir or test_image_dir.")
        if not os.path.isdir(self.test_image_root):
            raise FileNotFoundError(f"test_image_dir does not exist: {self.test_image_root}")
        if not os.path.isdir(self.pointcloud_dir):
            raise FileNotFoundError(f"pointcloud_dir does not exist: {self.pointcloud_dir}")

        split_path = opt.trainlist if self.training else opt.testlist
        self.items = self._load_items(split_path)
        if not self.training:
            self.items = self.items[:100]
        self._validate_items_or_raise(self.items)
        if len(self.items) == 0:
            split_name = "train" if self.training else "test"
            raise RuntimeError(f"No {split_name} samples found in split file.")

        if self.use_text and self.use_longclip:
            _, self.longclip_tokenize = resolve_longclip_module()

        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

    def _load_items(self, split_path):
        if not split_path.lower().endswith(".tsv"):
            raise ValueError(
                f"trainlist/testlist must be TSV now, got: {split_path}"
            )
        return self._load_items_from_tsv(split_path)

    def _load_items_from_tsv(self, split_path):
        items = []
        split_dir = os.path.dirname(os.path.abspath(split_path))
        with open(split_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                uid = (row.get("obj_id") or row.get("uid") or row.get("id") or "").strip()
                if not uid:
                    continue
                rough_val = row.get("rough") or row.get("roughness") or ""
                metal_val = row.get("metal") or row.get("metallic") or row.get("matellic") or ""
                item = {
                    "uid": uid,
                    "mesh": _resolve_path(row.get("mesh") or row.get("mesh_path") or "", split_dir),
                    "albedo": _resolve_path(row.get("albedo") or "", split_dir),
                    "rough": _resolve_path(rough_val, split_dir),
                    "metal": _resolve_path(metal_val, split_dir),
                    "normal": _resolve_path(row.get("normal") or "", split_dir),
                    "caption_short": (row.get("caption_short") or "").strip(),
                    "caption_long": (row.get("caption_long") or "").strip(),
                }
                items.append(item)
        return items

    @staticmethod
    def _read_image(path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if image.ndim == 2:
            image = image[:, :, None]
        return torch.from_numpy(image.astype(np.float32) / 255.0)  # [H, W, C]

    @staticmethod
    def _first_existing(candidates):
        seen = set()
        for cand in candidates:
            if not cand:
                continue
            if cand in seen:
                continue
            seen.add(cand)
            if os.path.exists(cand):
                return cand
        return None

    @staticmethod
    def _resize_hwc(image, h, w):
        if image.shape[0] == h and image.shape[1] == w:
            return image
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
        image = image.squeeze(0).permute(1, 2, 0).contiguous()
        return image

    @staticmethod
    def _channel_aliases(channel):
        if channel == "rough":
            return ["rough", "roughness"]
        if channel == "metal":
            return ["metal", "metallic", "matellic"]
        return [channel]

    def _expand_from_spec(self, spec, channel, vid, prefix):
        if not spec:
            return []
        vid03 = f"{vid:03d}"
        results = []
        aliases = self._channel_aliases(channel)

        if any(k in spec for k in ("{vid}", "{vid03}", "{prefix}", "{channel}")):
            for c in aliases:
                try:
                    results.append(spec.format(vid=vid, vid03=vid03, prefix=prefix, channel=c))
                except Exception:
                    pass
            return results

        if os.path.isdir(spec):
            for c in aliases:
                results.extend([
                    os.path.join(spec, f"{vid}_{c}.png"),
                    os.path.join(spec, f"{vid03}_{c}.png"),
                    os.path.join(spec, f"{prefix}_{c}.png"),
                    os.path.join(spec, "unlit", f"{vid}_{c}.png"),
                    os.path.join(spec, "unlit", f"{vid03}_{c}.png"),
                    os.path.join(spec, "unlit", f"{prefix}_{c}.png"),
                ])
            return results

        base_dir = os.path.dirname(spec)
        if base_dir:
            for c in aliases:
                results.extend([
                    os.path.join(base_dir, f"{vid}_{c}.png"),
                    os.path.join(base_dir, f"{vid03}_{c}.png"),
                    os.path.join(base_dir, f"{prefix}_{c}.png"),
                    os.path.join(base_dir, "unlit", f"{vid}_{c}.png"),
                    os.path.join(base_dir, "unlit", f"{vid03}_{c}.png"),
                    os.path.join(base_dir, "unlit", f"{prefix}_{c}.png"),
                ])
        return results

    def _expand_from_image_dir(self, image_dir, channel, vid, prefix):
        aliases = self._channel_aliases(channel)
        vid03 = f"{vid:03d}"
        roots = [os.path.join(image_dir, "unlit"), image_dir]
        id_tokens = [prefix, vid03, str(vid)]
        candidates = []

        for root in roots:
            for token in id_tokens:
                for c in aliases:
                    candidates.extend([
                        os.path.join(root, f"{token}_{c}.png"),
                        os.path.join(root, f"{c}_{token}.png"),
                        os.path.join(root, c, f"{token}.png"),
                    ])
        return candidates

    def _candidate_paths(self, item, image_dir, vid, prefix, channel):
        candidates = []
        # Prefer IMAGE_DIR/{uid} rendered supervision. TSV channel paths are fallback only.
        candidates.extend(self._expand_from_image_dir(image_dir, channel, vid, prefix))
        path_spec = item.get(channel, "")
        candidates.extend(self._expand_from_spec(path_spec, channel, vid, prefix))
        return candidates

    def _camera_source(self, image_dir):
        camera_path = os.path.join(image_dir, 'cameras.npz')
        transforms_path = os.path.join(image_dir, 'transforms.json')

        if os.path.exists(camera_path):
            return "npz", np.load(camera_path)
        if os.path.exists(transforms_path):
            with open(transforms_path, "r", encoding="utf-8") as f:
                transforms = json.load(f)
            frames = transforms.get("frames", [])
            if frames:
                return "transforms", frames
        raise FileNotFoundError(
            f"Missing camera metadata for {image_dir}: "
            "expected cameras.npz or transforms.json."
        )

    @staticmethod
    def _has_camera_metadata(image_dir):
        return (
            os.path.isfile(os.path.join(image_dir, "cameras.npz"))
            or os.path.isfile(os.path.join(image_dir, "transforms.json"))
        )

    def _has_multiview_channel_from_image_dir(self, image_dir, channel):
        aliases = self._channel_aliases(channel)
        roots = [os.path.join(image_dir, "unlit"), image_dir]
        for root in roots:
            for c in aliases:
                patterns = [
                    os.path.join(root, f"*_{c}.png"),
                    os.path.join(root, f"{c}_*.png"),
                    os.path.join(root, c, "*.png"),
                ]
                for pat in patterns:
                    if glob.glob(pat):
                        return True
        return False

    def _is_valid_item(self, item):
        uid = item["uid"]
        image_root = self.train_image_root if self.training else self.test_image_root
        image_dir = os.path.join(image_root, uid)
        pointcloud_path = os.path.join(self.pointcloud_dir, uid + ".npz")

        if not os.path.isfile(pointcloud_path):
            return False, "missing_pointcloud"
        if not self._has_camera_metadata(image_dir):
            return False, "missing_camera_metadata"
        if not self._has_multiview_channel_from_image_dir(image_dir, "albedo"):
            return False, "missing_albedo"
        if self.use_material:
            if not self._has_multiview_channel_from_image_dir(image_dir, "rough"):
                return False, "missing_rough"
            if not self._has_multiview_channel_from_image_dir(image_dir, "metal"):
                return False, "missing_metal"
        if self.use_normal_head:
            if not self._has_multiview_channel_from_image_dir(image_dir, "normal"):
                return False, "missing_normal"
        return True, ""

    def _validate_items_or_raise(self, items):
        split_name = "train" if self.training else "test"
        image_root = self.train_image_root if self.training else self.test_image_root
        for idx, item in enumerate(items):
            ok, reason = self._is_valid_item(item)
            if ok:
                continue
            uid = item["uid"]
            pointcloud_path = os.path.join(self.pointcloud_dir, uid + ".npz")
            image_dir = os.path.join(image_root, uid)
            raise RuntimeError(
                f"Invalid {split_name} sample at index={idx}, uid={uid}, reason={reason}. "
                f"pointcloud={pointcloud_path}, image_dir={image_dir}"
            )

    @staticmethod
    def _normalize_points(points):
        center = (points.max(axis=0) + points.min(axis=0)) * 0.5
        points = points - center
        bbox_extent = points.max(axis=0) - points.min(axis=0)
        max_dim = float(np.max(bbox_extent))
        if max_dim > 1e-8:
            points = points / max_dim
        return points

    @staticmethod
    def _normalize_normals(normals):
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        return normals / norms

    def _sanitize_pointcloud(self, uid, points, normals):
        points = np.asarray(points, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)

        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f"Invalid pointcloud points for uid={uid}: shape={points.shape}")
        if normals.ndim != 2 or normals.shape != points.shape:
            raise ValueError(
                f"Invalid pointcloud normals for uid={uid}: "
                f"points_shape={points.shape}, normals_shape={normals.shape}"
            )
        if not np.isfinite(points).all() or not np.isfinite(normals).all():
            raise ValueError(f"Pointcloud contains NaN/Inf for uid={uid}")

        points = self._normalize_points(points)
        normals = self._normalize_normals(normals)
        return {
            "points": np.ascontiguousarray(points),
            "normals": np.ascontiguousarray(normals),
        }

    def _load_pointcloud(self, item):
        uid = item["uid"]
        pointcloud_path = os.path.join(self.pointcloud_dir, uid + ".npz")
        if not os.path.isfile(pointcloud_path):
            raise FileNotFoundError(
                f"Missing precomputed pointcloud for uid={uid}: {pointcloud_path}"
            )

        with np.load(pointcloud_path) as pointcloud:
            if "points" not in pointcloud or "normals" not in pointcloud:
                raise KeyError(
                    f"Invalid pointcloud npz for uid={uid}: missing 'points' or 'normals' ({pointcloud_path})"
                )
            return self._sanitize_pointcloud(
                uid,
                pointcloud["points"],
                pointcloud["normals"],
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        item = self.items[idx]
        uid = item["uid"]

        image_root = self.train_image_root if self.training else self.test_image_root
        image_dir = os.path.join(image_root, uid)

        if self.opt.gaussian_loss:
            gaussian_path = os.path.join(self.opt.gaussian_dir, uid, 'gaussian.pth')

        results = {}

        results['uid'] = uid

        pointcloud = self._load_pointcloud(item)
        results['points'] = pointcloud['points']
        results['normals'] = pointcloud['normals']
        results['point_mesh_normals'] = pointcloud['normals']

        if self.opt.gaussian_loss:
            gaussian = torch.load(gaussian_path)
            gaussian.requires_grad_(False)
            results['gaussian'] = gaussian

        # load num_views images
        images = []
        masks = []
        normal_maps = []
        if self.use_material:
            material_images = []
        cam_poses = []

        camera_mode, camera_data = self._camera_source(image_dir)
        if camera_mode == "npz":
            total_views = int(camera_data['poses'].shape[0])
            vids = np.random.permutation(total_views).tolist()
        elif camera_mode == "transforms":
            total_views = len(camera_data)
            vids = np.random.permutation(total_views).tolist()

        for vid in vids:
            if camera_mode == "npz":
                prefix = f"{vid:03d}"
                c2w = camera_data['poses'][vid]  # [4, 4]
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
            elif camera_mode == "transforms":
                frame = camera_data[vid]
                prefix = str(frame.get("file_prefix", f"{vid:03d}"))
                if "camera_to_world" in frame:
                    c2w = np.array(frame["camera_to_world"], dtype=np.float32)
                else:
                    w2c = np.array(frame["world_to_camera"], dtype=np.float32)
                    c2w = np.linalg.inv(w2c)
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)

            try:
                # Albedo supervision (RGB + alpha mask if available).
                albedo_path = self._first_existing(self._candidate_paths(item, image_dir, vid, prefix, "albedo"))
                if albedo_path is None:
                    continue
                albedo = self._read_image(albedo_path)  # [H, W, C]
                if albedo.shape[2] >= 4:
                    mask = albedo[:, :, 3:4]
                else:
                    mask = torch.ones_like(albedo[:, :, :1])
                image = albedo[:, :, :3]
                if image.shape[2] == 1:
                    image = image.repeat(1, 1, 3)
                image = image.permute(2, 0, 1)  # [3, H, W]
                image = image[[2, 1, 0]].contiguous()  # bgr -> rgb
                mask = mask.permute(2, 0, 1).contiguous()  # [1, H, W]
                image = image * mask + (1 - mask)  # white bg

                h, w = image.shape[1], image.shape[2]

                if self.use_material:
                    rough_path = self._first_existing(self._candidate_paths(item, image_dir, vid, prefix, "rough"))
                    metal_path = self._first_existing(self._candidate_paths(item, image_dir, vid, prefix, "metal"))
                    if rough_path is None or metal_path is None:
                        continue
                    rough = self._read_image(rough_path)
                    metal = self._read_image(metal_path)
                    rough = self._resize_hwc(rough, h, w)
                    metal = self._resize_hwc(metal, h, w)
                    rough = rough[:, :, :1].permute(2, 0, 1).contiguous()  # [1, H, W]
                    metal = metal[:, :, :1].permute(2, 0, 1).contiguous()  # [1, H, W]
                    rough = rough * mask + (1 - mask)
                    metal = metal * mask + (1 - mask)

                if self.use_normal_head:
                    normal_path = self._first_existing(self._candidate_paths(item, image_dir, vid, prefix, "normal"))
                    if normal_path is None:
                        continue
                    normal_map = self._read_image(normal_path)
                    normal_map = self._resize_hwc(normal_map, h, w)
                    if normal_map.shape[2] >= 3:
                        normal_map = normal_map[:, :, :3]
                    else:
                        normal_map = normal_map.repeat(1, 1, 3)
                    normal_map = normal_map.permute(2, 0, 1)  # [3, H, W]
                    normal_map = normal_map[[2, 1, 0]].contiguous()  # bgr -> rgb
            except Exception:
                continue

            images.append(image)
            if self.use_material:
                material_images.append(torch.cat([rough, metal], dim=0))
            if self.use_normal_head:
                normal_maps.append(normal_map)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            if len(images) == self.opt.num_views:
                break

        if len(images) == 0:
            raise RuntimeError(
                f"Failed to load supervision for uid={uid}. "
                f"Checked image_dir={image_dir} and TSV paths."
            )
        if len(images) < self.opt.num_views:
            pad_n = self.opt.num_views - len(images)
            for _ in range(pad_n):
                images.append(images[-1].clone())
                masks.append(masks[-1].clone())
                cam_poses.append(cam_poses[-1].clone())
                if self.use_material:
                    material_images.append(material_images[-1].clone())
                if self.use_normal_head:
                    normal_maps.append(normal_maps[-1].clone())

        images = torch.stack(images, dim=0) # [V, C, H, W]

        if self.use_material:
            material_images = torch.stack(material_images, dim=0)

        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        if self.use_material:
            material_images = F.interpolate(material_images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
            results['rough_images_output'] = material_images[:, :1, ...]
            results['metallic_images_output'] = material_images[:, 1:2, ...]

        if self.use_normal_head:
            normal_maps = torch.stack(normal_maps, dim=0)
            results['gt_normal_map'] = F.interpolate(normal_maps, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        if self.use_text:
            text = ""
            if self.caption_field in item and item.get(self.caption_field):
                text = item.get(self.caption_field)
            elif item.get("caption_long"):
                text = item.get("caption_long")
            elif item.get("caption_short"):
                text = item.get("caption_short")

            if self.use_longclip:
                token = self.longclip_tokenize(
                    text,
                    context_length=self.opt.longclip_context_length,
                    truncate=True,
                )
            else:
                token = tokenize(text)
            token = token.squeeze()
            results['token'] = token

        return results
