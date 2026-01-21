#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
render_gen_aligned.py

Render generated meshes/textures using the exact camera poses saved from GT renders
(`transforms.json`). This renders both lit PBR previews and unlit material channels
for aligned evaluation.

Supports multi-GPU parallel rendering via subprocess workers.
"""

import argparse
import contextlib
import csv
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Check if bpy module is available
try:
    import bpy
    from mathutils import Matrix
    HAS_BPY = True
except ImportError:
    HAS_BPY = False
    bpy = None
    Matrix = None

PASS_CONFIG = [
    ("albedo", "sRGB"),
    ("rough", "Non-Color"),
    ("metal", "Non-Color"),
    ("normal", "Non-Color"),
]


# ===================== 基础工具 =====================
def log(msg: str) -> None:
    print(f"[Render Gen] {msg}")


@contextlib.contextmanager
def suppress_render_output() -> Any:
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(devnull)
        os.close(old_stdout)


def resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def normalize_hdri_args(hdri_arg: Any) -> List[str]:
    if not hdri_arg:
        return []
    if isinstance(hdri_arg, str):
        raw = [hdri_arg]
    else:
        raw = list(hdri_arg)
    paths: List[str] = []
    for item in raw:
        if not item:
            continue
        parts = [p.strip() for p in item.split(",") if p.strip()]
        paths.extend(parts)
    return paths


def sanitize_hdri_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", base)
    return base.strip("_") or "hdri"


def build_hdri_entries(hdri_arg: Any) -> List[Dict[str, str]]:
    paths = normalize_hdri_args(hdri_arg)
    entries: List[Dict[str, str]] = []
    seen_paths = set()
    name_counts: Dict[str, int] = {}
    for idx, path in enumerate(paths):
        abs_path = os.path.abspath(path)
        if abs_path in seen_paths:
            continue
        seen_paths.add(abs_path)
        base = sanitize_hdri_name(abs_path)
        if not base:
            base = f"hdri_{idx}"
        count = name_counts.get(base, 0)
        name = base if count == 0 else f"{base}_{count + 1}"
        name_counts[base] = count + 1
        entries.append({"path": abs_path, "name": name})
    return entries


def find_normal_path(explicit_normal: str, fallback_from: str) -> str:
    _ = fallback_from
    if explicit_normal and os.path.exists(explicit_normal):
        return explicit_normal
    return ""


def set_color_management(scene: bpy.types.Scene) -> None:
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.sequencer_colorspace_settings.name = "sRGB"


def reset_scene(width: int, height: int, samples: int) -> bpy.types.Scene:
    bpy.ops.wm.read_homefile(use_empty=True)
    sc = bpy.context.scene

    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.device = "GPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
    except Exception:
        sc.cycles.device = "CPU"

    sc.render.resolution_x = int(width)
    sc.render.resolution_y = int(height)
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.dither_intensity = 0.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "8"

    set_color_management(sc)

    # 清空灯光 / 黑色背景
    for obj in list(sc.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    world = sc.world or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    out = nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    return sc


def setup_background(scene: bpy.types.Scene, mode: str, hdri_path: str = "", strength: float = 1.0) -> None:
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputWorld")

    has_hdri = hdri_path and os.path.exists(hdri_path)

    if has_hdri:
        # 1. Create the HDRI lighting node (always needed for lighting)
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri_path)
        bg_light = nodes.new("ShaderNodeBackground")
        bg_light.inputs["Strength"].default_value = strength
        links.new(env.outputs["Color"], bg_light.inputs["Color"])

        if mode == "hdri":
            scene.render.film_transparent = False
            links.new(bg_light.outputs["Background"], out.inputs["Surface"])
        elif mode == "transparent":
            scene.render.film_transparent = True
            links.new(bg_light.outputs["Background"], out.inputs["Surface"])
        else:
            # black or white
            scene.render.film_transparent = False
            
            mix = nodes.new("ShaderNodeMixShader")
            light_path = nodes.new("ShaderNodeLightPath")
            bg_cam = nodes.new("ShaderNodeBackground")
            
            if mode == "white":
                bg_cam.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
                bg_cam.inputs["Strength"].default_value = 1.0
            else: # black
                bg_cam.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
                bg_cam.inputs["Strength"].default_value = 0.0
            
            links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
            links.new(bg_light.outputs["Background"], mix.inputs[1])
            links.new(bg_cam.outputs["Background"], mix.inputs[2])
            links.new(mix.outputs["Shader"], out.inputs["Surface"])
            
    else:
        # No HDRI provided (e.g. unlit mode, or user didn't provide one)
        bg = nodes.new("ShaderNodeBackground")
        if mode == "white":
            scene.render.film_transparent = False
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs["Strength"].default_value = 1.0
        elif mode == "black":
            scene.render.film_transparent = False
            bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            bg.inputs["Strength"].default_value = 0.0
        else: # transparent or fallback
            scene.render.film_transparent = True
            bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            bg.inputs["Strength"].default_value = 0.0
            
        links.new(bg.outputs["Background"], out.inputs["Surface"])


def import_and_normalize(mesh_path: str) -> bpy.types.Object:
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".obj":
        try:
            bpy.ops.import_scene.obj(filepath=mesh_path, use_image_search=False, use_split_objects=False)
        except Exception:
            bpy.ops.wm.obj_import(filepath=mesh_path)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    else:
        return None

    imported = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not imported:
        return None

    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported:
        obj.select_set(True)
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.context.view_layer.objects.active = imported[0]
    if len(imported) > 1:
        bpy.ops.object.join()

    mesh_obj = bpy.context.view_layer.objects.active
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    mesh_obj.location = (0, 0, 0)
    bpy.context.view_layer.update()

    max_dim = max(mesh_obj.dimensions)
    if max_dim > 0:
        scale = 1.0 / max_dim
        mesh_obj.scale = (scale, scale, scale)
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.transform_apply(scale=True)

    bpy.ops.object.shade_smooth()
    return mesh_obj


def build_emission_material(name: str, image_path: str, colorspace: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image_path)
    tex.image.colorspace_settings.name = colorspace
    tex.interpolation = "Smart"

    emis = nodes.new("ShaderNodeEmission")
    emis.inputs["Strength"].default_value = 1.0

    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(tex.outputs["Color"], emis.inputs["Color"])
    links.new(emis.outputs["Emission"], out.inputs["Surface"])
    return mat


def build_world_normal_material(name: str, image_path: str) -> bpy.types.Material:
    """Builds an emission material that outputs world-space normals remapped to [0, 1]."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image_path)
    tex.image.colorspace_settings.name = "Non-Color"
    tex.interpolation = "Smart"

    nrm = nodes.new("ShaderNodeNormalMap")
    nrm.space = "TANGENT"

    vec = nodes.new("ShaderNodeVectorMath")
    vec.operation = "MULTIPLY_ADD"
    vec.inputs[1].default_value = (0.5, 0.5, 0.5)
    vec.inputs[2].default_value = (0.5, 0.5, 0.5)

    emis = nodes.new("ShaderNodeEmission")
    emis.inputs["Strength"].default_value = 1.0

    out = nodes.new("ShaderNodeOutputMaterial")

    links.new(tex.outputs["Color"], nrm.inputs["Color"])
    links.new(nrm.outputs["Normal"], vec.inputs[0])
    links.new(vec.outputs["Vector"], emis.inputs["Color"])
    links.new(emis.outputs["Emission"], out.inputs["Surface"])
    return mat


def build_pbr_material(albedo: str, rough: str, metal: str, normal: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name="PBR_MATERIAL")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def add_tex(path: str, colorspace: str, loc_y: int):
        node = nodes.new("ShaderNodeTexImage")
        node.location = (-400, loc_y)
        node.image = bpy.data.images.load(path)
        node.image.colorspace_settings.name = colorspace
        node.interpolation = "Smart"
        return node

    if albedo:
        alb = add_tex(albedo, "sRGB", 200)
        links.new(alb.outputs["Color"], bsdf.inputs["Base Color"])
    if rough:
        rgh = add_tex(rough, "Non-Color", 0)
        links.new(rgh.outputs["Color"], bsdf.inputs["Roughness"])
    if metal:
        mtl = add_tex(metal, "Non-Color", -200)
        links.new(mtl.outputs["Color"], bsdf.inputs["Metallic"])

    if normal:
        nrm_tex = add_tex(normal, "Non-Color", -400)
        nrm_node = nodes.new("ShaderNodeNormalMap")
        nrm_node.location = (-200, -400)
        links.new(nrm_tex.outputs["Color"], nrm_node.inputs["Color"])
        links.new(nrm_node.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def build_geometry_normal_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    # Fallback for missing normal map: output world-space geometry normals in [0, 1].
    geo = nodes.new("ShaderNodeNewGeometry")
    vec = nodes.new("ShaderNodeVectorMath")
    vec.operation = "MULTIPLY_ADD"
    vec.inputs[1].default_value = (0.5, 0.5, 0.5)
    vec.inputs[2].default_value = (0.5, 0.5, 0.5)

    emis = nodes.new("ShaderNodeEmission")
    emis.inputs["Strength"].default_value = 1.0
    
    out = nodes.new("ShaderNodeOutputMaterial")
    
    links.new(geo.outputs["Normal"], vec.inputs[0])
    links.new(vec.outputs["Vector"], emis.inputs["Color"])
    links.new(emis.outputs["Emission"], out.inputs["Surface"])
    return mat


def compute_intrinsics(cam_obj: bpy.types.Object, scene: bpy.types.Scene) -> Dict[str, float]:
    cam = cam_obj.data
    render = scene.render
    scale = render.resolution_percentage / 100.0
    res_x = render.resolution_x * scale
    res_y = render.resolution_y * scale
    aspect_ratio = render.pixel_aspect_x / render.pixel_aspect_y

    sensor_fit = cam.sensor_fit
    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if res_x >= res_y else "VERTICAL"

    if sensor_fit == "VERTICAL":
        s_u = res_x / (cam.sensor_height * aspect_ratio)
        s_v = res_y / cam.sensor_height
    else:
        s_u = res_x / cam.sensor_width
        s_v = res_y / (cam.sensor_width / aspect_ratio)

    fx = cam.lens * s_u
    fy = cam.lens * s_v
    w = int(round(res_x))
    h = int(round(res_y))
    cx = w * 0.5
    cy = h * 0.5
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": w, "h": h}


def create_camera(scene: bpy.types.Scene, focal_length: float) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = focal_length
    cam_data.sensor_fit = "AUTO"
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def clear_data_blocks() -> None:
    for sc in list(bpy.data.scenes):
        sc.world = None
    for world in list(bpy.data.worlds):
        if world.use_nodes and world.node_tree:
            world.node_tree.links.clear()
            world.node_tree.nodes.clear()
        bpy.data.worlds.remove(world, do_unlink=True)

    for block in list(bpy.data.objects):
        bpy.data.objects.remove(block, do_unlink=True)
    for block in list(bpy.data.cameras):
        bpy.data.cameras.remove(block)
    for block in list(bpy.data.lights):
        bpy.data.lights.remove(block)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block)
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)

    for img in list(bpy.data.images):
        if img.name == "Render Result":
            continue
        img.user_clear()
        try:
            bpy.data.images.remove(img)
        except RuntimeError:
            pass


# ===================== 渲染逻辑 =====================
def load_transforms(transform_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    with open(transform_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    intrinsics = data.get("intrinsics") or {}
    frames = data.get("frames") or []
    meta = data.get("meta") or {}
    return intrinsics, frames, meta


def frame_basename(frame: Dict[str, Any], idx: int) -> str:
    if frame.get("file_prefix"):
        return str(frame["file_prefix"])
    if frame.get("file_name"):
        base = os.path.splitext(os.path.basename(frame["file_name"]))[0]
        if base.endswith("_lit"):
            return base[:-4]
        return base
    return f"{idx:03d}"


def to_matrix(w2c_raw: Any) -> Matrix:
    mat = Matrix(w2c_raw)
    if len(mat) != 4 or len(mat[0]) != 4:
        raise ValueError("world_to_camera must be 4x4")
    return mat


def validate_intrinsics(cam: bpy.types.Object, scene: bpy.types.Scene, target: Dict[str, Any], oid: str) -> None:
    if not target:
        return
    actual = compute_intrinsics(cam, scene)
    keys = ["fx", "fy", "cx", "cy", "w", "h"]
    diffs = []
    for k in keys:
        if k in target:
            diffs.append(abs(actual[k] - float(target[k])))
    if diffs and max(diffs) > 1e-2:
        log(f"{oid}: intrinsics drift detected (max diff {max(diffs):.4f})")


def render_object(row: Dict[str, str], args: argparse.Namespace) -> bool:
    oid = row.get("obj_id", "unknown")
    mesh_path = row.get("mesh")
    albedo = row.get("albedo")
    rough = row.get("rough") or row.get("roughness")
    metal = row.get("metal") or row.get("metallic")
    normal = find_normal_path(row.get("normal"), albedo or mesh_path)

    transform_path = row.get("transforms")
    if not transform_path:
        if args.transforms_subdir:
            legacy_path = os.path.join(args.gt_root, args.transforms_subdir, oid, "transforms.json")
            nested_path = os.path.join(args.gt_root, oid, args.transforms_subdir, "transforms.json")
            transform_path = legacy_path if os.path.exists(legacy_path) else nested_path
        else:
            transform_path = os.path.join(args.gt_root, oid, "transforms.json")

    if not mesh_path or not os.path.exists(mesh_path):
        log(f"{oid}: mesh missing or not found.")
        return False
    if not os.path.exists(transform_path):
        log(f"{oid}: transforms.json not found at {transform_path}")
        return False

    intrinsics, frames, meta = load_transforms(transform_path)
    if not frames:
        log(f"{oid}: no frames in transforms.json, skip.")
        return False
    if "w" not in intrinsics or "h" not in intrinsics:
        log(f"{oid}: transforms missing resolution.")
        return False

    width = int(intrinsics["w"])
    height = int(intrinsics["h"])
    lens_mm = meta.get("focal_length_mm")
    if lens_mm is None and intrinsics.get("fx"):
        sensor_width = 36.0
        lens_mm = float(intrinsics["fx"]) * sensor_width / float(width)
    lens_mm = float(lens_mm) if lens_mm is not None else args.fallback_focal

    obj_root = os.path.join(args.out_root, oid)
    unlit_dir = os.path.join(obj_root, "unlit")

    frame_data: List[Tuple[int, Dict[str, Any], Matrix, str]] = []
    for idx, frame in enumerate(frames):
        try:
            w2c = to_matrix(frame["world_to_camera"])
        except Exception as e:
            log(f"{oid}: invalid world_to_camera at frame {idx} ({e})")
            continue
        base = frame_basename(frame, idx)
        frame_data.append((idx, frame, w2c, base))

    if not frame_data:
        log(f"{oid}: no valid frames to render.")
        return False

    _, last_frame, _, last_base = frame_data[-1]
    lit_name = last_frame.get("file_name") or f"{last_base}_lit.png"
    images = last_frame.get("images") or {}
    unlit_names = {
        name: images.get(name) or f"{last_base}_{name}.png" for name, _ in PASS_CONFIG
    }
    os.makedirs(unlit_dir, exist_ok=True)
    lit_configs: List[Dict[str, str]] = []
    lit_configs_to_render: List[Dict[str, str]] = []
    for entry in args.hdris:
        subdir = "lit" if not args.multi_hdri else os.path.join("lit", entry["name"])
        lit_dir = os.path.join(obj_root, subdir)
        os.makedirs(lit_dir, exist_ok=True)
        cfg = {
            "path": entry["path"],
            "name": entry["name"],
            "subdir": subdir,
            "dir": lit_dir,
        }
        lit_configs.append(cfg)
        if not os.path.exists(os.path.join(lit_dir, lit_name)):
            lit_configs_to_render.append(cfg)

    unlit_done = all(os.path.exists(os.path.join(unlit_dir, fname)) for fname in unlit_names.values())
    if not lit_configs_to_render and unlit_done:
        log(f"{oid}: found existing renders, skip.")
        return True

    scene = reset_scene(width, height, args.samples)
    mesh_obj = import_and_normalize(mesh_path)
    if not mesh_obj:
        log(f"{oid}: import/normalize failed.")
        return False

    def require_texture(name: str, path: str) -> bool:
        if not path or not os.path.exists(path):
            log(f"{oid}: {name} texture missing or not found.")
            return False
        return True

    if not require_texture("albedo", albedo):
        return False
    if not require_texture("rough", rough):
        return False
    if not require_texture("metal", metal):
        return False

    normal_path = normal if normal and os.path.exists(normal) else ""
    if not normal_path:
        log(f"{oid}: normal map missing, using geometry normals for lit and unlit.")

    mat_lit = build_pbr_material(albedo, rough, metal, normal_path)
    emission_mats: Dict[str, bpy.types.Material] = {}
    for name, cs in PASS_CONFIG:
        if name == "normal":
            emission_mats[name] = (
                build_world_normal_material(f"{name.upper()}_EMIT", normal_path)
                if normal_path
                else build_geometry_normal_material(f"{name.upper()}_GEO")
            )
            continue
        path = {"albedo": albedo, "rough": rough, "metal": metal}.get(name)
        if path and os.path.exists(path):
            emission_mats[name] = build_emission_material(f"{name.upper()}_EMIT", path, cs)
        else:
            log(f"{oid}: {name} texture missing for unlit render.")
            return False

    cam = create_camera(scene, lens_mm)
    validate_intrinsics(cam, scene, intrinsics, oid)

    def apply_camera(w2c: Matrix, idx: int) -> bool:
        try:
            cam.matrix_world = w2c.inverted()
        except Exception as e:
            log(f"{oid}: failed to invert w2c at frame {idx} ({e})")
            return False
        bpy.context.view_layer.update()
        restored_w2c = cam.matrix_world.inverted()
        diff = max(abs(restored_w2c[i][j] - w2c[i][j]) for i in range(4) for j in range(4))
        if diff > 1e-4:
            log(f"{oid}: w2c mismatch at frame {idx}, max diff {diff:.6f}")
        return True

    # Lit pass
    if lit_configs_to_render:
        scene.cycles.samples = args.samples
        mesh_obj.data.materials.clear()
        mesh_obj.data.materials.append(mat_lit)
        for lit_cfg in lit_configs_to_render:
            try:
                setup_background(scene, args.background, lit_cfg["path"], args.hdri_strength)
            except Exception as e:
                log(f"{oid}: background setup failed for {lit_cfg['name']} ({e}), skip.")
                continue
            for idx, frame, w2c, base in frame_data:
                if not apply_camera(w2c, idx):
                    continue
                out_name = frame.get("file_name") or f"{base}_lit.png"
                scene.render.filepath = os.path.join(lit_cfg["dir"], out_name)
                with suppress_render_output():
                    bpy.ops.render.render(write_still=True)

    # Unlit pass (emission)
    if not unlit_done:
        scene.cycles.samples = 1
        setup_background(scene, "transparent", "", 0.0)
        for idx, frame, w2c, base in frame_data:
            if not apply_camera(w2c, idx):
                continue
            images = frame.get("images") or {}
            for pass_name, _ in PASS_CONFIG:
                mat = emission_mats.get(pass_name)
                if mat is None:
                    continue
                mesh_obj.data.materials.clear()
                mesh_obj.data.materials.append(mat)
                out_name = images.get(pass_name) or f"{base}_{pass_name}.png"
                scene.render.filepath = os.path.join(unlit_dir, out_name)
                with suppress_render_output():
                    bpy.ops.render.render(write_still=True)

    if args.save_blend:
        blend_path = os.path.join(obj_root, "scene.blend")
        try:
            bpy.ops.wm.save_mainfile(filepath=blend_path)
            log(f"{oid}: saved blend to {blend_path}")
        except Exception as e:
            log(f"{oid}: failed to save blend file ({e})")

    log(f"{oid}: rendered to {obj_root}")
    return True


# ===================== 入口 =====================
def pick_field(row: Dict[str, str], candidates: List[str]) -> str:
    for key in candidates:
        if key in row and row[key]:
            return row[key]
    return ""


def parse_args(argv: List[str]) -> argparse.Namespace:
    # Check if running in worker mode first (worker only needs config file)
    if "--_worker-mode" in argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--_worker-mode", action="store_true")
        parser.add_argument("--_worker-config", type=str, required=True)
        return parser.parse_args(argv)
    
    # Normal mode - full argument parsing
    parser = argparse.ArgumentParser(description="Render generated assets with GT camera poses (transforms.json).")
    parser.add_argument("--manifest", required=True, help="TSV manifest with columns: obj_id, mesh, albedo[, rough, metal, normal, transforms].")
    parser.add_argument("--gt-root", required=True, help="GT render root containing transforms (e.g., datasets/texverse_rendered).")
    parser.add_argument(
        "--transforms-subdir",
        default="",
        help=(
            "Optional subfolder for transforms.json. "
            "Checks {gt-root}/{subdir}/{obj_id}/transforms.json first, then {gt-root}/{obj_id}/{subdir}/transforms.json."
        ),
    )
    parser.add_argument("--out-root", required=True, help="Where rendered images will be saved (per obj_id subfolder).")
    parser.add_argument(
        "--hdri",
        action="append",
        default=[],
        help="HDRI path(s) for lit renders. Repeat or pass a comma-separated list.",
    )
    parser.add_argument("--hdri-strength", type=float, default=1.0, help="HDRI intensity for lit renders.")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples for lit renders (unlit uses 1).")
    parser.add_argument("--background", choices=["black", "white", "hdri", "transparent"], default=None, help="Background mode for lit renders.")
    parser.add_argument("--fallback-focal", type=float, default=50.0, help="Fallback focal length (mm) if transforms.json lacks focal metadata.")
    parser.add_argument("--save-blend", action="store_true", help="Save a .blend per object for debugging.")
    
    # Multi-GPU parallel rendering options
    parser.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated GPU IDs (e.g., '0,1,2,3').")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--workers-per-gpu", type=str, default="auto", help="Workers per GPU: 'auto' or integer.")
    
    return parser.parse_args(argv)


# ===================== Multi-GPU Utilities =====================
def parse_gpu_ids(gpu_ids_str: str) -> List[int]:
    """Parse GPU IDs from comma-separated string."""
    gpu_ids_str = gpu_ids_str.strip()
    if gpu_ids_str.startswith('[') and gpu_ids_str.endswith(']'):
        gpu_ids_str = gpu_ids_str[1:-1]
    return [int(x.strip()) for x in gpu_ids_str.split(',') if x.strip()]


def estimate_workers_per_gpu_blender(gpu_id: int, vram_per_worker_gb: float = 4.0) -> Tuple[int, float]:
    """Estimate workers for Blender rendering based on GPU memory.
    
    Blender rendering is less memory-intensive than deep learning,
    so we can run more workers per GPU.
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits', '-i', str(gpu_id)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            total_mb = int(result.stdout.strip())
            total_gb = total_mb / 1024
            estimated = int(total_gb * 0.85 / vram_per_worker_gb)
            workers = max(1, min(estimated, 6))  # Blender can handle more workers
            return workers, total_gb
    except Exception:
        pass
    return 2, 0  # Safe default


def calculate_workers_per_gpu_blender(gpu_ids: List[int], workers_str: str) -> int:
    """Calculate workers per GPU for Blender rendering."""
    workers_str = str(workers_str).strip().lower()
    log(f"[DEBUG] calculate_workers_per_gpu_blender: input workers_str='{workers_str}', type={type(workers_str)}")
    
    if workers_str == 'auto':
        if not gpu_ids:
            log("[DEBUG] workers_str is 'auto' but no gpu_ids, returning 1")
            return 1
        workers, total_mem = estimate_workers_per_gpu_blender(gpu_ids[0])
        if total_mem > 0:
            log(f"Auto-detected GPU memory: {total_mem:.1f} GB")
        log(f"Auto-calculated workers_per_gpu: {workers}")
        return workers
    else:
        try:
            result = max(1, int(workers_str))
            log(f"[DEBUG] Parsed workers_per_gpu from '{workers_str}' = {result}")
            return result
        except ValueError:
            log(f"[DEBUG] Failed to parse workers_str '{workers_str}', returning default 2")
            return 2


def run_worker_subprocess(
    gpu_id: int,
    worker_id: int,
    tasks: List[Dict[str, str]],
    args_dict: Dict[str, Any],
    result_file: str,
    script_path: str
) -> subprocess.Popen:
    """Launch a Python subprocess for a worker (using bpy module)."""
    
    # Create config file for this worker
    config = {
        'tasks': tasks,
        'args_dict': args_dict,
        'result_file': result_file,
        'gpu_id': gpu_id,
        'worker_id': worker_id,
    }
    config_file = result_file.replace('.pkl', '_config.pkl')
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)
    
    # Set up environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Build Python command (using bpy module instead of blender executable)
    cmd = [
        sys.executable,  # Use current Python interpreter
        script_path,
        '--_worker-mode',
        '--_worker-config', config_file,
    ]
    
    log(f"Launching worker GPU {gpu_id} W{worker_id} ({len(tasks)} tasks)...")
    
    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return p


def run_multi_gpu_render(args: argparse.Namespace, tasks: List[Dict[str, str]], manifest_dir: str) -> int:
    """Coordinate multi-GPU parallel rendering."""
    
    log(f"[DEBUG] run_multi_gpu_render called with args.gpu_ids='{args.gpu_ids}', args.workers_per_gpu='{args.workers_per_gpu}'")
    
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    log(f"[DEBUG] Parsed gpu_ids: {gpu_ids}")
    
    # Use num_gpus to limit the number of GPUs if specified
    if args.num_gpus > 0 and args.num_gpus < len(gpu_ids):
        gpu_ids = gpu_ids[:args.num_gpus]
    
    workers_per_gpu = calculate_workers_per_gpu_blender(gpu_ids, args.workers_per_gpu)
    log(f"[DEBUG] After calculate_workers_per_gpu_blender: workers_per_gpu={workers_per_gpu}")
    
    num_gpus = len(gpu_ids)
    
    total_workers = num_gpus * workers_per_gpu
    log(f"Using {num_gpus} GPU(s): {gpu_ids}")
    log(f"Workers per GPU: {workers_per_gpu}")
    log(f"Total parallel workers: {total_workers}")
    
    # Distribute tasks across workers
    worker_tasks = [[] for _ in range(total_workers)]
    for idx, task in enumerate(tasks):
        worker_idx = idx % total_workers
        worker_tasks[worker_idx].append(task)
    
    # Show distribution
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        gpu_total = sum(len(worker_tasks[gpu_idx * workers_per_gpu + w]) for w in range(workers_per_gpu))
        log(f"  GPU {gpu_id}: {gpu_total} tasks ({workers_per_gpu} workers)")
    
    # Prepare args dict for workers (excluding non-serializable items)
    args_dict = {
        'gt_root': args.gt_root,
        'out_root': args.out_root,
        'transforms_subdir': args.transforms_subdir,
        'hdri': args.hdri,
        'hdri_strength': args.hdri_strength,
        'samples': args.samples,
        'background': args.background,
        'fallback_focal': args.fallback_focal,
        'save_blend': args.save_blend,
        'manifest_dir': manifest_dir,
    }
    
    # Create temp directory for results
    temp_dir = tempfile.mkdtemp(prefix="blender_render_")
    log(f"Using temp directory: {temp_dir}")
    
    # Get script path
    script_path = os.path.abspath(__file__)
    
    # Launch workers
    processes = []
    result_files = []
    
    log(f"[DEBUG] Starting worker launch loop: gpu_ids={gpu_ids}, workers_per_gpu={workers_per_gpu}")
    
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        for local_worker_id in range(workers_per_gpu):
            global_worker_idx = gpu_idx * workers_per_gpu + local_worker_id
            worker_task_list = worker_tasks[global_worker_idx]
            
            log(f"[DEBUG] gpu_idx={gpu_idx}, gpu_id={gpu_id}, local_worker_id={local_worker_id}, global_worker_idx={global_worker_idx}, tasks={len(worker_task_list)}")
            
            if not worker_task_list:
                log(f"[DEBUG] Skipping worker {global_worker_idx} (no tasks)")
                continue
            
            result_file = os.path.join(temp_dir, f"result_gpu{gpu_id}_w{local_worker_id}.pkl")
            result_files.append((gpu_id, local_worker_id, result_file))
            
            p = run_worker_subprocess(
                gpu_id, local_worker_id, worker_task_list, args_dict,
                result_file, script_path
            )
            processes.append((gpu_id, local_worker_id, p))
    
    # Wait for all processes and collect output
    log(f"Waiting for {len(processes)} workers to complete...")
    for gpu_id, worker_id, p in processes:
        stdout, _ = p.communicate()
        if stdout:
            # Print worker output with prefix
            for line in stdout.decode('utf-8', errors='replace').splitlines():
                print(f"[GPU {gpu_id} W{worker_id}] {line}")
        if p.returncode != 0:
            log(f"Worker GPU {gpu_id} W{worker_id} exited with code {p.returncode}")
    
    # Collect results
    total_success = 0
    for gpu_id, worker_id, result_file in result_files:
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            total_success += result.get('success', 0)
            log(f"Collected from GPU {gpu_id} W{worker_id}: {result.get('success', 0)}/{result.get('total', 0)} success")
    
    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        log(f"Failed to cleanup temp dir: {e}")
    
    return total_success


def worker_main(config_file: str) -> None:
    """Worker entry point - runs inside Blender subprocess."""
    
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    
    tasks = config['tasks']
    args_dict = config['args_dict']
    result_file = config['result_file']
    gpu_id = config['gpu_id']
    worker_id = config['worker_id']
    
    worker_tag = f"[GPU {gpu_id} W{worker_id}]"
    log(f"{worker_tag} Starting with {len(tasks)} tasks")
    log(f"{worker_tag} CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # Reconstruct args namespace
    class Args:
        pass
    args = Args()
    for k, v in args_dict.items():
        setattr(args, k, v)
    
    # Build hdri entries
    args.hdris = build_hdri_entries(args.hdri)
    args.multi_hdri = len(args.hdris) > 1
    
    success = 0
    for idx, row in enumerate(tasks):
        oid = row.get("obj_id", f"idx_{idx}")
        log(f"{worker_tag} [{idx + 1}/{len(tasks)}] Rendering {oid}")
        try:
            ok = render_object(row, args)
            success += int(ok)
        except Exception as e:
            log(f"{worker_tag} {oid}: render failed with error {e}")
        finally:
            clear_data_blocks()
    
    # Save results
    result = {
        'success': success,
        'total': len(tasks),
        'gpu_id': gpu_id,
        'worker_id': worker_id,
    }
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    
    log(f"{worker_tag} Finished. Success: {success}/{len(tasks)}")


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    
    # Check if running in worker mode (inside a subprocess)
    if getattr(args, '_worker_mode', False):
        if not args._worker_config:
            log("Worker mode requires --_worker-config")
            return
        worker_main(args._worker_config)
        return
    
    args.out_root = os.path.abspath(args.out_root)
    args.gt_root = os.path.abspath(args.gt_root)
    manifest_path = os.path.abspath(args.manifest)
    manifest_dir = os.path.dirname(manifest_path)

    if args.background is None:
        args.background = "hdri"

    if not os.path.exists(manifest_path):
        log(f"Manifest not found: {manifest_path}")
        return

    args.hdris = build_hdri_entries(args.hdri)
    if not args.hdris:
        log("Lit renders require at least one valid --hdri path.")
        return
    missing_hdris = [entry["path"] for entry in args.hdris if not os.path.exists(entry["path"])]
    if missing_hdris:
        log(f"Lit renders missing HDRI files: {', '.join(missing_hdris)}")
        return
    args.multi_hdri = len(args.hdris) > 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows_raw = list(reader)

    tasks: List[Dict[str, str]] = []
    for row in rows_raw:
        new_row = dict(row)
        new_row["mesh"] = resolve_path(pick_field(row, ["mesh", "mesh_path"]), manifest_dir)
        new_row["albedo"] = resolve_path(row.get("albedo"), manifest_dir)
        new_row["rough"] = resolve_path(row.get("rough") or row.get("roughness"), manifest_dir)
        new_row["metal"] = resolve_path(row.get("metal") or row.get("metallic"), manifest_dir)
        new_row["normal"] = resolve_path(row.get("normal"), manifest_dir)
        new_row["transforms"] = resolve_path(row.get("transforms"), manifest_dir)
        tasks.append(new_row)

    os.makedirs(args.out_root, exist_ok=True)
    log(f"Loaded {len(tasks)} entries from manifest.")
    
    # Determine if we should use multi-process mode
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    workers_per_gpu = calculate_workers_per_gpu_blender(gpu_ids, args.workers_per_gpu)
    total_workers = len(gpu_ids) * workers_per_gpu
    
    # Use multi-process mode when total_workers > 1
    # Each subprocess will import bpy and do the actual rendering
    if total_workers > 1:
        # Running as coordinator process - launch multiple worker subprocesses
        log(f"Running in multi-process mode (GPUs: {gpu_ids}, workers/GPU: {workers_per_gpu}, total: {total_workers})...")
        success = run_multi_gpu_render(args, tasks, manifest_dir)
        log(f"Completed. Total success: {success}/{len(tasks)}")
    else:
        # Single worker mode - render directly in this process
        if not HAS_BPY:
            log("Error: bpy module not available. Cannot render.")
            return
        success = 0
        for idx, row in enumerate(tasks):
            oid = row.get("obj_id", f"idx_{idx}")
            log(f"[{idx + 1}/{len(tasks)}] Rendering {oid}")
            try:
                ok = render_object(row, args)
                success += int(ok)
            except Exception as e:
                log(f"{oid}: render failed with error {e}")
            finally:
                clear_data_blocks()
        log(f"Completed. Success: {success}/{len(tasks)}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    main(argv)
