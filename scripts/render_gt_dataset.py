#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
render_gt_dataset.py

Purpose:
    - Renders unlit channels (albedo/rough/metal/normal/depth) and optionally lit PBR previews per object.

Output layout:
    {out_root}/{obj_id}/lit/   -> 000_lit.png (single HDRI)
    {out_root}/{obj_id}/lit/{hdri_name}/ -> 000_lit.png (multi-HDRI)
    {out_root}/{obj_id}/unlit/ -> 000_albedo.png, 000_rough.png, 000_metal.png, 000_normal.png
    {out_root}/{obj_id}/transforms.json
"""

import argparse
import contextlib
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict, List, Optional

import bpy
from mathutils import Vector

# ===================== 通用配置 =====================
PASS_CONFIG_UNLIT = [
    ("albedo", "sRGB"),
    ("rough", "Non-Color"),
    ("metal", "Non-Color"),
    ("normal", "Non-Color"),  # 法线贴图仅当作颜色
    ("depth", "Non-Color"),   # 深度图用于多视角一致性评估
]

DEFAULT_VIEWS = 64
DEFAULT_RESOLUTION = 512
DEFAULT_BLEND_NAME = "scene.blend"


def resolve_path(path: str, base_dir: str) -> str:
    """Resolve a possibly-relative path using manifest directory as base."""
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


# ===================== 工具函数 =====================
def log(msg: str) -> None:
    print(f"[Render GT] {msg}")


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


def set_color_management(scene: bpy.types.Scene) -> None:
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.sequencer_colorspace_settings.name = "sRGB"


def reset_scene(resolution: int, samples: int) -> bpy.types.Scene:
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

    sc.render.resolution_x = resolution
    sc.render.resolution_y = resolution
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


def find_normal_path(explicit_normal: str, fallback_from: str) -> str:
    """
    查找可用的法线贴图路径：
        1) 如果显式提供 normal 且文件存在，直接返回。
        2) 否则不做自动搜索，直接返回空字符串。
    """
    _ = fallback_from
    if explicit_normal and os.path.exists(explicit_normal):
        return explicit_normal
    return ""


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


def build_geometry_normal_material(name: str) -> bpy.types.Material:
    """Builds an emission material that outputs world-space geometry normals in [0, 1]."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

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


def build_depth_material(name: str, near: float = 0.1, far: float = 10.0) -> bpy.types.Material:
    """Builds a material that outputs normalized camera-space depth.
    
    The depth is normalized to [0, 1] range where:
    - 0 = near plane (closest)
    - 1 = far plane (farthest)
    
    This is essential for multi-view consistency evaluation via reprojection.
    
    Args:
        name: Material name
        near: Near plane distance
        far: Far plane distance
    
    Returns:
        Blender material outputting normalized depth as grayscale emission
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    # Camera Data node gives us view depth (distance from camera in camera-space Z)
    cam_data = nodes.new("ShaderNodeCameraData")
    cam_data.location = (-600, 0)
    
    # Math node to normalize: (depth - near) / (far - near)
    # Step 1: Subtract near
    sub_near = nodes.new("ShaderNodeMath")
    sub_near.operation = "SUBTRACT"
    sub_near.inputs[1].default_value = near
    sub_near.location = (-400, 0)
    
    # Step 2: Divide by (far - near)
    div_range = nodes.new("ShaderNodeMath")
    div_range.operation = "DIVIDE"
    div_range.inputs[1].default_value = far - near
    div_range.location = (-200, 0)
    
    # Clamp to [0, 1]
    clamp = nodes.new("ShaderNodeClamp")
    clamp.inputs["Min"].default_value = 0.0
    clamp.inputs["Max"].default_value = 1.0
    clamp.location = (0, 0)
    
    # Convert to RGB (grayscale)
    combine = nodes.new("ShaderNodeCombineXYZ")
    combine.location = (200, 0)
    
    # Emission output
    emis = nodes.new("ShaderNodeEmission")
    emis.inputs["Strength"].default_value = 1.0
    emis.location = (400, 0)
    
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    
    # Connect: CameraData.ViewZ -> Sub -> Div -> Clamp -> Combine -> Emission -> Output
    links.new(cam_data.outputs["View Z Depth"], sub_near.inputs[0])
    links.new(sub_near.outputs["Value"], div_range.inputs[0])
    links.new(div_range.outputs["Value"], clamp.inputs["Value"])
    links.new(clamp.outputs["Result"], combine.inputs["X"])
    links.new(clamp.outputs["Result"], combine.inputs["Y"])
    links.new(clamp.outputs["Result"], combine.inputs["Z"])
    links.new(combine.outputs["Vector"], emis.inputs["Color"])
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


def fibonacci_sphere(samples: int, radius: float, rnd: float) -> List[Vector]:
    """Generate camera positions on a fibonacci sphere.
    
    Args:
        samples: Number of camera positions
        radius: Sphere radius
        rnd: Random offset value (pre-computed for reproducibility)
    
    Returns:
        List of camera position vectors
    """
    points = []
    offset = 2.0 / samples
    increment = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(samples):
        y = ((i * offset) - 1.0) + (offset * 0.5)
        r = math.sqrt(max(0.0, 1.0 - y * y))
        phi = (i + rnd) * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        points.append(Vector((x, z, y)) * radius)  # Z up
    return points


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


def look_at_origin(cam_obj: bpy.types.Object, target: Vector) -> None:
    direction = (target - cam_obj.location).normalized()
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


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


def clear_data_blocks() -> None:
    # Detach world references to avoid dangling image users
    for sc in list(bpy.data.scenes):
        sc.world = None
    for world in list(bpy.data.worlds):
        if world.use_nodes and world.node_tree:
            world.node_tree.links.clear()
            world.node_tree.nodes.clear()
        bpy.data.worlds.remove(world, do_unlink=True)

    # Remove scene objects first to clear material/mesh users
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

    # Finally remove images; skip Render Result to avoid Blender internal asserts
    for img in list(bpy.data.images):
        if img.name == "Render Result":
            continue
        img.user_clear()
        try:
            bpy.data.images.remove(img)
        except RuntimeError:
            pass


# ===================== 渲染流程 =====================
def render_object(row: Dict[str, str], args: argparse.Namespace, rnd_value: float) -> bool:
    """Render a single object with pre-computed random offset.
    
    Args:
        row: Object data from manifest
        args: Command line arguments
        rnd_value: Pre-computed random offset for camera positions (for reproducibility)
    """
    oid = row.get("obj_id", "unknown")
    mesh_path = row.get("mesh")
    albedo = row.get("albedo")
    rough = row.get("rough") or row.get("roughness")
    metal = row.get("metal") or row.get("metallic")
    normal_raw = row.get("normal")
    normal = find_normal_path(normal_raw, albedo or mesh_path)
    # log(f"{oid}: normal candidate='{normal_raw}', resolved='{normal}', exists={bool(normal and os.path.exists(normal))}")

    required = {
        "mesh": mesh_path,
        "albedo": albedo,
        "rough": rough,
        "metal": metal,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        log(f"{oid}: missing required paths ({', '.join(missing)}), skip.")
        return False
    if not all(os.path.exists(p) for p in required.values() if p):
        log(f"{oid}: file not found, skip.")
        return False
    if not normal:
        log(f"{oid}: normal map missing, using geometry normals for lit and unlit.")

    obj_root = os.path.join(args.out_root, oid)
    unlit_dir = os.path.join(obj_root, "unlit")
    os.makedirs(unlit_dir, exist_ok=True)

    # Build lit configs first (needed for completeness check)
    lit_configs: List[Dict[str, str]] = []
    if not args.unlit_only:
        for entry in args.hdris:
            subdir = "lit" if not args.multi_hdri else os.path.join("lit", entry["name"])
            lit_dir = os.path.join(obj_root, subdir)
            os.makedirs(lit_dir, exist_ok=True)
            lit_configs.append({
                "path": entry["path"],
                "name": entry["name"],
                "subdir": subdir,
                "dir": lit_dir,
            })

    # Check if object is completely rendered (object-level checkpoint)
    # depth is the last pass in PASS_CONFIG_UNLIT
    unlit_done = os.path.join(unlit_dir, f"{args.views - 1:03d}_depth.png")
    object_complete = os.path.exists(unlit_done)
    if object_complete and not args.unlit_only:
        # Also check all lit HDRIs are complete
        for lit_cfg in lit_configs:
            lit_done = os.path.join(lit_cfg["dir"], f"{args.views - 1:03d}_lit.png")
            if not os.path.exists(lit_done):
                object_complete = False
                break

    if object_complete:
        log(f"{oid}: found existing complete renders, skip.")
        return True

    scene = reset_scene(args.resolution, samples=args.samples)
    mesh_obj = import_and_normalize(mesh_path)
    if not mesh_obj:
        log(f"{oid}: import failed.")
        return False

    pbr_mat = build_pbr_material(albedo, rough, metal, normal)
    # Depth material: use camera radius as far plane for normalization
    depth_near = 0.1
    depth_far = args.radius * 2.0  # Object is at origin, camera at radius
    unlit_mats = {
        "albedo": build_emission_material("ALBEDO_EMIT", albedo, "sRGB"),
        "rough": build_emission_material("ROUGH_EMIT", rough, "Non-Color"),
        "metal": build_emission_material("METAL_EMIT", metal, "Non-Color"),
        "normal": build_world_normal_material("NORMAL_EMIT", normal)
        if normal
        else build_geometry_normal_material("NORMAL_GEO"),
        "depth": build_depth_material("DEPTH_EMIT", near=depth_near, far=depth_far),
    }

    cam = create_camera(scene, args.focal_length)
    points = fibonacci_sphere(args.views, args.radius, rnd_value)
    intrinsics = compute_intrinsics(cam, scene)
    frames: List[Dict[str, Any]] = []
    frame_views: List[Dict[str, Any]] = []
    zero = Vector((0.0, 0.0, 0.0))

    for idx, pos in enumerate(points):
        cam.location = pos
        look_at_origin(cam, zero)
        bpy.context.view_layer.update()

        w2c = cam.matrix_world.inverted()
        frame_prefix = f"{idx:03d}"
        frame_views.append({"idx": idx, "pos": pos, "prefix": frame_prefix})
        frames.append(
            {
                "frame_id": idx,
                "file_prefix": frame_prefix,
                "file_name": f"{frame_prefix}_lit.png",
                "images": {
                    "albedo": f"{frame_prefix}_albedo.png",
                    "rough": f"{frame_prefix}_rough.png",
                    "metal": f"{frame_prefix}_metal.png",
                    "normal": f"{frame_prefix}_normal.png",
                    "depth": f"{frame_prefix}_depth.png",
                },
                "world_to_camera": [[float(v) for v in row_vec] for row_vec in w2c],
            }
        )

    # Lit pass (PBR) - render all HDRIs
    if not args.unlit_only and lit_configs:
        scene.cycles.samples = args.samples
        mesh_obj.data.materials.clear()
        mesh_obj.data.materials.append(pbr_mat)
        for lit_cfg in lit_configs:
            setup_background(scene, args.background, lit_cfg["path"], args.hdri_strength)
            for view in frame_views:
                cam.location = view["pos"]
                look_at_origin(cam, zero)
                bpy.context.view_layer.update()
                scene.render.filepath = os.path.join(lit_cfg["dir"], f"{view['prefix']}_lit.png")
                with suppress_render_output():
                    bpy.ops.render.render(write_still=True)

    # Unlit pass (emission, 1 sample)
    scene.cycles.samples = 1
    setup_background(scene, "transparent", "", 0.0)
    for view in frame_views:
        cam.location = view["pos"]
        look_at_origin(cam, zero)
        bpy.context.view_layer.update()

        frame_prefix = view["prefix"]
        for pass_name, _ in PASS_CONFIG_UNLIT:
            mesh_obj.data.materials.clear()
            mesh_obj.data.materials.append(unlit_mats[pass_name])
            scene.render.filepath = os.path.join(unlit_dir, f"{frame_prefix}_{pass_name}.png")
            with suppress_render_output():
                bpy.ops.render.render(write_still=True)

    if args.unlit_only:
        lit_meta = {"enabled": False}
    else:
        lit_meta = {
            "hdris": [
                {"path": cfg["path"], "name": cfg["name"], "subdir": cfg["subdir"]}
                for cfg in lit_configs
            ],
            "hdri_strength": args.hdri_strength,
            "samples": args.samples,
            "background": args.background,
        }
        if len(lit_configs) == 1:
            lit_meta["hdri"] = lit_configs[0]["path"]

    meta = {
        "obj_id": oid,
        "intrinsics": intrinsics,
        "frames": frames,
        "meta": {
            "views": args.views,
            "radius": args.radius,
            "focal_length_mm": args.focal_length,
            "resolution": args.resolution,
            "lit": lit_meta,
            "unlit": {
                "samples": 1,
                "background": "transparent",
            },
            "depth": {
                "near": depth_near,
                "far": depth_far,
                "normalized": True,
                "description": "Depth values are normalized to [0, 1] range where 0=near, 1=far",
            },
        },
    }
    with open(os.path.join(obj_root, "transforms.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2)

    if args.save_blend:
        blend_path = os.path.join(obj_root, DEFAULT_BLEND_NAME)
        try:
            bpy.ops.wm.save_mainfile(filepath=blend_path)
            log(f"{oid}: saved blend to {blend_path}")
        except Exception as e:
            log(f"{oid}: failed to save blend file ({e})")

    log(f"{oid}: renders saved to {obj_root}")
    return True


# ===================== 主入口 =====================
def pick_field(row: Dict[str, str], candidates: List[str]) -> str:
    for key in candidates:
        if key in row and row[key]:
            return row[key]
    return ""


def parse_args(argv: List[str]) -> argparse.Namespace:
    # Check if running in worker mode first (worker only needs minimal args)
    if "--_worker-mode" in argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--_worker-mode", action="store_true")
        parser.add_argument("--_worker-config", type=str, required=True)
        return parser.parse_args(argv)
    
    parser = argparse.ArgumentParser(description="Render GT dataset for TexGaussian (lit + unlit).")
    parser.add_argument("--manifest", required=True, help="Path to manifest_extracted.tsv (Step 2).")
    parser.add_argument("--out-root", required=True, help="Dataset root. Will create {obj_id}/lit and {obj_id}/unlit.")
    parser.add_argument(
        "--hdri",
        action="append",
        default=[],
        help="HDRI path(s) for lit renders. Repeat or pass a comma-separated list.",
    )
    parser.add_argument("--hdri-strength", type=float, default=1.0, help="HDRI intensity for lit renders.")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="Render resolution (square).")
    parser.add_argument("--views", type=int, default=None, help="Number of views. Defaults to 64.")
    parser.add_argument("--radius", type=float, default=2.0, help="Camera orbit radius.")
    parser.add_argument("--focal-length", type=float, default=50.0, help="Camera focal length in mm.")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples for lit renders (unlit uses 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--background", choices=["black", "white", "hdri", "transparent"], default=None, help="Background mode for lit renders (default: hdri).")
    parser.add_argument("--save-blend", action="store_true", help="Save a .blend file per object for inspection.")
    parser.add_argument("--unlit-only", action="store_true", help="Render unlit channels only (skip lit PBR).")
    # Multi-GPU parallel rendering options
    parser.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated GPU IDs (e.g., '0,1,2,3').")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--workers-per-gpu", type=str, default="auto", help="Workers per GPU: 'auto' or integer.")
    # Legacy arguments (kept for backward compatibility)
    parser.add_argument("--workers", type=int, default=None, help="(Deprecated) Use --num-gpus instead.")
    parser.add_argument("--gpu-list", type=str, default=None, help="(Deprecated) Use --gpu-ids instead.")
    return parser.parse_args(argv)


def precompute_rnd_values(num_objects: int, seed: int, views: int) -> List[float]:
    """Pre-compute rnd values for all objects in the same order as serial execution.
    
    This ensures that regardless of parallelization, each object gets the same
    camera positions as it would in serial execution.
    """
    rng = random.Random(seed)
    return [rng.random() * views for _ in range(num_objects)]


# ===================== Multi-GPU Utilities =====================
import pickle
import shutil

def parse_gpu_ids(gpu_ids_str: str) -> List[int]:
    """Parse GPU IDs from comma-separated string."""
    gpu_ids_str = gpu_ids_str.strip()
    if gpu_ids_str.startswith('[') and gpu_ids_str.endswith(']'):
        gpu_ids_str = gpu_ids_str[1:-1]
    return [int(x.strip()) for x in gpu_ids_str.split(',') if x.strip()]


def estimate_workers_per_gpu(gpu_id: int, vram_per_worker_gb: float = 4.0) -> tuple:
    """Estimate workers for Blender rendering based on GPU memory."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits', '-i', str(gpu_id)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            total_mb = int(result.stdout.strip())
            total_gb = total_mb / 1024
            estimated = int(total_gb * 0.85 / vram_per_worker_gb)
            workers = max(1, min(estimated, 6))
            return workers, total_gb
    except Exception:
        pass
    return 2, 0


def calculate_workers_per_gpu(gpu_ids: List[int], workers_str: str) -> int:
    """Calculate workers per GPU."""
    workers_str = str(workers_str).strip().lower()
    
    if workers_str == 'auto':
        if not gpu_ids:
            return 1
        workers, total_mem = estimate_workers_per_gpu(gpu_ids[0])
        if total_mem > 0:
            log(f"Auto-detected GPU memory: {total_mem:.1f} GB")
        log(f"Auto-calculated workers_per_gpu: {workers}")
        return workers
    else:
        try:
            return max(1, int(workers_str))
        except ValueError:
            return 2


def run_worker_subprocess(
    gpu_id: int,
    worker_id: int,
    tasks: List[Dict[str, str]],
    task_indices: List[int],
    rnd_values: List[float],
    args_dict: Dict[str, Any],
    result_file: str,
    script_path: str
) -> subprocess.Popen:
    """Launch a Python subprocess for a worker."""
    
    config = {
        'tasks': tasks,
        'task_indices': task_indices,
        'rnd_values': rnd_values,
        'args_dict': args_dict,
        'result_file': result_file,
        'gpu_id': gpu_id,
        'worker_id': worker_id,
    }
    config_file = result_file.replace('.pkl', '_config.pkl')
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [
        sys.executable,
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


def run_multi_gpu_render(args: argparse.Namespace, tasks: List[Dict[str, str]], rnd_values: List[float]) -> int:
    """Coordinate multi-GPU parallel rendering."""
    
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    
    if args.num_gpus > 0 and args.num_gpus < len(gpu_ids):
        gpu_ids = gpu_ids[:args.num_gpus]
    
    workers_per_gpu = calculate_workers_per_gpu(gpu_ids, args.workers_per_gpu)
    num_gpus = len(gpu_ids)
    total_workers = num_gpus * workers_per_gpu
    
    log(f"Using {num_gpus} GPU(s): {gpu_ids}")
    log(f"Workers per GPU: {workers_per_gpu}")
    log(f"Total parallel workers: {total_workers}")
    
    # Distribute tasks across workers
    worker_tasks = [[] for _ in range(total_workers)]
    worker_indices = [[] for _ in range(total_workers)]
    worker_rnds = [[] for _ in range(total_workers)]
    
    for idx, (task, rnd) in enumerate(zip(tasks, rnd_values)):
        worker_idx = idx % total_workers
        worker_tasks[worker_idx].append(task)
        worker_indices[worker_idx].append(idx)
        worker_rnds[worker_idx].append(rnd)
    
    # Show distribution
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        gpu_total = sum(len(worker_tasks[gpu_idx * workers_per_gpu + w]) for w in range(workers_per_gpu))
        log(f"  GPU {gpu_id}: {gpu_total} tasks ({workers_per_gpu} workers)")
    
    # Prepare args dict for workers
    args_dict = {
        'out_root': args.out_root,
        'hdri': args.hdri,
        'hdri_strength': args.hdri_strength,
        'resolution': args.resolution,
        'views': args.views,
        'radius': args.radius,
        'focal_length': args.focal_length,
        'samples': args.samples,
        'seed': args.seed,
        'background': args.background,
        'save_blend': args.save_blend,
        'unlit_only': args.unlit_only,
    }
    
    temp_dir = tempfile.mkdtemp(prefix="gt_render_")
    log(f"Using temp directory: {temp_dir}")
    
    script_path = os.path.abspath(__file__)
    
    processes = []
    result_files = []
    
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        for local_worker_id in range(workers_per_gpu):
            global_worker_idx = gpu_idx * workers_per_gpu + local_worker_id
            task_list = worker_tasks[global_worker_idx]
            idx_list = worker_indices[global_worker_idx]
            rnd_list = worker_rnds[global_worker_idx]
            
            if not task_list:
                continue
            
            result_file = os.path.join(temp_dir, f"result_gpu{gpu_id}_w{local_worker_id}.pkl")
            result_files.append((gpu_id, local_worker_id, result_file))
            
            p = run_worker_subprocess(
                gpu_id, local_worker_id, task_list, idx_list, rnd_list,
                args_dict, result_file, script_path
            )
            processes.append((gpu_id, local_worker_id, p))
    
    log(f"Waiting for {len(processes)} workers to complete...")
    for gpu_id, worker_id, p in processes:
        stdout, _ = p.communicate()
        if stdout:
            for line in stdout.decode('utf-8', errors='replace').splitlines():
                print(f"[GPU {gpu_id} W{worker_id}] {line}")
        if p.returncode != 0:
            log(f"Worker GPU {gpu_id} W{worker_id} exited with code {p.returncode}")
    
    total_success = 0
    for gpu_id, worker_id, result_file in result_files:
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            total_success += result.get('success', 0)
            log(f"Collected from GPU {gpu_id} W{worker_id}: {result.get('success', 0)}/{result.get('total', 0)} success")
    
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        log(f"Failed to cleanup temp dir: {e}")
    
    return total_success


def worker_main(config_file: str) -> None:
    """Worker entry point - runs inside subprocess."""
    
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    
    tasks = config['tasks']
    task_indices = config['task_indices']
    rnd_values = config['rnd_values']
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
    args.multi_hdri = len(args.hdris) > 1 if not args.unlit_only else False
    
    success = 0
    for i, (task, orig_idx, rnd_value) in enumerate(zip(tasks, task_indices, rnd_values)):
        oid = task.get("obj_id", f"idx_{orig_idx}")
        mode_label = "unlit-only" if args.unlit_only else "lit+unlit"
        log(f"{worker_tag} [{i + 1}/{len(tasks)}] Rendering {oid} ({mode_label})")
        try:
            ok = render_object(task, args, rnd_value)
            success += int(ok)
        except Exception as e:
            log(f"{oid}: render failed with error {e}")
        finally:
            clear_data_blocks()
    
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
    
    # Check if running in worker mode
    if getattr(args, '_worker_mode', False):
        if not args._worker_config:
            log("Worker mode requires --_worker-config")
            return
        worker_main(args._worker_config)
        return
    
    args.out_root = os.path.abspath(args.out_root)
    manifest_path = os.path.abspath(args.manifest)
    manifest_dir = os.path.dirname(manifest_path)

    if args.views is None:
        args.views = DEFAULT_VIEWS

    if args.background is None:
        args.background = "hdri"

    if not os.path.exists(manifest_path):
        log(f"Manifest not found: {manifest_path}")
        return

    args.hdris = build_hdri_entries(args.hdri)
    if not args.unlit_only:
        if not args.hdris:
            log("Lit renders require at least one valid --hdri path.")
            return
        missing_hdris = [entry["path"] for entry in args.hdris if not os.path.exists(entry["path"])]
        if missing_hdris:
            log(f"Lit renders missing HDRI files: {', '.join(missing_hdris)}")
            return
        args.multi_hdri = len(args.hdris) > 1
    else:
        args.multi_hdri = False

    os.makedirs(args.out_root, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        tasks_raw = list(reader)

    tasks: List[Dict[str, str]] = []
    for row in tasks_raw:
        new_row = dict(row)
        new_row["mesh"] = resolve_path(row.get("mesh") or row.get("mesh_path"), manifest_dir)
        new_row["albedo"] = resolve_path(row.get("albedo"), manifest_dir)
        new_row["rough"] = resolve_path(row.get("rough") or row.get("roughness"), manifest_dir)
        new_row["metal"] = resolve_path(row.get("metal") or row.get("metallic"), manifest_dir)
        new_row["normal"] = resolve_path(row.get("normal"), manifest_dir)
        tasks.append(new_row)

    log(f"Loaded {len(tasks)} entries from manifest.")

    # Handle backward compatibility with legacy arguments
    if args.workers is not None and args.workers > 1:
        # Legacy mode: --workers N means N GPUs with 1 worker each
        args.num_gpus = args.workers
        args.workers_per_gpu = "1"
        if args.gpu_list:
            args.gpu_ids = args.gpu_list
    
    # Pre-compute rnd values for all objects (same order as serial execution)
    rnd_values = precompute_rnd_values(len(tasks), args.seed, args.views)
    
    # Determine if we should use multi-process mode
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    workers_per_gpu = calculate_workers_per_gpu(gpu_ids, args.workers_per_gpu)
    total_workers = len(gpu_ids) * workers_per_gpu
    
    # Use multi-process mode when total_workers > 1
    if total_workers > 1:
        log(f"Running in multi-process mode (GPUs: {gpu_ids}, workers/GPU: {workers_per_gpu}, total: {total_workers})...")
        success = run_multi_gpu_render(args, tasks, rnd_values)
        log(f"Completed. Total success: {success}/{len(tasks)}")
    else:
        # Single worker mode - render directly in this process
        success = 0
        for idx, (task, rnd_value) in enumerate(zip(tasks, rnd_values)):
            oid = task.get("obj_id", f"idx_{idx}")
            mode_label = "unlit-only" if args.unlit_only else "lit+unlit"
            log(f"[{idx + 1}/{len(tasks)}] Rendering {oid} ({mode_label})")
            try:
                ok = render_object(task, args, rnd_value)
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
