"""
Material/textures overlay STL → stills/animation (new pipeline).

Goal: avoid "posterize + emission washout" and instead use a fixed vivid
palette with procedural textures driving roughness/normal and a small amount
of color mixing. This tends to produce visibly colored surfaces.
"""

import hashlib
import os
from math import radians
from pathlib import Path

import bmesh
import colorsys
import bpy
from mathutils import Vector


# Animation defaults (entrypoints can override for quick PNGs).
frames = 120
fps = 30
resolution = (512, 512)
rotation_axis = "Z"
degrees_to_rotate = 360
animation_length_seconds = 4

output_format = "MPEG4"
video_codec = "H264"
file_format = "FFMPEG"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TEXTURE_LIBRARY = _PROJECT_ROOT / "data" / "texture_library"


def _apply_render_settings() -> None:
    render = bpy.context.scene.render
    # Blender 5.1 moved video format handling away from image_settings.file_format.
    # Try the newer render.file_format first, then legacy fallback.
    try:
        render.file_format = file_format
    except Exception:
        try:
            render.image_settings.file_format = file_format
        except Exception:
            pass

    render.fps = fps
    render.resolution_x = resolution[0]
    render.resolution_y = resolution[1]
    render.resolution_percentage = 100

    render.ffmpeg.format = output_format
    render.ffmpeg.codec = video_codec
    render.ffmpeg.constant_rate_factor = "HIGH"

    render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.cycles.use_denoising = True

    # Reduce clipping/whitening a bit.
    bpy.context.scene.view_settings.exposure = 0.95
    try:
        bpy.context.scene.view_settings.view_transform = "Standard"
        bpy.context.scene.view_settings.look = "None"
    except Exception:
        pass


_apply_render_settings()


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.lights:
        bpy.data.lights.remove(block)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block)


def center_and_scale_object(obj, target_size: float = 2.0) -> float:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    if bm.verts:
        verts = [v.co.copy() for v in bm.verts]
        min_x = min(v.x for v in verts)
        max_x = max(v.x for v in verts)
        min_y = min(v.y for v in verts)
        max_y = max(v.y for v in verts)
        min_z = min(v.z for v in verts)
        max_z = max(v.z for v in verts)

        center = Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2))
        dimensions = Vector((max_x - min_x, max_y - min_y, max_z - min_z))
        max_dim = max(dimensions.x, dimensions.y, dimensions.z)

        for v in bm.verts:
            v.co -= center

        if max_dim > 0:
            scale_factor = target_size / max_dim
            for v in bm.verts:
                v.co *= scale_factor

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")

    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    obj.scale = (1, 1, 1)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return target_size


def setup_lighting(object_size: float, *, material_style: str = "overlay") -> None:
    distance = object_size * 4

    if material_style == "realistic":
        # Lower-contrast lighting to avoid one side blowing out.
        main_energy = 6.5
        fill_energy = 4.8
        top_energy = 2.8
        front_energy = 2.6
        backfill_energy = 3.2
    else:
        main_energy = 10.0
        fill_energy = 12.0
        top_energy = 6.0
        front_energy = 5.0
        backfill_energy = 7.0

    main_light_data = bpy.data.lights.new(name="MainLight", type="SUN")
    main_light_data.energy = main_energy
    main_light = bpy.data.objects.new(name="MainLight", object_data=main_light_data)
    bpy.context.collection.objects.link(main_light)
    main_light.location = (distance, -distance, distance)
    main_light.rotation_euler = (radians(45), 0, radians(45))

    fill_light_data = bpy.data.lights.new(name="FillLight", type="SUN")
    fill_light_data.energy = fill_energy
    fill_light = bpy.data.objects.new(name="FillLight", object_data=fill_light_data)
    bpy.context.collection.objects.link(fill_light)
    fill_light.location = (-distance, distance, distance * 0.5)
    fill_light.rotation_euler = (radians(30), 0, radians(-135))

    top_light_data = bpy.data.lights.new(name="TopLight", type="SUN")
    top_light_data.energy = top_energy
    top_light = bpy.data.objects.new(name="TopLight", object_data=top_light_data)
    bpy.context.collection.objects.link(top_light)
    top_light.location = (0, 0, distance * 2)
    top_light.rotation_euler = (0, 0, 0)

    front_light_data = bpy.data.lights.new(name="FrontLight", type="SUN")
    front_light_data.energy = front_energy
    front_light = bpy.data.objects.new(name="FrontLight", object_data=front_light_data)
    bpy.context.collection.objects.link(front_light)
    front_light.location = (0, -distance, 0)
    front_light.rotation_euler = (radians(90), 0, 0)

    # Extra back fill so the far side doesn't go nearly black.
    back_fill_data = bpy.data.lights.new(name="BackFillLight", type="SUN")
    back_fill_data.energy = backfill_energy
    back_fill = bpy.data.objects.new(name="BackFillLight", object_data=back_fill_data)
    bpy.context.collection.objects.link(back_fill)
    back_fill.location = (distance, -distance, distance * 0.2)
    back_fill.rotation_euler = (radians(55), 0, radians(140))


def setup_world_background(*, material_style: str = "overlay") -> None:
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    # Dark neutral background makes saturated colors read better.
    bg_node = nodes.new(type="ShaderNodeBackground")
    if material_style == "realistic":
        bg_node.inputs[0].default_value = (0.08, 0.08, 0.08, 1)
        bg_node.inputs[1].default_value = 0.55
    else:
        bg_node.inputs[0].default_value = (0.04, 0.04, 0.04, 1)
        bg_node.inputs[1].default_value = 0.9

    output_node = nodes.new(type="ShaderNodeOutputWorld")
    world.node_tree.links.new(bg_node.outputs[0], output_node.inputs[0])


def _stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _hsv_to_rgba(h: float, s: float, v: float):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b, 1.0)


def _offset_hue(h: float, offset: float) -> float:
    return (h + offset) % 1.0


def _palette_color(seed: int):
    # Vivid palette; no white/near-white to avoid "chalk" results.
    palette_hsv = [
        (0.02, 0.95, 0.85),  # red
        (0.14, 0.95, 0.85),  # yellow
        (0.30, 0.95, 0.85),  # green
        (0.52, 0.95, 0.85),  # cyan
        (0.72, 0.95, 0.85),  # blue
        (0.88, 0.95, 0.85),  # magenta
    ]
    idx = seed % len(palette_hsv)
    c1_hsv = palette_hsv[idx]
    c2_hsv = palette_hsv[(idx + 2) % len(palette_hsv)]
    return _hsv_to_rgba(c1_hsv[0], c1_hsv[1], c1_hsv[2]), _hsv_to_rgba(
        c2_hsv[0], c2_hsv[1], c2_hsv[2]
    )


def _preset_from_seed(seed: int) -> str:
    presets = ["plastic", "rubber", "painted_metal", "ceramic"]
    return presets[seed % len(presets)]


def _set_principled_specular(principled, value: float) -> None:
    try:
        principled.inputs["Specular IOR Level"].default_value = value
    except KeyError:
        try:
            principled.inputs["Specular"].default_value = value
        except KeyError:
            pass


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _iter_texture_set_dirs(root: Path):
    if not root.exists():
        return []
    out = []
    for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir():
            continue
        has_img_here = any(
            c.is_file() and c.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".exr"}
            for c in p.iterdir()
        )
        has_img_nested = any(
            c.is_file() and c.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".exr"}
            for c in p.rglob("*")
        )
        if has_img_here or has_img_nested:
            out.append(p)
    return out


def _find_map_file(set_dir: Path, keywords):
    files = sorted([p for p in set_dir.rglob("*") if p.is_file()], key=lambda p: str(p).lower())
    for p in files:
        n = p.name.lower()
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".exr"}:
            continue
        if "preview" in n:
            continue
        if any(k in n for k in keywords):
            return p
    return None


def _resolve_pbr_maps(set_dir: Path):
    # Covers common naming across texture sites/exporters.
    return {
        "basecolor": _find_map_file(set_dir, ["basecolor", "albedo", "color", "diffuse", "diff"]),
        "ao": _find_map_file(set_dir, ["ambientocclusion", "ambient_occlusion", "ao"]),
        "roughness": _find_map_file(set_dir, ["roughness", "rough"]),
        "metallic": _find_map_file(set_dir, ["metalness", "metallic", "metal"]),
        "normal": _find_map_file(set_dir, ["normalgl", "normal", "nor"]),
        "height": _find_map_file(set_dir, ["height", "displacement", "disp", "bump"]),
    }


def _pick_texture_set(seed: int, *, prefer_keywords):
    root_raw = os.environ.get("ALICE_TEXTURE_LIBRARY", "").strip()
    root = Path(root_raw) if root_raw else _DEFAULT_TEXTURE_LIBRARY
    set_dirs = _iter_texture_set_dirs(root)
    if not set_dirs:
        return None

    preferred = []
    if prefer_keywords:
        kws = [k.lower() for k in prefer_keywords]
        for d in set_dirs:
            name = d.name.lower()
            if any(k in name for k in kws):
                preferred.append(d)
    pool = preferred if preferred else set_dirs
    return pool[seed % len(pool)]


def _new_image_node(nodes, image_path: Path, *, colorspace: str):
    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path), check_existing=True)
    try:
        tex.image.colorspace_settings.name = colorspace
    except Exception:
        pass
    tex.projection = "BOX"
    tex.projection_blend = 0.2
    return tex


def _build_pbr_textured_material(
    mat,
    *,
    texture_set_dir: Path,
    tint_rgba,
    metallic: float,
    roughness_default: float,
    normal_strength: float,
    tex_scale: float,
    tint_strength: float,
) -> bool:
    maps = _resolve_pbr_maps(texture_set_dir)
    base_map = maps.get("basecolor")
    if base_map is None:
        return False

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")

    mapping.inputs["Scale"].default_value = (tex_scale, tex_scale, tex_scale)
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness_default
    _set_principled_specular(principled, 0.32)

    base_tex = _new_image_node(nodes, base_map, colorspace="sRGB")
    base_color_socket = base_tex.outputs["Color"]
    ao_map = maps.get("ao")
    if ao_map is not None:
        ao_tex = _new_image_node(nodes, ao_map, colorspace="Non-Color")
        ao_mix = nodes.new(type="ShaderNodeMixRGB")
        ao_mix.blend_type = "MULTIPLY"
        ao_mix.inputs["Fac"].default_value = 0.5
        links.new(mapping.outputs["Vector"], ao_tex.inputs["Vector"])
        links.new(base_tex.outputs["Color"], ao_mix.inputs["Color1"])
        links.new(ao_tex.outputs["Color"], ao_mix.inputs["Color2"])
        base_color_socket = ao_mix.outputs["Color"]

    tint = nodes.new(type="ShaderNodeMixRGB")
    tint.blend_type = "MIX"
    tint.inputs["Fac"].default_value = tint_strength
    tint.inputs["Color2"].default_value = tint_rgba

    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], base_tex.inputs["Vector"])
    links.new(base_color_socket, tint.inputs["Color1"])
    links.new(tint.outputs["Color"], principled.inputs["Base Color"])

    rough_map = maps.get("roughness")
    if rough_map is not None:
        rough_tex = _new_image_node(nodes, rough_map, colorspace="Non-Color")
        links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
        links.new(rough_tex.outputs["Color"], principled.inputs["Roughness"])

    metallic_map = maps.get("metallic")
    if metallic_map is not None:
        metal_tex = _new_image_node(nodes, metallic_map, colorspace="Non-Color")
        links.new(mapping.outputs["Vector"], metal_tex.inputs["Vector"])
        links.new(metal_tex.outputs["Color"], principled.inputs["Metallic"])

    normal_map = maps.get("normal")
    height_map = maps.get("height")
    if normal_map is not None:
        ntex = _new_image_node(nodes, normal_map, colorspace="Non-Color")
        nmap = nodes.new(type="ShaderNodeNormalMap")
        nmap.inputs["Strength"].default_value = normal_strength
        links.new(mapping.outputs["Vector"], ntex.inputs["Vector"])
        links.new(ntex.outputs["Color"], nmap.inputs["Color"])
        if height_map is not None:
            htex = _new_image_node(nodes, height_map, colorspace="Non-Color")
            bump = nodes.new(type="ShaderNodeBump")
            bump.inputs["Strength"].default_value = max(0.15, 0.45 * normal_strength)
            links.new(mapping.outputs["Vector"], htex.inputs["Vector"])
            links.new(htex.outputs["Color"], bump.inputs["Height"])
            links.new(nmap.outputs["Normal"], bump.inputs["Normal"])
            links.new(bump.outputs["Normal"], principled.inputs["Normal"])
        else:
            links.new(nmap.outputs["Normal"], principled.inputs["Normal"])
    elif height_map is not None:
        htex = _new_image_node(nodes, height_map, colorspace="Non-Color")
        bump = nodes.new(type="ShaderNodeBump")
        bump.inputs["Strength"].default_value = max(0.12, 0.4 * normal_strength)
        links.new(mapping.outputs["Vector"], htex.inputs["Vector"])
        links.new(htex.outputs["Color"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], principled.inputs["Normal"])

    links.new(principled.outputs[0], output.inputs[0])
    return True


def _build_matte_solid_material(mat, color_rgba) -> None:
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled.inputs["Base Color"].default_value = color_rgba
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.82
    _set_principled_specular(principled, 0.22)
    links.new(principled.outputs[0], output.inputs[0])


def _build_patterned_solid_material(mat, color_light, color_dark, *, seed: int) -> None:
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    noise = nodes.new(type="ShaderNodeTexNoise")
    ramp = nodes.new(type="ShaderNodeValToRGB")
    bump = nodes.new(type="ShaderNodeBump")
    output = nodes.new(type="ShaderNodeOutputMaterial")

    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.58
    _set_principled_specular(principled, 0.35)
    bump.inputs["Strength"].default_value = 0.22

    noise.inputs["Scale"].default_value = 12.0 + (seed % 5)
    try:
        noise.inputs["Detail"].default_value = 10.0
        noise.inputs["Roughness"].default_value = 0.48
    except KeyError:
        pass

    ramp.color_ramp.elements[0].position = 0.38
    ramp.color_ramp.elements[1].position = 0.62
    ramp.color_ramp.elements[0].color = color_dark
    ramp.color_ramp.elements[1].color = color_light

    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    links.new(principled.outputs[0], output.inputs[0])


def apply_material_stimulus_variant(obj, seed: int, *, stimulus_mode: str, variant_index: int) -> None:
    """
    Deterministic 2-version materials for shape-bias stimuli.

    stimulus_mode:
      - B_controlled_simple: same color family, matte vs patterned.
      - A_auto_contrast: high-separation color/material pair.
    variant_index:
      - 1 or 2
    """
    if variant_index not in (1, 2):
        raise ValueError(f"variant_index must be 1 or 2, got {variant_index}")

    mode = (stimulus_mode or "").strip()
    if mode not in {"B_controlled_simple", "A_auto_contrast"}:
        raise ValueError(f"Unsupported stimulus_mode: {mode}")

    mat = bpy.data.materials.new(name=f"StimulusMaterial_{mode}_v{variant_index}")
    mat.use_nodes = True

    base_h = (seed % 360) / 360.0
    use_img_textures = _env_truthy("ALICE_STIMULUS_USE_IMAGE_TEXTURES", default=True)

    if mode == "B_controlled_simple":
        # Reference/texture_match share one textured material.
        # shape_match is always a contrasting hue so color never matches reference.
        c_base = _hsv_to_rgba(base_h, 0.58, 0.82)
        c_ref_tint = _hsv_to_rgba(base_h, 0.50, 0.90)
        alt_h = _offset_hue(base_h, 0.34)
        c_alt_base = _hsv_to_rgba(alt_h, 0.74, 0.88)
        c_alt_light = _hsv_to_rgba(alt_h, 0.62, 0.94)
        c_alt_dark = _hsv_to_rgba(alt_h, 0.78, 0.62)
        if variant_index == 1:
            used_pbr = False
            if use_img_textures:
                tex_set = _pick_texture_set(seed, prefer_keywords=["fabric", "cloth", "carpet", "leather"])
                if tex_set is not None:
                    used_pbr = _build_pbr_textured_material(
                        mat,
                        texture_set_dir=tex_set,
                        tint_rgba=c_ref_tint,
                        metallic=0.0,
                        roughness_default=0.62,
                        normal_strength=2.2,
                        tex_scale=0.8,
                        tint_strength=0.14,
                    )
            if not used_pbr:
                _build_patterned_solid_material(mat, c_ref_tint, c_base, seed=seed)
        else:
            if _env_truthy("ALICE_STIMULUS_B_PATTERNED_SHAPE_MATCH", default=False):
                _build_patterned_solid_material(mat, c_alt_light, c_alt_dark, seed=(seed ^ 0xABCD))
            else:
                _build_matte_solid_material(mat, c_alt_base)
    else:
        # High-separation pair: opposite hue and finish contrast.
        alt_h = (base_h + 0.5) % 1.0
        c_v1 = _hsv_to_rgba(base_h, 0.88, 0.86)
        c_v2_light = _hsv_to_rgba(alt_h, 0.95, 0.95)
        c_v2_dark = _hsv_to_rgba(alt_h, 0.90, 0.70)
        if variant_index == 1:
            used_pbr = False
            if use_img_textures:
                tex_set = _pick_texture_set(seed ^ 0x5A5A, prefer_keywords=["steel", "metal", "rust", "corrugated"])
                if tex_set is not None:
                    used_pbr = _build_pbr_textured_material(
                        mat,
                        texture_set_dir=tex_set,
                        tint_rgba=c_v2_light,
                        metallic=0.55,
                        roughness_default=0.48,
                        normal_strength=2.8,
                        tex_scale=0.7,
                        tint_strength=0.12,
                    )
            if not used_pbr:
                _build_patterned_solid_material(mat, c_v2_light, c_v2_dark, seed=(seed ^ 0x5A5A))
        else:
            _build_matte_solid_material(mat, c_v1)

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def apply_material_overlay(obj, seed: int, *, material_style: str = "overlay") -> None:
    mat = bpy.data.materials.new(name="ObjectMaterialOverlay")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.5
    # Emission can make materials look less physically plausible.
    if material_style == "realistic":
        principled.inputs["Emission Strength"].default_value = 0.0
    else:
        principled.inputs["Emission Strength"].default_value = 0.6

    # Two vivid palette colors; mix between them using procedural texture.
    c1, c2 = _palette_color(seed)

    # Procedural texture coordinates from object space.
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    noise = nodes.new(type="ShaderNodeTexNoise")
    bump = nodes.new(type="ShaderNodeBump")

    # Quantize noise to "material blotches" rather than smooth gradients.
    mult = nodes.new(type="ShaderNodeMath")
    mult.operation = "MULTIPLY"
    flo = nodes.new(type="ShaderNodeMath")
    flo.operation = "FLOOR"
    div = nodes.new(type="ShaderNodeMath")
    div.operation = "DIVIDE"

    mix = nodes.new(type="ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs["Fac"].default_value = 0.0

    # Roughness variation (texture-driven, not color-driven).
    rough_add = nodes.new(type="ShaderNodeMath")
    rough_add.operation = "ADD"
    rough_mul = nodes.new(type="ShaderNodeMath")
    rough_mul.operation = "MULTIPLY"

    output = nodes.new(type="ShaderNodeOutputMaterial")

    preset = _preset_from_seed(seed)
    if preset == "plastic":
        metallic = 0.05
        rough_base = 0.55 if material_style == "realistic" else 0.45
        bump_strength = 0.06 if material_style == "realistic" else 0.045
        tex_scale = 18.0 + (seed % 7)
    elif preset == "rubber":
        metallic = 0.0
        rough_base = 0.90 if material_style == "realistic" else 0.85
        bump_strength = 0.08 if material_style == "realistic" else 0.065
        tex_scale = 14.0 + (seed % 9)
    elif preset == "painted_metal":
        metallic = 0.75
        rough_base = 0.35 if material_style == "realistic" else 0.22
        bump_strength = 0.055 if material_style == "realistic" else 0.03
        tex_scale = 26.0 + (seed % 7)
    else:  # ceramic
        metallic = 0.02
        rough_base = 0.80 if material_style == "realistic" else 0.65
        bump_strength = 0.075 if material_style == "realistic" else 0.05
        tex_scale = 16.0 + (seed % 8)

    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = rough_base

    # Configure mapping + noise.
    tex_scale = float(tex_scale)
    mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    noise.inputs["Scale"].default_value = tex_scale
    bump.inputs["Strength"].default_value = bump_strength
    try:
        bump.inputs["Distance"].default_value = 0.05 if material_style == "realistic" else 0.1
    except KeyError:
        pass
    try:
        noise.inputs["Detail"].default_value = 7.0 + ((seed >> 3) % 6)
    except KeyError:
        pass

    # Quantization levels for blotches.
    levels = 5.0
    mult.inputs["Value_001"].default_value = float(levels)
    div.inputs["Value_001"].default_value = float(levels - 1.0)

    # Mix between colors using quantized noise.
    mix.inputs["Color1"].default_value = c1
    mix.inputs["Color2"].default_value = c2

    # Layout (helps when debugging in Blender).
    tex_coord.location = (-900, 0)
    mapping.location = (-650, 0)
    noise.location = (-450, 0)
    mult.location = (-250, 0)
    flo.location = (-50, 0)
    div.location = (80, 0)
    mix.location = (260, 0)
    principled.location = (520, 0)
    rough_mul.location = (260, -220)
    rough_add.location = (520, -220)
    bump.location = (-450, -250)
    output.location = (760, 0)

    # Links.
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    # Quantize the noise fac into a stable 0..1 index-ish factor.
    links.new(noise.outputs["Fac"], mult.inputs["Value"])
    links.new(mult.outputs["Value"], flo.inputs["Value"])
    links.new(flo.outputs["Value"], div.inputs["Value"])

    if material_style == "realistic":
        # Smooth mixing for a more physically plausible look.
        links.new(noise.outputs["Fac"], mix.inputs["Fac"])
    else:
        links.new(div.outputs["Value"], mix.inputs["Fac"])
    links.new(mix.outputs["Color"], principled.inputs["Base Color"])
    if material_style != "realistic":
        links.new(mix.outputs["Color"], principled.inputs["Emission Color"])

    # Roughness modulation from the same factor (keeps material variation without
    # destroying the albedo colors).
    rough_mul.inputs["Value_001"].default_value = 0.25 if material_style == "realistic" else 0.35
    if material_style == "realistic":
        links.new(noise.outputs["Fac"], rough_mul.inputs["Value"])
    else:
        links.new(div.outputs["Value"], rough_mul.inputs["Value"])
    links.new(rough_mul.outputs["Value"], rough_add.inputs["Value"])
    rough_add.inputs["Value_001"].default_value = float(rough_base)
    links.new(rough_add.outputs["Value"], principled.inputs["Roughness"])

    # Clearcoat helps bring back believable "surface depth" highlights.
    if material_style == "realistic":
        try:
            principled.inputs["Clearcoat"].default_value = 0.18
            principled.inputs["Clearcoat Roughness"].default_value = 0.12
        except KeyError:
            pass
        try:
            principled.inputs["Specular IOR Level"].default_value = 1.45
        except KeyError:
            pass

    # Feed bump from noise height only; do not inject vectors into normal input.
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])

    links.new(principled.outputs[0], output.inputs[0])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def setup_scene(obj, object_size: float, material_seed: int, *, material_style: str = "overlay") -> None:
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    distance = object_size * 3.0
    cam.location = (distance * 0.8, -distance * 0.8, object_size * 0.2)
    direction = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 50
    cam.data.clip_end = 1000

    if material_style == "realistic":
        bpy.context.scene.view_settings.exposure = 0.78
    else:
        bpy.context.scene.view_settings.exposure = 0.95

    setup_lighting(object_size, material_style=material_style)
    setup_world_background(material_style=material_style)
    apply_material_overlay(obj, material_seed, material_style=material_style)


def animate_rotation(obj, total_frames: int) -> None:
    obj.rotation_mode = "XYZ"
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    obj.keyframe_insert(data_path="rotation_euler", frame=1)
    obj.keyframe_insert(data_path="location", frame=1)

    if rotation_axis == "X":
        obj.rotation_euler = (radians(degrees_to_rotate), 0, 0)
    elif rotation_axis == "Y":
        obj.rotation_euler = (0, radians(degrees_to_rotate), 0)
    else:
        obj.rotation_euler = (0, 0, radians(degrees_to_rotate))

    obj.keyframe_insert(data_path="rotation_euler", frame=total_frames)
    obj.keyframe_insert(data_path="location", frame=total_frames)

    anim = obj.animation_data
    if not anim:
        return
    action = getattr(anim, "action", None)
    if not action:
        return

    # Blender 5.x may use slotted actions where direct .fcurves is unavailable.
    # Keep compatibility by setting interpolation only when curve access exists.
    fcurves = getattr(action, "fcurves", None)
    if fcurves is None:
        fcurves = getattr(action, "curves", None)
    if not fcurves:
        return

    for fcurve in fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def render_video(output_path: str) -> None:
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frames
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(animation=True)


def render_still(output_path: str) -> None:
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def main(
    input_folder: str,
    output_folder: str,
    *,
    render_mode: str = "mp4",
    material_style: str = "overlay",
) -> None:
    """
    Walk input_folder for .stl (any depth), mirror relative paths under output_folder.

    render_mode:
      - "mp4": H264 MP4 animation per STL (may still depend on ffmpeg in env)
      - "png": single PNG still per STL
    """
    os.makedirs(output_folder, exist_ok=True)
    input_folder = os.path.abspath(input_folder)

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if not filename.lower().endswith(".stl"):
                continue

            stl_path = os.path.join(root, filename)
            rel_dir = os.path.relpath(root, input_folder)
            if rel_dir in (os.curdir, ".", ""):
                output_subfolder = output_folder
            else:
                output_subfolder = os.path.join(output_folder, rel_dir)
            os.makedirs(output_subfolder, exist_ok=True)

            base_name = os.path.splitext(filename)[0]
            seed = _stable_int(stl_path)

            clear_scene()
            bpy.ops.wm.stl_import(filepath=stl_path)
            obj = bpy.context.selected_objects[0]
            object_size = center_and_scale_object(obj, target_size=2.0)
            setup_scene(obj, object_size, material_seed=seed, material_style=material_style)

            if render_mode == "png":
                out_path = os.path.join(output_subfolder, base_name + ".png")
                render_still(out_path)
            else:
                out_path = os.path.join(output_subfolder, base_name + ".mp4")
                animate_rotation(obj, frames)
                render_video(out_path)


if __name__ == "__main__":
    # For manual testing.
    here = Path(__file__).resolve().parent
    print("This module is intended to be imported by entrypoints.")

