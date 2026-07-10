"""ALICE STL batch -> MP4 using fixed_blender_centering scene + textured materials.

Default pipeline (v2, from updated gpt_suggestions.txt) targets dark/black back sides by:
  - brighter neutral world (less dark reflection in glossy materials)
  - recalculated / cleared STL normals + smooth shading + optional Weighted Normal
  - soft area-only lighting from multiple sides
  - after overlay: clamp Principled toward matte (roughness/specular)

Legacy pipeline (previous sun + area rig, darker world):
    ALICE_LEGACY_SCENE=1

Optional:
    ALICE_ONLY_STEMS=15,1,2
    ALICE_QUICK_FRAMES=48 ALICE_QUICK_FPS=24 ALICE_QUICK_SAMPLES=64
    ALICE_QUALITY_PROFILE=balanced  # default: balanced
    ALICE_RENDER_RES=1024           # square resolution override
    ALICE_RENDER_SAMPLES=96         # Cycles samples override
    ALICE_HERO_RES=1400             # hero PNG square resolution override
    ALICE_HERO_SAMPLES=128          # hero PNG samples override
    ALICE_HERO_ANGLE_DEG=0          # hero PNG object Z rotation
    ALICE_HERO_TILT_DEG=0           # hero PNG object X tilt
    ALICE_HERO_CAM_AZIMUTH_DEG=-45  # hero camera horizontal angle
    ALICE_HERO_CAM_ELEVATION_DEG=15 # hero camera vertical angle
    ALICE_HERO_CAM_LENS_MM=85       # hero camera lens (flatter perspective)
    ALICE_HERO_BG=black             # black, darkgray, or transparent
    ALICE_STIMULUS_MODE=B_controlled_simple  # B_controlled_simple, A_auto_contrast, off
    ALICE_STIMULUS_RES=1024         # controlled still resolution for stimuli
    ALICE_STIMULUS_SAMPLES=128      # controlled still samples for stimuli
    ALICE_STIMULUS_MATCH_REFERENCE=1 # match test-object side/angle to images/<id>.PNG
    ALICE_STIMULUS_MATCH_STEP_DEG=30 # Z-angle search step for reference matching
    ALICE_SUBSURF=1          mild Subdivision (slower; smoother STLs)
"""

import os
import sys
import csv
import tempfile
from pathlib import Path

_scripts = Path(__file__).resolve().parent / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import stl_spin_render as scene
import stl_material_overlay_render as mats


_PROJECT = Path(__file__).resolve().parent
_ALICE = _PROJECT / "data" / "ALICE_stl_(Xu & Sandhofer, 2024)"
_ALICE_STL = _ALICE / "stl"
_OUT = _ALICE / "animations_fixed_centering_texture"
_OUT_HERO = _ALICE / "animations_fixed_centering_texture_hero_png"
_OUT_STIM_B = _ALICE / "stimuli_B_controlled_simple"
_OUT_STIM_A = _ALICE / "stimuli_A_auto_contrast"


def _iter_stls(base: Path):
    for root, dirs, files in os.walk(base):
        for filename in files:
            if filename.lower().endswith(".stl"):
                yield Path(root) / filename


def _selected_stems_from_env():
    raw = os.environ.get("ALICE_ONLY_STEMS", "").strip()
    if not raw:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def _selected_stimulus_mode_from_env():
    raw = os.environ.get("ALICE_STIMULUS_MODE", "B_controlled_simple").strip()
    if not raw:
        return "B_controlled_simple"
    lowered = raw.lower()
    if lowered in {"off", "none", "media"}:
        return None
    if raw in {"B_controlled_simple", "A_auto_contrast"}:
        return raw
    print(f"WARNING: unknown ALICE_STIMULUS_MODE={raw!r}; falling back to B_controlled_simple")
    return "B_controlled_simple"


def _stimulus_output_root(mode: str) -> Path:
    return _OUT_STIM_B if mode == "B_controlled_simple" else _OUT_STIM_A


def _reference_image_for_stem(stem: str):
    p_upper = _ALICE / "images" / f"{stem}.PNG"
    if p_upper.exists():
        return p_upper
    p_lower = _ALICE / "images" / f"{stem}.png"
    if p_lower.exists():
        return p_lower
    return None


def _parse_positive_int(raw: str, default: int) -> int:
    try:
        value = int(raw)
    except Exception:
        return default
    return value if value > 0 else default


def _apply_quality_profile_from_env() -> None:
    """Balanced defaults with env overrides for clarity-focused direct MP4."""
    profile = os.environ.get("ALICE_QUALITY_PROFILE", "balanced").strip().lower() or "balanced"

    # Base presets. "balanced" is the default; "quick" remains available when needed.
    if profile == "quick":
        scene.resolution = (640, 640)
        scene.cycles_samples = 40
        scene.cycles_use_denoising = True
    elif profile == "high":
        scene.resolution = (1280, 1280)
        scene.cycles_samples = 144
        scene.cycles_use_denoising = True
    else:
        scene.resolution = (1024, 1024)
        scene.cycles_samples = 96
        scene.cycles_use_denoising = True

    res_raw = os.environ.get("ALICE_RENDER_RES", "").strip()
    samples_raw = os.environ.get("ALICE_RENDER_SAMPLES", "").strip()
    denoise_raw = os.environ.get("ALICE_RENDER_DENOISE", "").strip().lower()

    if res_raw:
        res = _parse_positive_int(res_raw, scene.resolution[0])
        scene.resolution = (res, res)
    if samples_raw:
        scene.cycles_samples = _parse_positive_int(samples_raw, scene.cycles_samples)
    if denoise_raw in {"0", "false", "no", "off"}:
        scene.cycles_use_denoising = False
    elif denoise_raw in {"1", "true", "yes", "on"}:
        scene.cycles_use_denoising = True


def _apply_quick_overrides_from_env() -> None:
    frames_raw = os.environ.get("ALICE_QUICK_FRAMES", "").strip()
    fps_raw = os.environ.get("ALICE_QUICK_FPS", "").strip()
    samples_raw = os.environ.get("ALICE_QUICK_SAMPLES", "").strip()
    if frames_raw:
        scene.frames = int(frames_raw)
    if fps_raw:
        scene.fps = int(fps_raw)
    scene._apply_render_settings()
    if samples_raw:
        try:
            scene.bpy.context.scene.cycles.samples = int(samples_raw)
        except Exception:
            pass


def _legacy_set_light_gray_background() -> None:
    world = scene.bpy.context.scene.world
    if world is None:
        world = scene.bpy.data.worlds.new("World")
        scene.bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new(type="ShaderNodeBackground")
    out = nodes.new(type="ShaderNodeOutputWorld")
    bg.inputs[0].default_value = (0.10, 0.10, 0.10, 1.0)
    bg.inputs[1].default_value = 0.8
    links.new(bg.outputs[0], out.inputs[0])


def _set_dark_gray_background() -> None:
    world = scene.bpy.context.scene.world
    if world is None:
        world = scene.bpy.data.worlds.new("World")
        scene.bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new(type="ShaderNodeBackground")
    out = nodes.new(type="ShaderNodeOutputWorld")
    # Dark gray studio background gives stronger silhouette contrast.
    bg.inputs[0].default_value = (0.08, 0.08, 0.08, 1.0)
    bg.inputs[1].default_value = 0.55
    links.new(bg.outputs[0], out.inputs[0])


def _set_balanced_color_management(*, exposure: float) -> None:
    view = scene.bpy.context.scene.view_settings
    try:
        view.view_transform = "Filmic"
        view.look = "None"
    except Exception:
        pass
    view.exposure = exposure
    view.gamma = 1.0


def _fix_mesh_normals_and_shading(obj, *, add_weighted_normal: bool) -> None:
    bpy = scene.bpy
    ctx = bpy.context

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    ctx.view_layer.objects.active = obj

    try:
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    except Exception:
        pass

    if obj.type != "MESH":
        return

    try:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        # STL repair pass: merge duplicate vertices and patch open boundaries.
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        try:
            bpy.ops.mesh.fill_holes(sides=0)
        except Exception:
            pass
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
        # Recalculate twice for stubborn imported meshes.
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

    try:
        for poly in obj.data.polygons:
            poly.use_smooth = True
    except Exception:
        pass

    try:
        if hasattr(obj.data, "use_auto_smooth"):
            obj.data.use_auto_smooth = True
            deg = float(os.environ.get("ALICE_AUTO_SMOOTH_DEG", "35"))
            obj.data.auto_smooth_angle = scene.radians(deg)
    except Exception:
        pass

    if add_weighted_normal:
        try:
            if not any(m.type == "WEIGHTED_NORMAL" for m in obj.modifiers):
                mod = obj.modifiers.new(name="WeightedNormal_ALICE", type="WEIGHTED_NORMAL")
                mod.keep_sharp = True
        except Exception:
            pass


def _optional_subdivision(obj) -> None:
    if os.environ.get("ALICE_SUBSURF", "").strip() != "1":
        return
    try:
        if any(m.type == "SUBSURF" for m in obj.modifiers):
            return
        mod = obj.modifiers.new(name="Subdivision_ALICE", type="SUBSURF")
        mod.levels = 1
        mod.render_levels = 1
    except Exception:
        pass


def _optional_voxel_remesh(obj) -> None:
    # Robust STL cleanup path for broken topology/normals.
    if os.environ.get("ALICE_VOXEL_REMESH", "").strip() != "1":
        return
    try:
        mod = obj.modifiers.new(name="VoxelRemesh_ALICE", type="REMESH")
        mod.mode = "VOXEL"
        # Default gives detail without exploding runtime.
        mod.voxel_size = float(os.environ.get("ALICE_VOXEL_SIZE", "0.04"))
        mod.adaptivity = 0.0
        scene.bpy.context.view_layer.objects.active = obj
        scene.bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception:
        pass


def _prepare_imported_object(obj) -> None:
    """Normals + smooth first; optional Subsurf; then weighted normal once."""
    _fix_mesh_normals_and_shading(obj, add_weighted_normal=False)
    _optional_voxel_remesh(obj)
    _optional_subdivision(obj)
    _fix_mesh_normals_and_shading(obj, add_weighted_normal=True)


def _rebalance_lighting_soft(object_size: float) -> None:
    bpy = scene.bpy
    for o in list(bpy.data.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o, do_unlink=True)

    d = object_size * 4.0
    area_specs = [
        ("FrontKey", 1200.0, (0, -d * 1.45, object_size * 0.80), (90, 0, 0), object_size * 3.2),
        ("BackFill", 950.0, (0, d * 1.45, object_size * 0.80), (90, 0, 180), object_size * 3.2),
        ("LeftFill", 750.0, (-d * 1.25, 0, object_size * 0.70), (90, 0, -90), object_size * 2.9),
        ("RightFill", 750.0, (d * 1.25, 0, object_size * 0.70), (90, 0, 90), object_size * 2.9),
        ("TopSoft", 850.0, (0, 0, d * 1.70), (180, 0, 0), object_size * 4.0),
    ]
    for name, energy, location, rotation_deg, size in area_specs:
        light_data = bpy.data.lights.new(name=name, type="AREA")
        light_data.energy = energy
        light_data.shape = "SQUARE"
        light_data.size = size
        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = location
        light_obj.rotation_euler = tuple(scene.radians(v) for v in rotation_deg)


def _rebalance_lighting_legacy(object_size: float) -> None:
    for obj in list(scene.bpy.data.objects):
        if obj.type == "LIGHT":
            scene.bpy.data.objects.remove(obj, do_unlink=True)

    distance = object_size * 4.0
    light_specs = [
        ("MainKey", "SUN", 3.8, (distance, -distance, distance * 0.9), (52, 0, 40)),
        ("BackKey", "SUN", 3.4, (-distance, distance, distance * 0.85), (55, 0, -140)),
        ("TopSoft", "SUN", 2.2, (0, 0, distance * 2.2), (0, 0, 0)),
    ]
    for name, light_type, energy, location, rotation_deg in light_specs:
        light_data = scene.bpy.data.lights.new(name=name, type=light_type)
        light_data.energy = energy
        light_obj = scene.bpy.data.objects.new(name=name, object_data=light_data)
        scene.bpy.context.collection.objects.link(light_obj)
        light_obj.location = location
        light_obj.rotation_euler = tuple(scene.radians(v) for v in rotation_deg)

    area_specs = [
        ("FrontAreaFill", 280.0, (0, -distance * 1.35, object_size * 0.3), (90, 0, 0), object_size * 2.8),
        ("BackAreaFill", 260.0, (0, distance * 1.35, object_size * 0.35), (90, 0, 180), object_size * 2.8),
        ("LeftAreaFill", 220.0, (-distance * 1.25, 0, object_size * 0.35), (90, 0, -90), object_size * 2.6),
        ("RightAreaFill", 220.0, (distance * 1.25, 0, object_size * 0.35), (90, 0, 90), object_size * 2.6),
    ]
    for name, energy, location, rotation_deg, size in area_specs:
        light_data = scene.bpy.data.lights.new(name=name, type="AREA")
        light_data.energy = energy
        light_data.shape = "SQUARE"
        light_data.size = size
        light_obj = scene.bpy.data.objects.new(name=name, object_data=light_data)
        scene.bpy.context.collection.objects.link(light_obj)
        light_obj.location = location
        light_obj.rotation_euler = tuple(scene.radians(v) for v in rotation_deg)


def _make_material_less_black_reflective(obj) -> None:
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            links = mat.node_tree.links
            if "Metallic" in node.inputs:
                node.inputs["Metallic"].default_value = 0.0
            if "Roughness" in node.inputs:
                # Force matte response to avoid dark rear reflections.
                node.inputs["Roughness"].default_value = 0.8
            if "Specular IOR Level" in node.inputs:
                node.inputs["Specular IOR Level"].default_value = 0.2
            elif "Specular" in node.inputs:
                node.inputs["Specular"].default_value = 0.2
            if "Sheen Tint" in node.inputs:
                try:
                    node.inputs["Sheen Tint"].default_value = 0.0
                except Exception:
                    pass
            # Final shadow-lift fallback: low emission from base color prevents
            # pure-black back sides on problematic STL normals/topology.
            if "Emission Strength" in node.inputs:
                node.inputs["Emission Strength"].default_value = 0.22
            if "Emission Color" in node.inputs and "Base Color" in node.inputs:
                try:
                    base_input = node.inputs["Base Color"]
                    emis_input = node.inputs["Emission Color"]
                    if base_input.is_linked:
                        src_socket = base_input.links[0].from_socket
                        links.new(src_socket, emis_input)
                    else:
                        emis_input.default_value = base_input.default_value
                except Exception:
                    pass


def _render_hero_png(obj, out_path: Path) -> None:
    render = scene.bpy.context.scene.render
    cycles = scene.bpy.context.scene.cycles
    cam = scene.bpy.context.scene.camera

    prev_res_x = int(render.resolution_x)
    prev_res_y = int(render.resolution_y)
    prev_samples = int(cycles.samples)
    prev_rot = tuple(obj.rotation_euler)
    prev_film_transparent = bool(getattr(render, "film_transparent", False))
    prev_lens = float(cam.data.lens) if cam and cam.type == "CAMERA" else None
    prev_cam_loc = tuple(cam.location) if cam and cam.type == "CAMERA" else None
    prev_cam_rot = tuple(cam.rotation_euler) if cam and cam.type == "CAMERA" else None

    hero_res = _parse_positive_int(os.environ.get("ALICE_HERO_RES", "").strip(), max(prev_res_x, prev_res_y))
    hero_samples = _parse_positive_int(os.environ.get("ALICE_HERO_SAMPLES", "").strip(), max(prev_samples, 128))
    hero_angle = float(os.environ.get("ALICE_HERO_ANGLE_DEG", "0"))
    hero_tilt = float(os.environ.get("ALICE_HERO_TILT_DEG", "0"))
    hero_cam_azimuth = float(os.environ.get("ALICE_HERO_CAM_AZIMUTH_DEG", "-45"))
    hero_cam_elevation = float(os.environ.get("ALICE_HERO_CAM_ELEVATION_DEG", "15"))
    hero_cam_lens = float(os.environ.get("ALICE_HERO_CAM_LENS_MM", "85"))
    hero_bg = os.environ.get("ALICE_HERO_BG", "black").strip().lower()

    render.resolution_x = hero_res
    render.resolution_y = hero_res
    cycles.samples = hero_samples

    if hero_bg == "transparent":
        render.film_transparent = True
    else:
        render.film_transparent = False
        world = scene.bpy.context.scene.world
        if world is not None:
            world.use_nodes = True
            nodes = world.node_tree.nodes
            links = world.node_tree.links
            nodes.clear()
            bg = nodes.new(type="ShaderNodeBackground")
            out = nodes.new(type="ShaderNodeOutputWorld")
            if hero_bg == "darkgray":
                bg.inputs[0].default_value = (0.08, 0.08, 0.08, 1.0)
                bg.inputs[1].default_value = 0.55
            else:
                bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
                bg.inputs[1].default_value = 1.0
            links.new(bg.outputs[0], out.inputs[0])

    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (scene.radians(hero_tilt), 0.0, scene.radians(hero_angle))

    if cam and cam.type == "CAMERA":
        import math

        az = math.radians(hero_cam_azimuth)
        el = math.radians(hero_cam_elevation)
        distance = 5.6
        cam.location = (
            distance * math.cos(el) * math.cos(az),
            distance * math.cos(el) * math.sin(az),
            distance * math.sin(el),
        )
        direction = scene.Vector((0, 0, 0)) - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = hero_cam_lens

    scene.bpy.context.scene.frame_set(1)

    scene.render_still(str(out_path))

    render.resolution_x = prev_res_x
    render.resolution_y = prev_res_y
    cycles.samples = prev_samples
    render.film_transparent = prev_film_transparent
    obj.rotation_euler = prev_rot
    if cam and cam.type == "CAMERA":
        if prev_cam_loc is not None:
            cam.location = prev_cam_loc
        if prev_cam_rot is not None:
            cam.rotation_euler = prev_cam_rot
        if prev_lens is not None:
            cam.data.lens = prev_lens


def _configure_stimulus_render_controls() -> None:
    render = scene.bpy.context.scene.render
    cycles = scene.bpy.context.scene.cycles
    stimulus_res = _parse_positive_int(os.environ.get("ALICE_STIMULUS_RES", "").strip(), 1024)
    stimulus_samples = _parse_positive_int(os.environ.get("ALICE_STIMULUS_SAMPLES", "").strip(), 128)
    render.resolution_x = stimulus_res
    render.resolution_y = stimulus_res
    cycles.samples = stimulus_samples


def _resize_mask(mask, src_w: int, src_h: int, dst_w: int, dst_h: int):
    if src_w == dst_w and src_h == dst_h:
        return mask
    out = [False] * (dst_w * dst_h)
    for y in range(dst_h):
        sy = int((y * src_h) / dst_h)
        row_off = y * dst_w
        src_row_off = sy * src_w
        for x in range(dst_w):
            sx = int((x * src_w) / dst_w)
            out[row_off + x] = mask[src_row_off + sx]
    return out


def _normalize_mask(mask, src_w: int, src_h: int, out_size: int = 128):
    xs = []
    ys = []
    for idx, val in enumerate(mask):
        if not val:
            continue
        y = idx // src_w
        x = idx - (y * src_w)
        xs.append(x)
        ys.append(y)
    if not xs:
        return [False] * (out_size * out_size)

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    bw = max(1, max_x - min_x + 1)
    bh = max(1, max_y - min_y + 1)

    out = [False] * (out_size * out_size)
    for oy in range(out_size):
        sy = min_y + int((oy * bh) / out_size)
        sy = min(max(sy, 0), src_h - 1)
        src_row = sy * src_w
        out_row = oy * out_size
        for ox in range(out_size):
            sx = min_x + int((ox * bw) / out_size)
            sx = min(max(sx, 0), src_w - 1)
            out[out_row + ox] = mask[src_row + sx]
    return out


def _load_mask_from_png(path: Path):
    img = scene.bpy.data.images.load(str(path), check_existing=False)
    try:
        w = int(img.size[0])
        h = int(img.size[1])
        px = list(img.pixels)
    finally:
        scene.bpy.data.images.remove(img)

    total = w * h
    alpha = [px[(i * 4) + 3] for i in range(total)]
    max_a = max(alpha) if alpha else 0.0
    min_a = min(alpha) if alpha else 0.0
    if max_a > 0.1 and min_a < 0.99:
        return [a > 0.08 for a in alpha], w, h

    luma = []
    for i in range(total):
        r = px[(i * 4) + 0]
        g = px[(i * 4) + 1]
        b = px[(i * 4) + 2]
        luma.append((0.2126 * r) + (0.7152 * g) + (0.0722 * b))
    corners = [0, w - 1, (h - 1) * w, h * w - 1]
    bg = sum(luma[i] for i in corners) / 4.0
    return [abs(v - bg) > 0.08 for v in luma], w, h


def _mask_iou(mask_a, mask_b) -> float:
    inter = 0
    union = 0
    for a, b in zip(mask_a, mask_b):
        if a and b:
            inter += 1
        if a or b:
            union += 1
    return float(inter) / float(union) if union else 0.0


def _match_object_z_to_reference(obj, stem: str) -> None:
    raw = os.environ.get("ALICE_STIMULUS_MATCH_REFERENCE", "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return

    ref_img = _reference_image_for_stem(stem)
    if ref_img is None:
        print(f"WARNING: no reference image found for stem {stem}; skipping pose match.")
        return

    render = scene.bpy.context.scene.render
    cycles = scene.bpy.context.scene.cycles
    prev_res_x = int(render.resolution_x)
    prev_res_y = int(render.resolution_y)
    prev_samples = int(cycles.samples)

    probe_w = 256
    probe_h = 256
    render.resolution_x = probe_w
    render.resolution_y = probe_h
    cycles.samples = 8

    ref_mask, ref_w, ref_h = _load_mask_from_png(ref_img)
    ref_mask = _normalize_mask(ref_mask, ref_w, ref_h, out_size=128)

    step = _parse_positive_int(os.environ.get("ALICE_STIMULUS_MATCH_STEP_DEG", "").strip(), 30)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"stim_match_{stem}_"))
    tmp_png = tmp_dir / "probe.png"

    best_deg = 0.0
    best_score = -1.0
    try:
        for deg in range(0, 360, step):
            obj.rotation_mode = "XYZ"
            obj.rotation_euler = (0.0, 0.0, scene.radians(float(deg)))
            scene.render_still(str(tmp_png))
            probe_mask, probe_w2, probe_h2 = _load_mask_from_png(tmp_png)
            probe_mask = _normalize_mask(probe_mask, probe_w2, probe_h2, out_size=128)
            score = _mask_iou(ref_mask, probe_mask)
            if score > best_score:
                best_score = score
                best_deg = float(deg)
    finally:
        try:
            tmp_png.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            tmp_dir.rmdir()
        except Exception:
            pass

    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, scene.radians(best_deg))
    print(f"Pose-matched stem {stem}: z={best_deg:.1f} deg, IoU={best_score:.4f}")

    render.resolution_x = prev_res_x
    render.resolution_y = prev_res_y
    cycles.samples = prev_samples


def _render_stimulus_versions(obj, seed: int, *, stimulus_mode: str, output_root: Path, stem: str) -> None:
    stem_dir = output_root / stem
    os.makedirs(stem_dir, exist_ok=True)
    records = []
    _match_object_z_to_reference(obj, stem)
    for variant_idx in (1, 2):
        mats.apply_material_stimulus_variant(obj, seed, stimulus_mode=stimulus_mode, variant_index=variant_idx)
        out_path = stem_dir / f"version_{variant_idx}.png"
        scene.render_still(str(out_path))
        records.append(
            {
                "stimulus_mode": stimulus_mode,
                "shape_id": stem,
                "pair_id": stem,
                "texture_version": f"version_{variant_idx}",
                "variant_index": str(variant_idx),
                "relative_path": str(out_path.relative_to(_ALICE)),
                "absolute_path": str(out_path),
            }
        )
    return records


def _write_stimulus_manifest(output_root: Path, records) -> None:
    if not records:
        return
    manifest_path = output_root / "manifest.csv"
    fields = [
        "stimulus_mode",
        "shape_id",
        "pair_id",
        "texture_version",
        "variant_index",
        "relative_path",
        "absolute_path",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print(f"Wrote manifest: {manifest_path}")


def _render_all() -> None:
    stimulus_mode = _selected_stimulus_mode_from_env()
    output_root = _stimulus_output_root(stimulus_mode) if stimulus_mode else None
    if output_root is None:
        os.makedirs(_OUT, exist_ok=True)
        os.makedirs(_OUT_HERO, exist_ok=True)
    else:
        os.makedirs(output_root, exist_ok=True)

    _apply_quality_profile_from_env()
    _apply_quick_overrides_from_env()
    selected_stems = _selected_stems_from_env()
    legacy = os.environ.get("ALICE_LEGACY_SCENE", "").strip() == "1"
    stimulus_records = []

    for stl_path in sorted(_iter_stls(_ALICE_STL), key=lambda p: p.name.lower()):
        stem = stl_path.stem
        if selected_stems is not None and stem not in selected_stems:
            continue

        if output_root is None:
            out_path = _OUT / f"{stem}.mp4"
            hero_png_path = _OUT_HERO / f"{stem}.png"
            print(f"Rendering {stl_path} -> {out_path} (+ hero: {hero_png_path})")
        else:
            print(f"Rendering stimuli {stl_path} -> {output_root / stem}")

        scene.clear_scene()
        scene.bpy.ops.wm.stl_import(filepath=str(stl_path))
        selected = list(scene.bpy.context.selected_objects)
        if not selected:
            print(f"WARNING: nothing imported from {stl_path}")
            continue

        obj = selected[0]
        object_size = scene.center_and_scale_object(obj, target_size=2.0)
        seed = scene._stable_int(str(stl_path))

        scene.setup_scene(obj, object_size, material_mode="flat", material_seed=seed)

        if legacy:
            _legacy_set_light_gray_background()
            _set_balanced_color_management(exposure=0.25)
            _rebalance_lighting_legacy(object_size)
            mats.apply_material_overlay(obj, seed, material_style="realistic")
        else:
            _prepare_imported_object(obj)
            _set_dark_gray_background()
            _set_balanced_color_management(exposure=0.20)
            _rebalance_lighting_soft(object_size)
            mats.apply_material_overlay(obj, seed, material_style="realistic")
            _make_material_less_black_reflective(obj)

        if output_root is not None:
            _configure_stimulus_render_controls()
            stimulus_records.extend(
                _render_stimulus_versions(
                    obj, seed, stimulus_mode=stimulus_mode, output_root=output_root, stem=stem
                )
            )
            continue

        _render_hero_png(obj, hero_png_path)
        scene.animate_rotation(obj, scene.frames)
        scene.render_video(str(out_path))

    if output_root is not None:
        _write_stimulus_manifest(output_root, stimulus_records)
        print(f"Done: stimulus set rendered ({stimulus_mode}) -> {output_root}")
    else:
        print("Done: fixed_blender_centering + texture overlay renders complete.")


if __name__ == "__main__":
    _render_all()
