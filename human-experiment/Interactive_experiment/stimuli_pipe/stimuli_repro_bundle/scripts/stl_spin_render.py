"""
Shared STL → rotating MP4 pipeline (Blender/bpy). Imported by entrypoint scripts at repo root.
"""
import bpy
import os
import hashlib
import colorsys
import shutil
import subprocess
import tempfile
from math import radians
from pathlib import Path
import bmesh
from mathutils import Vector

# Blender 4.x: STL is built-in (wm.stl_import), not the removed io_mesh_stl add-on.

frames = 120
fps = 30
resolution = (1024, 1024)
rotation_axis = "Z"
animation_length_seconds = 4
degrees_to_rotate = 360
output_format = "MPEG4"
video_codec = "H264"
file_format = "FFMPEG"
cycles_samples = 96
cycles_use_denoising = True
cycles_denoiser = "OPENIMAGEDENOISE"
ffmpeg_constant_rate_factor = "MEDIUM"
ffmpeg_preset = "GOOD"
ffmpeg_video_bitrate_kbps = 12000


def _set_video_output_mode() -> None:
    """Configure ffmpeg intent; Blender may still render PNG sequence."""
    render = bpy.context.scene.render
    try:
        render.file_format = file_format
    except Exception:
        pass
    try:
        render.image_settings.file_format = file_format
    except Exception:
        pass
    render.ffmpeg.format = output_format
    render.ffmpeg.codec = video_codec
    render.ffmpeg.constant_rate_factor = ffmpeg_constant_rate_factor
    try:
        render.ffmpeg.ffmpeg_preset = ffmpeg_preset
    except Exception:
        pass
    try:
        render.ffmpeg.gopsize = fps
    except Exception:
        pass
    try:
        render.ffmpeg.video_bitrate = ffmpeg_video_bitrate_kbps
    except Exception:
        pass
    # Helpful runtime signal when debugging format mismatches.
    print(f"[stl_spin_render] video mode: format={output_format} codec={video_codec}")


def _set_still_output_mode() -> None:
    """Set output format to PNG still image."""
    render = bpy.context.scene.render
    try:
        render.file_format = "PNG"
    except Exception:
        pass
    try:
        render.image_settings.file_format = "PNG"
    except Exception:
        pass


def _apply_render_settings():
    render = bpy.context.scene.render
    _set_still_output_mode()
    render.fps = fps
    render.resolution_x = resolution[0]
    render.resolution_y = resolution[1]
    render.resolution_percentage = 100
    render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = cycles_samples
    bpy.context.scene.cycles.use_denoising = cycles_use_denoising
    try:
        bpy.context.scene.cycles.denoiser = cycles_denoiser
    except Exception:
        pass
    # Slightly reduce exposure to avoid pushing colors into near-white.
    bpy.context.scene.view_settings.exposure = 0.85
    # Keep tone mapping simple/consistent across machines.
    try:
        bpy.context.scene.view_settings.view_transform = "Standard"
        bpy.context.scene.view_settings.look = "None"
    except Exception:
        pass


_apply_render_settings()


def clear_scene():
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


def center_and_scale_object(obj, target_size=2.0):
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


def setup_scene(obj, object_size, material_mode: str = "flat", material_seed: int = 0):
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

    setup_lighting(object_size)
    setup_world_background()
    if material_mode == "textured":
        apply_material_textured(obj, material_seed)
    else:
        apply_material_flat(obj)


def setup_lighting(object_size):
    distance = object_size * 4
    main_light_data = bpy.data.lights.new(name="MainLight", type="SUN")
    main_light_data.energy = 10
    main_light = bpy.data.objects.new(name="MainLight", object_data=main_light_data)
    bpy.context.collection.objects.link(main_light)
    main_light.location = (distance, -distance, distance)
    main_light.rotation_euler = (radians(45), 0, radians(45))
    fill_light_data = bpy.data.lights.new(name="FillLight", type="SUN")
    fill_light_data.energy = 8
    fill_light = bpy.data.objects.new(name="FillLight", object_data=fill_light_data)
    bpy.context.collection.objects.link(fill_light)
    fill_light.location = (-distance, distance, distance * 0.5)
    fill_light.rotation_euler = (radians(30), 0, radians(-135))
    top_light_data = bpy.data.lights.new(name="TopLight", type="SUN")
    top_light_data.energy = 6
    top_light = bpy.data.objects.new(name="TopLight", object_data=top_light_data)
    bpy.context.collection.objects.link(top_light)
    top_light.location = (0, 0, distance * 2)
    top_light.rotation_euler = (0, 0, 0)
    front_light_data = bpy.data.lights.new(name="FrontLight", type="SUN")
    front_light_data.energy = 5
    front_light = bpy.data.objects.new(name="FrontLight", object_data=front_light_data)
    bpy.context.collection.objects.link(front_light)
    front_light.location = (0, -distance, 0)
    front_light.rotation_euler = (radians(90), 0, 0)


def setup_world_background():
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    bg_node = nodes.new(type="ShaderNodeBackground")
    # Dark neutral background so saturated palette colors remain readable.
    bg_node.inputs[0].default_value = (0.02, 0.02, 0.02, 1)
    bg_node.inputs[1].default_value = 0.6
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    world.node_tree.links.new(bg_node.outputs[0], output_node.inputs[0])


def _stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _seed_to_hsv(seed: int):
    hue = (seed % 360) / 360.0
    # Keep colors vivid so they remain obvious under lighting/compression.
    sat = 0.82 + (((seed >> 8) % 18) / 100.0)  # ~0.82..0.99
    val = 0.78 + (((seed >> 16) % 22) / 100.0)  # ~0.78..1.00
    sat = min(1.0, sat)
    val = min(1.0, val)
    return hue, sat, val


def _hsv_to_rgba(h: float, s: float, v: float):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b, 1.0)


def _palette_color(seed: int):
    """Return two vivid palette colors (2-stop ramps), no pure white."""
    palette_hsv = [
        # Use slightly reduced V so colors don't clip to white.
        (0.02, 0.95, 0.85),  # red
        (0.14, 0.95, 0.85),  # yellow
        (0.30, 0.95, 0.85),  # green
        (0.52, 0.95, 0.85),  # cyan
        (0.72, 0.95, 0.85),  # blue
        (0.88, 0.95, 0.85),  # magenta
    ]
    idx = seed % len(palette_hsv)
    c_vivid = palette_hsv[idx]
    # Pick another vivid color far enough in the palette to be distinct.
    c_vivid_2 = palette_hsv[(idx + 3) % len(palette_hsv)]
    return (
        _hsv_to_rgba(c_vivid[0], c_vivid[1], c_vivid[2]),
        _hsv_to_rgba(c_vivid_2[0], c_vivid_2[1], c_vivid_2[2]),
    )


def apply_material_flat(obj):
    mat = bpy.data.materials.new(name="ObjectMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.6
    try:
        principled.inputs["Specular IOR Level"].default_value = 1.0
    except KeyError:
        try:
            principled.inputs["Specular"].default_value = 1.0
        except KeyError:
            pass
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(principled.outputs[0], output.inputs[0])
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def apply_material_textured(obj, seed: int):
    mat = bpy.data.materials.new(name="ObjectMaterialTextured")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Metallic"].default_value = 0.0

    # Keep the surface matte; drive visible color via Base Color (not emission).
    principled.inputs["Roughness"].default_value = 0.45
    # Small emission helps colors survive lighting/shadows without turning everything white.
    principled.inputs["Emission Strength"].default_value = 2.0
    try:
        principled.inputs["Specular IOR Level"].default_value = 1.0
    except KeyError:
        try:
            principled.inputs["Specular"].default_value = 1.0
        except KeyError:
            pass

    # Noise-driven blocky color ramp + a subtle normal bump.
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    noise = nodes.new(type="ShaderNodeTexNoise")
    ramp = nodes.new(type="ShaderNodeValToRGB")
    bump = nodes.new(type="ShaderNodeBump")
    mult = nodes.new(type="ShaderNodeMath")
    mult.operation = "MULTIPLY"
    flo = nodes.new(type="ShaderNodeMath")
    flo.operation = "FLOOR"
    div = nodes.new(type="ShaderNodeMath")
    div.operation = "DIVIDE"
    output = nodes.new(type="ShaderNodeOutputMaterial")

    # Deterministic palette: vivid block color + another vivid palette color.
    c_vivid, c_vivid_2 = _palette_color(seed)
    ramp.color_ramp.elements[0].color = c_vivid
    ramp.color_ramp.elements[1].color = c_vivid_2

    # Posterize noise into a few discrete blocks.
    levels = 4
    mult.inputs["Value_001"].default_value = float(levels)
    div.inputs["Value_001"].default_value = float(levels - 1 if levels > 1 else 1)

    # Noise settings.
    tex_scale = 22.0 + (seed % 9)  # controls block size
    noise.inputs["Scale"].default_value = tex_scale
    bump.inputs["Strength"].default_value = 0.01  # keep bump very subtle
    try:
        bump.inputs["Distance"].default_value = 0.06
    except KeyError:
        pass

    # Best-effort for optional inputs (names vary a bit between builds).
    try:
        noise.inputs["Detail"].default_value = 7.0 + ((seed >> 3) % 6)
    except KeyError:
        pass
    try:
        noise.inputs["Roughness"].default_value = 0.45 + (((seed >> 10) % 20) / 100.0)
    except KeyError:
        pass

    # Layout nodes.
    tex_coord.location = (-900, 0)
    mapping.location = (-650, 0)
    noise.location = (-350, 0)
    mult.location = (-200, 0)
    flo.location = (-50, 0)
    div.location = (80, 0)
    ramp.location = (220, 0)
    bump.location = (-350, -250)
    principled.location = (480, 0)
    output.location = (760, 0)

    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], mult.inputs["Value"])
    links.new(mult.outputs["Value"], flo.inputs["Value"])
    links.new(flo.outputs["Value"], div.inputs["Value"])
    links.new(div.outputs["Value"], ramp.inputs["Fac"])

    # Drive Base Color (and Emission) from the ramp.
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])
    links.new(ramp.outputs["Color"], principled.inputs["Emission Color"])

    # Blender 4.5 ShaderNodeBump uses "Normal" (not "Vector") as the input socket.
    links.new(mapping.outputs["Vector"], bump.inputs["Normal"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])

    links.new(principled.outputs[0], output.inputs[0])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def animate_rotation(obj, total_frames):
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


def render_video(output_path):
    scene = bpy.context.scene
    render = scene.render

    frame_tmp_dir = Path(tempfile.mkdtemp(prefix="alice_frames_"))
    frame_prefix = frame_tmp_dir / "frame_"
    frame_pattern = str(frame_tmp_dir / "frame_%04d.png")
    output_path = str(Path(output_path))

    try:
        _set_still_output_mode()
        scene.frame_start = 1
        scene.frame_end = frames
        render.filepath = str(frame_prefix)
        bpy.ops.render.render(animation=True)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "medium",
            output_path,
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"[stl_spin_render] encoded mp4: {output_path}")
    finally:
        shutil.rmtree(frame_tmp_dir, ignore_errors=True)


def render_still(output_path):
    _set_still_output_mode()
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def main(input_folder: str, output_folder: str, material_mode: str = "flat"):
    """Walk input_folder for .stl (any depth), mirror relative paths under output_folder."""
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
            output_path = os.path.join(output_subfolder, base_name + ".mp4")
            print(f"Processing {stl_path}")
            clear_scene()
            # bpy.ops.import_mesh.stl(filepath=stl_path) not used because it's not built-in in Blender 4.x
            bpy.ops.wm.stl_import(filepath=stl_path)
            obj = bpy.context.selected_objects[0]
            object_size = center_and_scale_object(obj, target_size=2.0)
            seed = _stable_int(stl_path)
            setup_scene(obj, object_size, material_mode=material_mode, material_seed=seed)
            animate_rotation(obj, frames)
            render_video(output_path)
            print(f"✅ Rendered: {output_path}")
    print("✅ All STL files processed.")
