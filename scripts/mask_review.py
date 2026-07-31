from pathlib import Path
import json
import tempfile
import numpy as np
from flask import Flask, send_file, request, jsonify
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "previous-lit-stimuli" / "geirhos_unaltered"
RMBG2_DIR = REPO_ROOT / "previous-lit-stimuli" / "geirhos_rmbg2" / "original_rmbg2"

# use debug images only to get the list of unique shapes
images = []
seen_shapes = set()

cue_root = DATA_DIR / "cue_conflict"

for img in sorted(cue_root.rglob("*.png")):
    shape_id = img.stem.split("-")[0]

    if shape_id in seen_shapes:
        continue

    seen_shapes.add(shape_id)
    images.append(img)

print("FOUND IMAGES:", len(images))
print(images[:5])

RESULTS_FILE = REPO_ROOT / "previous-lit-stimuli" / "rmbg2_reviews.json"

if RESULTS_FILE.exists():
    results = json.loads(RESULTS_FILE.read_text())
else:
    results = {}

app = Flask(__name__)


def make_preview(cue_path):

    shape_id = cue_path.stem.split("-")[0]

    shape = "".join(c for c in shape_id if c.isalpha())
    num = "".join(c for c in shape_id if c.isdigit())

    original_path = (
        DATA_DIR /
        "original" /
        shape /
        f"{shape}{num}.png"
    )

    mask_path = (
        RMBG2_DIR /
        shape /
        f"{shape}{num}.png"
    )

    cue = np.array(Image.open(cue_path).convert("RGB"))

    rgba = np.array(Image.open(mask_path).convert("RGBA"))

    alpha = rgba[:, :, 3]

    a = alpha.astype(np.float32) / 255.0
    a = a[..., None]

    white = np.full_like(cue, 255)

    result = (
        cue * a +
        white * (1 - a)
    ).astype(np.uint8)

    mask_img = np.stack([alpha] * 3, axis=2)

    original_rgb = np.array(
        Image.open(original_path).convert("RGB")
    )

    rembg_rgb = rgba[:, :, :3]

    tetrad = np.concatenate(
    [original_rgb, rembg_rgb, mask_img, result],
    axis=1,
    )

    temp = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    )

    Image.fromarray(tetrad).save(temp.name)

    return temp.name


@app.route("/")
def index():
    return f"""
<!DOCTYPE html>
<html>
<body style="text-align:center">

<h2 id="title"></h2>
<h3 id="progress"></h3>

<img id="image" style="max-width:90%;">

<br><br>

<button onclick="rate('good')">Good</button>
<button onclick="rate('medium')">Medium</button>
<button onclick="rate('poor')">Poor</button>
<button onclick="next()">Skip</button>


<script>

let i = 0;
let total = {len(images)};
let images = {json.dumps([str(x) for x in images])};

function show() {{
    updateImage();

    document.getElementById("title").innerHTML =
        "Image " + i;

    document.getElementById("progress").innerHTML =
        (i+1) + "/" + total;
}}


function updateImage() {{
    document.getElementById("image").src =
        "/preview?i=" + i;
}}


function next() {{
    i++;

    if (i < total) {{
        show();
    }}
}}


function rate(x) {{
    fetch("/rate", {{
        method:"POST",
        headers:{{
            "Content-Type":"application/json"
        }},
        body:JSON.stringify({{
            image: images[i],
            rating: x,
        }})
    }});

    next();
}}

show();

</script>

</body>
</html>
"""


@app.route("/preview")
def preview():
    i = int(request.args["i"])
    path = make_preview(images[i])

    return send_file(path)


@app.route("/rate", methods=["POST"])
def rate():
    data = request.json

    shape_id = Path(data["image"]).stem.split("-")[0]

    results[shape_id] = data["rating"]

    RESULTS_FILE.write_text(
        json.dumps(results, indent=2)
    )

    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(port=5001)