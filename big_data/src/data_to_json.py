import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

output_dir = BASE_DIR / "data" / "curated"
input_dir  = BASE_DIR / "data" / "landing"

images = []
annotations = []
categories = []

species = [
    "Felis catus",
    "Canis lupus familiaris",
    "Equus caballus",
    "Bos taurus",
    "Columba livia"
]

img_id = 1

for cat_id, specie in enumerate(species, start=1):

    categories.append({
        "id": cat_id,
        "common_name": specie
    })

    files = sorted(input_dir.glob(f"{specie.replace(' ','_')}*.jpg"))

    for file in files:

        images.append({
            "id": img_id,
            "file_name": file.name
        })

        annotations.append({
            "image_id": img_id,
            "category_id": cat_id
        })

        img_id += 1

tiny_json = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(output_dir / "train.json", "w") as f:
    json.dump(tiny_json, f, indent=2)

with open(output_dir / "val.json", "w") as f:
    json.dump(tiny_json, f, indent=2)