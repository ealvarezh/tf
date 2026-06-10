import requests
from pathlib import Path

species = [
    "Felis catus",
    "Canis lupus familiaris",
    "Equus caballus",
    "Bos taurus",
    "Columba livia"
]
output_dir = rf"..\data\landing\ "

for specie in species:

    response = requests.get(
        "https://api.inaturalist.org/v1/observations",
        params={
            "taxon_name": specie,
            "photos": "true",
            "quality_grade": "research",
            "per_page": 2
        }
    )

    observations = response.json()["results"]

    for i, obs in enumerate(observations):

        photo_url = obs["photos"][0]["url"]
        photo_url = photo_url.replace("square", "large")

        img = requests.get(photo_url)

        filename = f"{specie}_{i}.jpg".replace(" ", "_")

        with open(output_dir[:-1] + filename, "wb") as f:
            f.write(img.content)

        print("Downloaded", filename)