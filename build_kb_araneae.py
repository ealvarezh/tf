"""
build_kb_araneae.py
-------------------
Construye la knowledge base de arañas peruanas (GBIF Arachnida) para el pipeline VR-RAG.

Uso:
    python build_kb_araneae.py                     # construye todo
    python build_kb_araneae.py --max-images 5      # N imágenes por especie (default: 5)
    python build_kb_araneae.py --force-rebuild      # ignora caché y reescribe todo

Outputs en kb_araneae/:
    knowledge_base.json   → metadata + rutas locales de imágenes por especie
    images/<species_key>/ → imágenes descargadas
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ── Configuración ──────────────────────────────────────────────────────────────
PARQUET_DIR = Path(r"C:\Users\eah\Downloads\tesis\0059496-260519110011954\parquet")
KB_DIR      = Path(__file__).parent / "kb_araneae"
KB_JSON     = KB_DIR / "knowledge_base.json"
IMG_DIR     = KB_DIR / "images"

MAX_IMAGES_DEFAULT  = 5
GLOBAL_RECORDS_THRESHOLD = 100   # < este valor → especie subrepresentada
DOWNLOAD_TIMEOUT    = 10          # segundos por imagen
DOWNLOAD_DELAY      = 0.1         # segundos entre requests (cortesía al servidor)
# ──────────────────────────────────────────────────────────────────────────────


def load_parquet_data() -> pd.DataFrame:
    """
    Carga y cruza occurrence + multimedia.
    Filtra a Perú, orden Araneae, con imagen disponible y especie identificada.
    """
    print("Cargando occurrence.parquet...")
    occ = pd.read_parquet(
        PARQUET_DIR / "occurrence.parquet",
        columns=[
            "gbifID", "species", "genus", "family", "order",
            "countryCode", "decimalLatitude", "decimalLongitude",
            "year", "iucnRedListCategory", "vernacularName", "speciesKey",
        ],
    )

    print("Cargando multimedia.parquet...")
    mm = pd.read_parquet(
        PARQUET_DIR / "multimedia.parquet",
        columns=["gbifID", "identifier", "license"],
    )

    # Filtro principal: Perú, arañas, especie conocida
    mask = (
        (occ["countryCode"] == "PE") &
        (occ["order"] == "Araneae") &
        (occ["species"].notna())
    )
    pe = occ[mask].copy()

    # Flag de subrepresentación: registros globales de la especie
    global_counts = (
        occ[occ["order"] == "Araneae"]
        .groupby("species")
        .size()
        .rename("global_records")
    )
    pe = pe.join(global_counts, on="species")
    pe["subrepresented"] = pe["global_records"] < GLOBAL_RECORDS_THRESHOLD

    # Cruzar con imágenes
    mm_clean = mm[mm["identifier"].notna() & mm["identifier"].str.startswith("http")]
    merged = pe.merge(mm_clean, on="gbifID", how="inner")

    print(f"  -> {merged['species'].nunique()} especies con imagen en Peru")
    return merged


def build_species_index(df: pd.DataFrame, max_images: int) -> dict:
    """
    Agrupa por especie y selecciona hasta max_images URLs.
    Prioriza especies subrepresentadas primero.
    """
    index = {}
    grouped = df.groupby("species", sort=False)

    for species, group in grouped:
        row = group.iloc[0]
        urls = group["identifier"].dropna().unique().tolist()[:max_images]

        # Nombre de carpeta seguro para el sistema de archivos
        folder_name = species.replace(" ", "_").replace("/", "-")

        index[species] = {
            "species":          species,
            "genus":            row.get("genus", ""),
            "family":           row.get("family", ""),
            "order":            "Araneae",
            "vernacular_name":  row.get("vernacularName", ""),
            "iucn_category":    row.get("iucnRedListCategory", ""),
            "global_records":   int(row.get("global_records", 0)),
            "subrepresented":   bool(row.get("subrepresented", False)),
            "image_urls":       urls,
            "local_images":     [],    # se llena en la descarga
            "image_folder":     str(IMG_DIR / folder_name),
        }

    return index


def download_images(index: dict) -> dict:
    """
    Descarga imágenes faltantes. Si ya existe el archivo local, lo omite.
    Devuelve el index actualizado con las rutas locales.
    """
    headers = {"User-Agent": "tesis-vr-rag/1.0 (estela.alvarez.hernani@gmail.com)"}
    total_species = len(index)
    downloaded = skipped = failed = 0

    for i, (species, entry) in enumerate(index.items(), 1):
        folder = Path(entry["image_folder"])
        folder.mkdir(parents=True, exist_ok=True)

        local_images = []
        for j, url in enumerate(entry["image_urls"]):
            ext = _ext_from_url(url)
            filename = folder / f"img_{j:02d}{ext}"

            if filename.exists():
                skipped += 1
                local_images.append(str(filename))
                continue

            try:
                resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, headers=headers)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    filename.write_bytes(resp.content)
                    local_images.append(str(filename))
                    downloaded += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            time.sleep(DOWNLOAD_DELAY)

        entry["local_images"] = local_images

        if i % 50 == 0 or i == total_species:
            print(f"  [{i}/{total_species}] {species[:40]:<40} | "
                  f"descargadas={downloaded} omitidas={skipped} fallidas={failed}")

    return index


def _ext_from_url(url: str) -> str:
    url_lower = url.lower().split("?")[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        if url_lower.endswith(ext):
            return ext.replace(".jpeg", ".jpg")
    return ".jpg"


def save_kb(index: dict):
    KB_DIR.mkdir(parents=True, exist_ok=True)
    with open(KB_JSON, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\nKnowledge base guardada en: {KB_JSON}")


def load_kb() -> dict:
    with open(KB_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary(index: dict):
    total       = len(index)
    subrep      = sum(1 for e in index.values() if e["subrepresented"])
    with_images = sum(1 for e in index.values() if e["local_images"])
    total_imgs  = sum(len(e["local_images"]) for e in index.values())

    print("\n── Resumen de la knowledge base ──────────────────────")
    print(f"  Especies totales         : {total}")
    print(f"  Subrepresentadas         : {subrep}  ({subrep*100//total}%)")
    print(f"  Especies con imagen local: {with_images}")
    print(f"  Imágenes descargadas     : {total_imgs}")
    print(f"  Ubicación                : {KB_DIR}")
    print("──────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images",    type=int,  default=MAX_IMAGES_DEFAULT)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    # ── Caché: si ya existe la KB y no se pidió rebuild, cargar directo ────────
    if KB_JSON.exists() and not args.force_rebuild:
        print(f"Knowledge base encontrada en {KB_JSON}. Cargando sin re-procesar...")
        print("(usa --force-rebuild para regenerar desde cero)\n")
        index = load_kb()
        print_summary(index)
        return index

    # ── Construcción desde cero ────────────────────────────────────────────────
    print("=== Construyendo knowledge base de arañas peruanas (GBIF) ===\n")

    df = load_parquet_data()
    print(f"\nConstruyendo índice de especies (max {args.max_images} imágenes por especie)...")
    index = build_species_index(df, args.max_images)
    print(f"  → {len(index)} especies indexadas")

    print("\nDescargando imágenes...")
    index = download_images(index)

    save_kb(index)
    print_summary(index)
    return index


if __name__ == "__main__":
    main()
