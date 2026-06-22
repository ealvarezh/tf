"""
build_kb_insecta.py
-------------------
Construye la knowledge base de insectos (GBIF Insecta, polígono Peru+alrededores)
para el pipeline VR-RAG.

Uso:
    python build_kb_insecta.py                          # Lepidoptera + Coleoptera, 5 img/especie
    python build_kb_insecta.py --orders Lepidoptera     # solo mariposas
    python build_kb_insecta.py --all-orders             # todos los ordenes (~65k imagenes)
    python build_kb_insecta.py --max-images 3           # menos imagenes por especie
    python build_kb_insecta.py --suggest-test 20        # muestra 20 candidatos para imagen de prueba
    python build_kb_insecta.py --force-rebuild          # ignora cache y reescribe todo
    
    # Este debe ser por defecto el primer comando a correr para generar la KB y las imagenes locales.
    python build_kb_insecta.py --force-rebuild --all-orders


Outputs en kb_insecta/:
    knowledge_base.json       metadatos + rutas locales por especie
    test_candidates.json      especies subrepresentadas sugeridas para imagen de prueba
    images/<Especie>/         imagenes descargadas

"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ── Configuracion ──────────────────────────────────────────────────────────────
PARQUET_DIR = Path(r"C:\Users\jc.ruedah\Downloads\0072981-260519110011954\parquet")
KB_DIR      = Path(__file__).parent / "kb_insecta"
KB_JSON     = KB_DIR / "knowledge_base.json"
TEST_JSON   = KB_DIR / "test_candidates.json"
IMG_DIR     = KB_DIR / "images"

# Ordenes por defecto: los mas visuales y con mejor cobertura de imagenes
DEFAULT_ORDERS = ["Lepidoptera", "Coleoptera"]

# Especie subrepresentada = menos de N registros en el dataset (polígono Peru)
SUBREP_THRESHOLD  = 50
MIN_IMAGES        = 2      # descartar especies con menos de N imagenes disponibles
MAX_IMAGES_DEFAULT = 5
DOWNLOAD_TIMEOUT  = 10
DOWNLOAD_DELAY    = 0.08
# ──────────────────────────────────────────────────────────────────────────────


def load_and_filter(orders: list) -> pd.DataFrame:
    print("Cargando occurrence.parquet...")
    occ = pd.read_parquet(
        PARQUET_DIR / "occurrence.parquet",
        columns=[
            "gbifID", "species", "genus", "family", "order", "class",
            "decimalLatitude", "decimalLongitude", "countryCode",
            "year", "iucnRedListCategory", "vernacularName", "speciesKey",
        ],
    )

    print("Cargando multimedia.parquet...")
    mm = pd.read_parquet(
        PARQUET_DIR / "multimedia.parquet",
        columns=["gbifID", "identifier", "license"],
    )

    # Filtro: Insecta, especie identificada a nivel de especie
    mask = (occ["class"] == "Insecta") & occ["species"].notna()
    if orders:
        mask &= occ["order"].isin(orders)
    base = occ[mask].copy()

    # Flag subrepresentada: registros dentro del dataset (poligono Peru+alrededores)
    sp_counts = base.groupby("species").size().rename("dataset_records")
    base = base.join(sp_counts, on="species")
    base["subrepresented"] = base["dataset_records"] < SUBREP_THRESHOLD

    # Cruzar con imagenes
    mm_clean = mm[mm["identifier"].notna() & mm["identifier"].str.startswith("http")]
    merged = base.merge(mm_clean, on="gbifID", how="inner")

    # Filtrar especies con al menos MIN_IMAGES urls disponibles
    img_counts = merged.groupby("species")["identifier"].count()
    valid_species = img_counts[img_counts >= MIN_IMAGES].index
    merged = merged[merged["species"].isin(valid_species)]

    orders_str = ", ".join(orders) if orders else "todos los ordenes"
    print(f"  Filtro: {orders_str}")
    print(f"  -> {merged['species'].nunique()} especies con >= {MIN_IMAGES} imagenes")
    print(f"  -> subrepresentadas ({SUBREP_THRESHOLD} registros): "
          f"{merged[merged['subrepresented']]['species'].nunique()}")
    return merged


def build_index(df: pd.DataFrame, max_images: int) -> dict:
    index = {}
    for species, group in df.groupby("species", sort=False):
        row = group.iloc[0]
        urls = group["identifier"].dropna().unique().tolist()[:max_images]
        folder_name = species.replace(" ", "_").replace("/", "-")

        index[species] = {
            "species":        species,
            "genus":          str(row["genus"]) if pd.notna(row["genus"]) else "",
            "family":         str(row["family"]) if pd.notna(row["family"]) else "",
            "order":          str(row["order"]) if pd.notna(row["order"]) else "",
            "vernacular_name": str(row["vernacularName"]) if pd.notna(row.get("vernacularName")) else "",
            "iucn_category":  str(row["iucnRedListCategory"]) if pd.notna(row.get("iucnRedListCategory")) else "",
            "dataset_records": int(row["dataset_records"]),
            "subrepresented": bool(row["subrepresented"]),
            "image_urls":     urls,
            "local_images":   [],
            "image_folder":   str(IMG_DIR / folder_name),
        }
    return index


def download_images(index: dict) -> dict:
    headers = {"User-Agent": "tesis-vr-rag/1.0 (estela.alvarez.hernani@gmail.com)"}
    total = len(index)
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

        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}]  descargadas={downloaded}  omitidas={skipped}  fallidas={failed}")

    return index


def _ext_from_url(url: str) -> str:
    u = url.lower().split("?")[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if u.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def build_test_candidates(index: dict, n: int = 30) -> list:
    """
    Devuelve las N mejores especies subrepresentadas para usar como imagen de prueba.
    Criterio: subrepresentada + tiene imagenes descargadas + ordenes visuales primero.
    """
    ORDER_PRIORITY = {"Lepidoptera": 0, "Coleoptera": 1, "Odonata": 2}
    candidates = [
        e for e in index.values()
        if e["subrepresented"] and len(e["local_images"]) >= MIN_IMAGES
    ]
    candidates.sort(key=lambda e: (
        ORDER_PRIORITY.get(e["order"], 9),
        e["dataset_records"],
    ))
    return candidates[:n]


def to_species_db(index: dict) -> list:
    """
    Convierte la KB al formato que espera demo_vr_rag.py (SPECIES_DB).
    Genera descripcion morfologica basica desde taxonomia cuando no hay texto.
    Para produccion: reemplazar _auto_description por Wikipedia/iNaturalist API.
    """
    result = []
    for entry in index.values():
        if not entry["local_images"]:
            continue
        result.append({
            "id":             entry["species"].lower().replace(" ", "_"),
            "name":           entry["species"],
            "family":         entry["family"],
            "order":          entry["order"],
            "description":    _auto_description(entry),
            "image_url":      entry["local_images"][0],
            "image_anchors":  entry["local_images"],   # todas las imagenes para DINOv2
            "subrepresented": entry["subrepresented"],
            "dataset_records": entry["dataset_records"],
        })
    return result


def _auto_description(entry: dict) -> str:
    """
    Descripcion minima generada desde campos taxonomicos.
    Suficiente para que BioCLIP/CLIP generen un vector semantico util.
    """
    parts = [
        f"{entry['species']}.",
        f"Order: {entry['order']}." if entry["order"] else "",
        f"Family: {entry['family']}." if entry["family"] else "",
        f"Genus: {entry['genus']}." if entry["genus"] else "",
        f"Insect found in Peru and surrounding Andean-Amazonian region.",
        f"Common name: {entry['vernacular_name']}." if entry["vernacular_name"] else "",
        f"IUCN: {entry['iucn_category']}." if entry["iucn_category"] else "",
    ]
    return " ".join(p for p in parts if p)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_kb() -> dict:
    with open(KB_JSON, encoding="utf-8") as f:
        return json.load(f)


def print_summary(index: dict):
    total      = len(index)
    subrep     = sum(1 for e in index.values() if e["subrepresented"])
    with_imgs  = sum(1 for e in index.values() if e["local_images"])
    total_imgs = sum(len(e["local_images"]) for e in index.values())
    orders     = {}
    for e in index.values():
        orders[e["order"]] = orders.get(e["order"], 0) + 1

    print("\n── Resumen knowledge base ─────────────────────────────")
    print(f"  Especies totales          : {total}")
    print(f"  Subrepresentadas          : {subrep}  ({subrep*100//max(total,1)}%)")
    print(f"  Con imagenes locales      : {with_imgs}")
    print(f"  Total imagenes descargadas: {total_imgs}")
    print(f"  Por orden: {orders}")
    print(f"  Ubicacion                 : {KB_DIR}")
    print("───────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders",        nargs="+", default=DEFAULT_ORDERS,
                        help=f"Ordenes a incluir (default: {DEFAULT_ORDERS})")
    parser.add_argument("--all-orders",    action="store_true",
                        help="Incluir todos los ordenes de Insecta")
    parser.add_argument("--max-images",    type=int, default=MAX_IMAGES_DEFAULT)
    parser.add_argument("--suggest-test",  type=int, default=0, metavar="N",
                        help="Mostrar N candidatos para imagen de prueba y salir")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    orders = [] if args.all_orders else args.orders

    # ── Si solo se pide sugerir candidatos de prueba ───────────────────────────
    if args.suggest_test:
        if not KB_JSON.exists():
            print("Knowledge base no encontrada. Corre sin --suggest-test primero.")
            return
        index = load_kb()
        candidates = build_test_candidates(index, n=args.suggest_test)
        print(f"\n── Top {len(candidates)} candidatos para imagen de prueba ──────────────")
        print(f"  (subrepresentadas, con imagenes descargadas, ordenadas por orden visual)")
        print()
        for i, c in enumerate(candidates, 1):
            print(f"  {i:2}. {c['species']:<40} orden={c['order']:<15} "
                  f"registros={c['dataset_records']:3}  imgs={len(c['local_images'])}")
            print(f"      URL prueba sugerida: {c['image_urls'][0][:80]}")
        save_json(candidates, TEST_JSON)
        print(f"\n  Guardado en: {TEST_JSON}")
        return

    # ── Cache: si ya existe y no se pidio rebuild, cargar directo ─────────────
    if KB_JSON.exists() and not args.force_rebuild:
        print(f"Knowledge base encontrada en {KB_JSON}. Cargando sin re-procesar...")
        print("(usa --force-rebuild para regenerar desde cero)\n")
        index = load_kb()
        print_summary(index)
        return

    # ── Construccion desde cero ────────────────────────────────────────────────
    print("=== Construyendo knowledge base de insectos (GBIF Insecta, Peru) ===\n")

    df = load_and_filter(orders)

    print(f"\nConstruyendo indice (max {args.max_images} imagenes/especie)...")
    index = build_index(df, args.max_images)

    print("\nDescargando imagenes...")
    index = download_images(index)

    save_json(index, KB_JSON)

    # Generar SPECIES_DB listo para pegar en demo_vr_rag.py
    species_db = to_species_db(index)
    save_json(species_db, KB_DIR / "species_db.json")

    print_summary(index)
    print(f"  SPECIES_DB listo en: {KB_DIR / 'species_db.json'}")
    print(f"\n  Siguiente paso: python build_kb_insecta.py --suggest-test 20")


if __name__ == "__main__":
    main()
