"""
batch_test_insecta.py
---------------------
Corre el pipeline VR-RAG sobre una lista de especies candidatas y genera
un resumen comparativo de resultados.

Uso:
    python batch_test_insecta.py
    python batch_test_insecta.py --top-m 30 --top-k 10
"""

import argparse
import json
import sys
import urllib.request
import io
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
from PIL import Image

# Reutiliza las funciones del script de test
sys.path.insert(0, str(Path(__file__).parent))
from test_vr_rag_insecta import (
    load_species_db, load_encoders, load_dino,
    stage1, stage2,
    _encode_img, _encode_texts, _encode_dino,
    mrr_at_k,
)

KB_DIR        = Path(__file__).parent / "kb_insecta"
RESULTS_JSON  = KB_DIR / "batch_results.json"
_MAX_SIDE     = 512

# ── Lista de candidatos ────────────────────────────────────────────────────────
CANDIDATES = [
    {
        "species":  "Argyractis zamoralis",
        "family":   "Crambidae",
        "order":    "Lepidoptera",
        "img_url":  "https://collections.nmnh.si.edu/media/?i=15786364&h=1015",
    },
    {
        "species":  "Euharpyia comita",
        "family":   "Notodontidae",
        "order":    "Lepidoptera",
        "img_url":  "https://collections.nmnh.si.edu/media/?i=11827820&h=2000",
    },
    {
        "species":  "Argyria croceicinctella",
        "family":   "Crambidae",
        "order":    "Lepidoptera",
        "img_url":  "https://collections.nmnh.si.edu/media/?i=15786377&h=1480",
    },
    {
        "species":  "Stenia midalis",
        "family":   "Crambidae",
        "order":    "Lepidoptera",
        "img_url":  "https://collections.nmnh.si.edu/media/?i=15791920&h=1667",
    },
    {
        "species":  "Battus zetides",
        "family":   "Papilionidae",
        "order":    "Lepidoptera",
        "img_url":  "https://images.collections.yale.edu/iiif/2/ypm:d0ef06fe-c8e9-42fc-b1ac-6a2fcad08472/full/max/0/default.jpg",
    },
    {
        "species":  "Heliconius heurippa",
        "family":   "Nymphalidae",
        "order":    "Lepidoptera",
        "img_url":  "https://zenodo.org/record/2686762/files/CAM009116_d.JPG",
    },
    {
        "species":  "Pereute swainsoni",
        "family":   "Pieridae",
        "order":    "Lepidoptera",
        "img_url":  "https://iiif.mcz.harvard.edu/iiif/3/1421033/full/max/0/default.jpg",
    },
    {
        "species":  "Pieris rapae",
        "family":   "Pieridae",
        "order":    "Lepidoptera",
        "img_url":  "https://iiif.mcz.harvard.edu/iiif/3/1416636/full/max/0/default.jpg",
    },
    {
        "species":  "Decinea milesi",
        "family":   "Hesperiidae",
        "order":    "Lepidoptera",
        "img_url":  "https://iiif.mcz.harvard.edu/iiif/3/1467188/full/max/0/default.jpg",
    },
    {
        "species":  "Hamadryas arethusa",
        "family":   "Nymphalidae",
        "order":    "Lepidoptera",
        "img_url":  "https://iiif.mcz.harvard.edu/iiif/3/1450930/full/max/0/default.jpg",
    },
    {
        "species":  "Dalla diraspes",
        "family":   "Hesperiidae",
        "order":    "Lepidoptera",
        "img_url":  "https://iiif.mcz.harvard.edu/iiif/3/224261/full/max/0/default.jpg",
    },
    {
        "species":  "Hypanartia fassli",
        "family":   "Nymphalidae",
        "order":    "Lepidoptera",
        "img_url":  "https://iiif.mcz.harvard.edu/iiif/3/1524803/full/max/0/default.jpg",
    },
    {
        "species":  "Paratrytone argentea",
        "family":   "Hesperiidae",
        "order":    "Lepidoptera",
        "img_url":  "https://iiif.mcz.harvard.edu/iiif/3/1467227/full/max/0/default.jpg",
    },
    {
        "species":  "Olceclostera angelica",
        "family":   "Apatelodidae",
        "order":    "Lepidoptera",
        "img_url":  "https://inaturalist-open-data.s3.amazonaws.com/photos/632882281/original.jpg",
    },
]
# URLs con manifiestos JSON (nrm.se) se omiten — no son imágenes directas


def fetch_image(url: str) -> Image.Image | None:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (tesis-vr-rag)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if max(img.size) > _MAX_SIDE:
            img.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"    ERROR descargando imagen: {e}")
        return None


def taxonomic_hit(result_species: dict, target: dict) -> str:
    """Clasifica el resultado por nivel taxonomico."""
    if result_species["name"].strip().lower() == target["species"].strip().lower():
        return "especie"
    if result_species.get("family", "").lower() == target["family"].lower():
        return "familia"
    if result_species.get("order", "").lower() == target["order"].lower():
        return "orden"
    return "miss"


def run_batch(species_db, encoders, dino_model, dino_proc, device, top_m, top_k):
    results = []

    for i, cand in enumerate(CANDIDATES, 1):
        print(f"\n[{i}/{len(CANDIDATES)}] {cand['species']} ({cand['family']})")
        print(f"  URL: {cand['img_url'][:70]}...")

        img = fetch_image(cand["img_url"])
        if img is None:
            results.append({**cand, "status": "error", "top1": None, "hit_level": "error"})
            continue

        # Etapa 1
        stage1_results = stage1(img, species_db, encoders, device, top_m)
        # Etapa 2
        top_k_results  = stage2(img, stage1_results, dino_model, dino_proc,
                                device, lam=0.7, top_k=top_k)

        top1    = top_k_results[0]["species"]
        hit     = taxonomic_hit(top1, cand)
        mrr1    = mrr_at_k(top_k_results, cand["species"], k=1)
        mrr10   = mrr_at_k(top_k_results, cand["species"], k=10)
        mrr_e1  = mrr_at_k(stage1_results, cand["species"], k=top_m)

        print(f"  -> Top-1: {top1['name']} ({top1.get('family','')}) | hit: {hit.upper()}")

        results.append({
            "target_species": cand["species"],
            "target_family":  cand["family"],
            "target_order":   cand["order"],
            "predicted_top1": top1["name"],
            "predicted_family": top1.get("family", ""),
            "hit_level":      hit,
            "mrr_at_1":       mrr1,
            "mrr_at_10":      mrr10,
            "mrr_e1":         mrr_e1,
            "score_final":    top_k_results[0]["score_final"],
            "top5": [
                {
                    "rank": r["rank_final"],
                    "name": r["species"]["name"],
                    "family": r["species"].get("family", ""),
                    "score": round(r["score_final"], 4),
                    "hit": taxonomic_hit(r["species"], cand),
                }
                for r in top_k_results[:5]
            ],
        })

    return results


def print_summary(results: list):
    print(f"\n{'='*70}")
    print("  RESUMEN BATCH")
    print(f"{'='*70}")

    # Normalizar claves — entradas de error usan 'species' en lugar de 'target_species'
    for r in results:
        if "target_species" not in r:
            r["target_species"]  = r.get("species", "ERROR")
            r["target_family"]   = r.get("family", "")
            r["target_order"]    = r.get("order", "")
            r["predicted_top1"]  = r.get("top1") or "ERROR"
            r.setdefault("mrr_at_10", 0)
            r.setdefault("hit_level", "error")

    total   = len([r for r in results if r["hit_level"] != "error"])
    hits    = {"especie": 0, "familia": 0, "orden": 0, "miss": 0, "error": 0}
    mrr_sum = 0.0

    print(f"\n  {'Especie objetivo':<35} {'Top-1 predicho':<35} {'Hit':<8} MRR@10")
    print("  " + "-" * 90)
    for r in results:
        hits[r["hit_level"]] += 1
        mrr_sum += r.get("mrr_at_10", 0)
        print(f"  {r['target_species']:<35} "
              f"{r.get('predicted_top1', 'ERROR'):<35} "
              f"{r['hit_level'].upper():<8} "
              f"{r.get('mrr_at_10', 0):.4f}")

    print(f"\n  Resultados por nivel taxonomico:")
    print(f"    Especie exacta  : {hits['especie']}")
    print(f"    Familia correcta: {hits['familia']}")
    print(f"    Orden correcto  : {hits['orden']}")
    print(f"    Miss            : {hits['miss']}")
    print(f"    Error descarga  : {hits['error']}")
    if total > 0:
        familia_ok = hits["especie"] + hits["familia"]
        orden_ok   = familia_ok + hits["orden"]
        print(f"\n  Precision familia+  : {familia_ok}/{total} ({familia_ok*100//total}%)")
        print(f"  Precision orden+    : {orden_ok}/{total}  ({orden_ok*100//total}%)")
        print(f"  MRR@10 promedio     : {mrr_sum/total:.4f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-m",        type=int, default=30)
    parser.add_argument("--top-k",        type=int, default=10)
    parser.add_argument("--summary-only", action="store_true",
                        help="Leer batch_results.json y mostrar resumen sin re-correr")
    args = parser.parse_args()

    # Modo solo resumen — lee el JSON ya guardado
    if args.summary_only:
        if not RESULTS_JSON.exists():
            print(f"ERROR: {RESULTS_JSON} no encontrado. Corre sin --summary-only primero.")
            return
        with open(RESULTS_JSON, encoding="utf-8") as f:
            results = json.load(f)
        print_summary(results)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"  VR-RAG Insecta — Batch Test ({len(CANDIDATES)} especies)")
    print(f"  Dispositivo: {device}  |  top-m={args.top_m}  top-k={args.top_k}")
    print(f"{'='*70}\n")

    species_db = load_species_db()
    encoders   = load_encoders(device)
    dino_model, dino_proc = load_dino(device)

    results = run_batch(species_db, encoders, dino_model, dino_proc,
                        device, args.top_m, args.top_k)

    KB_DIR.mkdir(exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {RESULTS_JSON}")

    print_summary(results)


if __name__ == "__main__":
    main()
