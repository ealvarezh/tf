"""
batch_test_insecta.py
---------------------
Corre el pipeline VR-RAG sobre una lista de especies candidatas y genera
un resumen comparativo de resultados.

Uso:
    python prueba_20.py                          # correr pipeline completo
    python prueba_20.py --top-m 30 --top-k 10
    python prueba_20.py --summary-only           # leer batch_results.json sin re-correr
    python prueba_20.py --rebuild-cache          # forzar recomputo de embeddings
"""

import argparse
import json
import sys
import urllib.request
import io
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, BooleanType, ArrayType, MapType, LongType
import os 

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ["HADOOP_HOME"] = rf"C:\Users\jc.ruedah\hadoop"
os.environ["PATH"] += r";C:\Users\jc.ruedah\hadoop\bin"

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from test_vr_rag_insecta import (
    load_species_db, load_encoders, load_dino,
    stage1, stage2, stage3,
    _encode_img, _encode_texts, _encode_dino,
    mrr_at_k,
)

KB_DIR        = Path(__file__).parent.parent / "data" / "bronze" / "kb_insecta"
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


def fetch_image(url: str) -> "Image.Image | None":
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
    if result_species["name"].strip().lower() == target["species"].strip().lower():
        return "especie"
    if result_species.get("family", "").lower() == target["family"].lower():
        return "familia"
    if result_species.get("order", "").lower() == target["order"].lower():
        return "orden"
    return "miss"




# ── Stage 1 con cache ──────────────────────────────────────────────────────────

def stage1_cached(query_img, species_db, encoders, text_embs, device, top_m):
    """Stage 1 usando embeddings de texto precalculados (sin re-tokenizar)."""
    print(f"\n{'─'*60}")
    print("  ETAPA 1 — Recuperacion cross-modal (BioCLIP + CLIP) [cache]")
    print(f"{'─'*60}")

    all_scores = {}
    for name, (model, preprocess, _) in encoders.items():
        q_emb = _encode_img(model, preprocess, query_img, device)
        t_embs = text_embs[name].to(device)
        scores = (q_emb @ t_embs.T).squeeze(0)
        all_scores[name] = scores.cpu()

    ensemble = torch.stack(list(all_scores.values())).mean(dim=0)
    top_idx  = ensemble.argsort(descending=True)[:top_m]

    results = []
    for rank, idx in enumerate(top_idx):
        results.append({
            "rank":           rank + 1,
            "species":        species_db[idx],
            "score_bioclip":  all_scores["bioclip"][idx].item(),
            "score_clip":     all_scores["clip"][idx].item(),
            "score_ensemble": ensemble[idx].item(),
        })

    print(f"\n  Top-{top_m}:")
    for r in results:
        subrep = " [SUBREP]" if r["species"]["subrepresented"] else ""
        print(f"  {r['rank']}. {r['species']['name']:<40} "
              f"ensemble={r['score_ensemble']:.4f}{subrep}")
    return results


# ── Stage 2 con cache ──────────────────────────────────────────────────────────

def stage2_cached(query_img, candidates, dino_model, dino_proc,
                  dino_embs, device, lam, top_k):
    """Stage 2 usando embeddings DINOv2 precalculados para los anclas."""
    print(f"\n{'─'*60}")
    print(f"  ETAPA 2 — Re-ranking visual DINOv2  (lambda={lam}) [cache]")
    print(f"{'─'*60}")

    q_dino = _encode_dino(dino_model, dino_proc, query_img, device)

    reranked = []
    for cand in candidates:
        sp = cand["species"]
        sp_name = sp["name"]
        s_dino = 0.0
        if sp_name in dino_embs:
            anchors = dino_embs[sp_name].to(device)
            sims = (q_dino @ anchors.T).squeeze(0)
            s_dino = sims.max().item() if sims.numel() > 0 else 0.0
        s_final = lam * cand["score_ensemble"] + (1 - lam) * s_dino
        reranked.append({**cand, "score_dino": s_dino, "score_final": s_final})

    reranked.sort(key=lambda x: x["score_final"], reverse=True)
    for i, r in enumerate(reranked):
        r["rank_final"] = i + 1

    print(f"\n  {'Especie':<40} {'s_cross':>8} {'s_dino':>8} {'s_final':>8}  Cambio")
    print("  " + "-" * 72)
    for r in reranked:
        delta = r["rank"] - r["rank_final"]
        ch = f"+{delta}" if delta > 0 else (str(delta) if delta < 0 else "=")
        top = " <- TOP1" if r["rank_final"] == 1 else ""
        print(f"  {r['species']['name']:<40} "
              f"{r['score_ensemble']:>8.4f} {r['score_dino']:>8.4f} "
              f"{r['score_final']:>8.4f}  {ch:>5}{top}")
    return reranked[:top_k]


# ── Batch ──────────────────────────────────────────────────────────────────────

def run_batch(species_db, encoders, dino_model, dino_proc,
              text_embs, dino_embs, device, top_m, top_k):
    results = []

    for i, cand in enumerate(CANDIDATES, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(CANDIDATES)}] {cand['species']} ({cand['family']})")
        print(f"  URL: {cand['img_url'][:70]}...")

        img = fetch_image(cand["img_url"])
        if img is None:
            results.append({**cand, "status": "error", "top1": None,
                            "hit_level": "error", "lmm_response": "ERROR: imagen no descargada"})
            continue

        stage1_results = stage1_cached(img, species_db, encoders, text_embs, device, top_m)
        top_k_results  = stage2_cached(img, stage1_results, dino_model, dino_proc,
                                       dino_embs, device, lam=0.7, top_k=top_k)
        lmm_result     = stage3(top_k_results, image_url=cand["img_url"])

        top1 = top_k_results[0]["species"]
        hit  = taxonomic_hit(top1, cand)
        mrr1   = mrr_at_k(top_k_results, cand["species"], k=1)
        mrr10  = mrr_at_k(top_k_results, cand["species"], k=10)
        mrr_e1 = mrr_at_k(stage1_results, cand["species"], k=top_m)

        print(f"\n  -> Top-1: {top1['name']} ({top1.get('family','')}) | hit: {hit.upper()}")
        print(f"  -> LMM predicho: {lmm_result['predicted']}")

        results.append({
            "target_species":   cand["species"],
            "target_family":    cand["family"],
            "target_order":     cand["order"],
            "predicted_top1":   top1["name"],
            "predicted_family": top1.get("family", ""),
            "lmm_predicted":    lmm_result["predicted"],
            "lmm_response":     lmm_result["lmm_response"],
            "hit_level":        hit,
            "mrr_at_1":         mrr1,
            "mrr_at_10":        mrr10,
            "mrr_e1":           mrr_e1,
            "score_final":      top_k_results[0]["score_final"],
            "top5": [
                {
                    "rank":   r["rank_final"],
                    "name":   r["species"]["name"],
                    "family": r["species"].get("family", ""),
                    "score":  round(r["score_final"], 4),
                    "hit":    taxonomic_hit(r["species"], cand),
                }
                for r in top_k_results[:5]
            ],
        })

    return results


def print_summary(results: list):
    print(f"\n{'='*70}")
    print("  RESUMEN BATCH")
    print(f"{'='*70}")

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

    has_lmm = any("lmm_response" in r for r in results)
    header  = f"  {'Especie objetivo':<35} {'Top-1 predicho':<35} {'Hit':<8} MRR@10"
    if has_lmm:
        header += "  LMM"
    print(header)
    print("  " + "-" * (90 + (30 if has_lmm else 0)))

    for r in results:
        hits[r["hit_level"]] += 1
        mrr_sum += r.get("mrr_at_10", 0)
        line = (f"  {r['target_species']:<35} "
                f"{r.get('predicted_top1', 'ERROR'):<35} "
                f"{r['hit_level'].upper():<8} "
                f"{r.get('mrr_at_10', 0):.4f}")
        if has_lmm:
            lmm_text = r.get("lmm_response", "")[:60].replace("\n", " ")
            line += f"  {lmm_text}"
        print(line)

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
    parser.add_argument("--top-m",         type=int, default=30)
    parser.add_argument("--top-k",         type=int, default=10)
    parser.add_argument("--summary-only",  action="store_true",
                        help="Leer batch_results.json y mostrar resumen sin re-correr")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Forzar recomputo de embeddings aunque exista cache")
    args = parser.parse_args()

    if args.summary_only:
        if not RESULTS_JSON.exists():
            print(f"ERROR: {RESULTS_JSON} no encontrado.")
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

    #Obtener los embeddings desde gold 
    spark = (
        SparkSession.builder
        .appName("obtener_datos_gold")
        .master("local[*]")
        # .config("spark.hadoop.hadoop.native.lib","false")
        # .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        # .config("spark.hadoop.fs.file.impl.disable.cache","true")
        .getOrCreate()
        )
    
    text_embs, dino_embs = {}, {}

    """
    Precalcula embeddings de texto (BioCLIP + CLIP) para todas las especies.
    Retorna dict: {"bioclip": Tensor[N,D], "clip": Tensor[N,D], "n_species": int}
    """
    ruta_gold_distribuido = Path(__file__).parent.parent / "data" / "gold" / "embeddings_textuales_distribuidos" / "clip"
    df_clip = spark.read.parquet(str(ruta_gold_distribuido))
    ruta_gold_distribuido = Path(__file__).parent.parent / "data" / "gold" / "embeddings_textuales_distribuidos" / "bioclip"
    df_bioclip = spark.read.parquet(str(ruta_gold_distribuido))

    text_embs["bioclip"] = torch.tensor(df_bioclip.select("embedding").rdd.map(lambda row: row["embedding"]).collect()).to(device)
    text_embs["clip"]    = torch.tensor(df_clip.select("embedding").rdd.map(lambda row: row["embedding"]).collect()).to(device)
    text_embs["n_species"] = text_embs["bioclip"].shape[0]

    """
    Precalcula embeddings DINOv2 de todas las imagenes ancla por especie.
    Retorna dict: {species_name: Tensor[n_anchors, D]}
    """
    ruta_gold_distribuido = Path(__file__).parent.parent / "data" / "gold" / "embeddings_imagenes_distribuidas" / "dino"
    df_dino = spark.read.parquet(str(ruta_gold_distribuido))
    df_grouped = df_dino.groupBy("id_str").orderBy("timestamp").agg(F.collect_list("embedding").alias("embeddings"))

    from pyspark.sql import functions as F
    for row in df_grouped.select("id_str", "embeddings").toLocalIterator():
        dino_embs[row["id_str"]] = torch.tensor([emb for emb in row["embeddings"]])

    results = run_batch(species_db, encoders, dino_model, dino_proc,
                        text_embs, dino_embs, device, args.top_m, args.top_k)

    # KB_DIR.mkdir(exist_ok=True)
    # with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)
    # print(f"\nResultados guardados en: {RESULTS_JSON}")

    print_summary(results)


if __name__ == "__main__":
    main()
