"""
eval_mrr_loo.py
----------------
Evalua el pipeline VR-RAG (Etapa 1 BioCLIP+CLIP, Etapa 2 DINOv2) sobre el set
de 20 especies generado por select_test_species.py, usando leave-one-out:
para cada especie se usa una imagen local como query (la que se dejo afuera
del set de anclas) y se mide en que posicion del ranking aparece la especie
correcta.

Calcula MRR@3, MRR@10 y MRR@30 (promedio sobre las 20 especies) y guarda el
detalle por especie en kb_insecta/mrr_results.json.

Correr en la maquina con los datos y modelos (junto a kb_insecta/ y
big_data_project/src/test_vr_rag_insecta.py):

    python eval_mrr_loo.py
    python eval_mrr_loo.py --candidates kb_insecta/test_candidates_loo.json
    python eval_mrr_loo.py --rebuild-cache
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "big_data_project" / "src"))
from test_vr_rag_insecta import (
    load_species_db, load_encoders, load_dino,
    _encode_img, _encode_texts, _encode_dino,
)

KB_DIR          = Path(__file__).parent / "kb_insecta"
CANDIDATES_JSON = KB_DIR / "test_candidates_loo.json"
RESULTS_JSON    = KB_DIR / "mrr_results.json"
TEXT_EMB_CACHE  = KB_DIR / "text_emb_cache.pt"
DINO_EMB_CACHE  = KB_DIR / "dino_emb_cache.pt"
TOP_M           = 30   # tambien define el techo de MRR (MRR@30)
LAM             = 0.7  # peso del score cross-modal vs dino en Etapa 2


def build_text_emb_cache(species_db, encoders, device):
    descriptions = [e["description"] for e in species_db]
    text_embs = {}
    for name, (model, _, tokenizer) in encoders.items():
        text_embs[name] = _encode_texts(model, tokenizer, descriptions, device).cpu()
    text_embs["n_species"] = len(species_db)
    torch.save(text_embs, TEXT_EMB_CACHE)
    return text_embs


def build_dino_emb_cache(species_db, dino_model, dino_proc, device):
    dino_embs = {}
    for sp in species_db:
        anchors = []
        for img_path in sp.get("image_anchors", [sp["image_url"]]):
            try:
                img = Image.open(img_path).convert("RGB")
                anchors.append(_encode_dino(dino_model, dino_proc, img, device).cpu())
            except Exception:
                pass
        if anchors:
            dino_embs[sp["name"]] = torch.cat(anchors, dim=0)
    torch.save(dino_embs, DINO_EMB_CACHE)
    return dino_embs


def load_or_build_caches(species_db, encoders, dino_model, dino_proc, device, rebuild):
    text_embs = dino_embs = None
    if TEXT_EMB_CACHE.exists() and not rebuild:
        saved = torch.load(TEXT_EMB_CACHE, map_location="cpu")
        if saved.get("n_species") == len(species_db):
            text_embs = saved
    if text_embs is None:
        print("Calculando cache de texto (BioCLIP+CLIP)...")
        text_embs = build_text_emb_cache(species_db, encoders, device)

    if DINO_EMB_CACHE.exists() and not rebuild:
        saved = torch.load(DINO_EMB_CACHE, map_location="cpu")
        if len(saved) == len(species_db):
            dino_embs = saved
    if dino_embs is None:
        print("Calculando cache DINOv2 de imagenes ancla...")
        dino_embs = build_dino_emb_cache(species_db, dino_model, dino_proc, device)

    return text_embs, dino_embs


def stage1_scores(query_img, species_db, encoders, text_embs, device):
    all_scores = {}
    for name, (model, preprocess, _) in encoders.items():
        q_emb = _encode_img(model, preprocess, query_img, device)
        t_embs = text_embs[name].to(device)
        all_scores[name] = (q_emb @ t_embs.T).squeeze(0).cpu()
    return torch.stack(list(all_scores.values())).mean(dim=0)


def rerank(query_img, species_db, ensemble, dino_model, dino_proc, dino_embs,
           device, lam, exclude_species, exclude_image, top_m):
    top_idx = ensemble.argsort(descending=True)[:top_m]
    q_dino  = _encode_dino(dino_model, dino_proc, query_img, device)

    reranked = []
    for idx in top_idx:
        idx = idx.item()
        sp = species_db[idx]
        s_cross = ensemble[idx].item()

        if sp["name"] == exclude_species:
            # Recalcular DINO excluyendo la imagen query usada como ancla
            anchors = [p for p in sp.get("image_anchors", []) if p != exclude_image]
            scores = []
            for p in anchors:
                try:
                    a = _encode_dino(dino_model, dino_proc, Image.open(p).convert("RGB"), device)
                    scores.append((q_dino @ a.T).item())
                except Exception:
                    pass
            s_dino = max(scores) if scores else 0.0
        else:
            s_dino = 0.0
            if sp["name"] in dino_embs:
                sims = (q_dino @ dino_embs[sp["name"]].to(device).T).squeeze(0)
                s_dino = sims.max().item() if sims.numel() > 0 else 0.0

        s_final = lam * s_cross + (1 - lam) * s_dino
        reranked.append({"name": sp["name"], "order": sp["order"], "family": sp["family"],
                          "score_final": s_final})

    reranked.sort(key=lambda x: x["score_final"], reverse=True)
    return reranked


def mrr_at_k(ranking, target, k):
    for i, r in enumerate(ranking[:k]):
        if r["name"].strip().lower() == target.strip().lower():
            return 1.0 / (i + 1)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=str(CANDIDATES_JSON))
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        print(f"ERROR: no existe {cand_path}. Corre primero select_test_species.py")
        return
    with open(cand_path, encoding="utf-8") as f:
        candidates = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}  |  {len(candidates)} especies de prueba\n")

    species_db = load_species_db()
    encoders = load_encoders(device)
    dino_model, dino_proc = load_dino(device)
    text_embs, dino_embs = load_or_build_caches(
        species_db, encoders, dino_model, dino_proc, device, args.rebuild_cache
    )

    results = []
    for i, c in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {c['species']} ({c['order']})")
        try:
            query_img = Image.open(c["query_image"]).convert("RGB")
        except Exception as e:
            print(f"  ERROR cargando imagen: {e}")
            results.append({**c, "mrr_at_3": 0.0, "mrr_at_10": 0.0, "mrr_at_30": 0.0,
                             "rank": None, "error": str(e)})
            continue

        ensemble = stage1_scores(query_img, species_db, encoders, text_embs, device)
        ranking  = rerank(query_img, species_db, ensemble, dino_model, dino_proc,
                          dino_embs, device, LAM, c["species"], c["query_image"], TOP_M)

        rank_pos = next((i_ for i_, r in enumerate(ranking, 1)
                          if r["name"].strip().lower() == c["species"].strip().lower()), None)

        m3, m10, m30 = (mrr_at_k(ranking, c["species"], k) for k in (3, 10, 30))
        print(f"  -> posicion en ranking: {rank_pos}  MRR@3={m3:.4f}  MRR@10={m10:.4f}  MRR@30={m30:.4f}")

        results.append({**c, "rank": rank_pos,
                         "mrr_at_3": m3, "mrr_at_10": m10, "mrr_at_30": m30})

    KB_DIR.mkdir(exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n = len(results)
    avg3  = sum(r["mrr_at_3"] for r in results) / n
    avg10 = sum(r["mrr_at_10"] for r in results) / n
    avg30 = sum(r["mrr_at_30"] for r in results) / n

    print(f"\n{'='*60}")
    print(f"  RESUMEN ({n} especies, ordenes={len(set(r['order'] for r in results))})")
    print(f"{'='*60}")
    print(f"  MRR@3  promedio : {avg3:.4f}")
    print(f"  MRR@10 promedio : {avg10:.4f}")
    print(f"  MRR@30 promedio : {avg30:.4f}")
    print(f"\n  Detalle guardado en: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
