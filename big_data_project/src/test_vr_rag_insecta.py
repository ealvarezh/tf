"""
test_vr_rag_insecta.py
-----------------------
Prueba el pipeline VR-RAG con la knowledge base de insectos peruanos (GBIF).
La imagen de prueba debe ser de una especie subrepresentada que SI esta en la KB,
pero la imagen en si NO debe ser una de las usadas como anclaje.

Uso:
    python test_vr_rag_insecta.py --image-url "https://..." --top-m 5 --top-k 5
    python test_vr_rag_insecta.py --image mi_foto.jpg --target-species "Morpho helenor"
"""

import argparse
import json
import os
import sys
import urllib.request
import io
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
from PIL import Image
from openai import OpenAI

# ── LMM config (DeepSeek via NVIDIA) ──────────────────────────────────────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "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")
LMM_MODEL      = "deepseek-ai/deepseek-v4-flash"
# ──────────────────────────────────────────────────────────────────────────────

KB_DIR          = Path(__file__).parent / "kb_insecta"
KB_JSON         = KB_DIR / "knowledge_base.json"
SPECIES_DB_JSON = KB_DIR / "species_db.json"
TEST_IMG_PATH   = KB_DIR / "test_query.jpg"
_MAX_SIDE       = 512


def load_species_db() -> list:
    if not SPECIES_DB_JSON.exists():
        print(f"ERROR: {SPECIES_DB_JSON} no encontrado.")
        print("Corre primero: python build_kb_insecta.py")
        sys.exit(1)
    with open(SPECIES_DB_JSON, encoding="utf-8") as f:
        db = json.load(f)
    db = [e for e in db if e.get("image_anchors")]
    print(f"Knowledge base cargada: {len(db)} especies")
    print(f"  Subrepresentadas: {sum(1 for e in db if e['subrepresented'])}")
    return db


def download_test_image(url: str) -> Image.Image:
    print(f"Descargando imagen de prueba...")
    headers = {"User-Agent": "Mozilla/5.0 (tesis-vr-rag)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if max(img.size) > _MAX_SIDE:
        img.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.LANCZOS)
    img.save(TEST_IMG_PATH, "JPEG", quality=90)
    print(f"  -> {TEST_IMG_PATH} ({img.size[0]}x{img.size[1]} px)")
    return img


# ── Encoders ───────────────────────────────────────────────────────────────────

def load_encoders(device: str) -> dict:
    import open_clip
    print("\nCargando BioCLIP...")
    bio_model, _, bio_pre = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    bio_tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    bio_model = bio_model.to(device).eval()

    print("Cargando CLIP...")
    clip_model, _, clip_pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device).eval()

    return {
        "bioclip": (bio_model, bio_pre, bio_tok),
        "clip":    (clip_model, clip_pre, clip_tok),
    }


def load_dino(device: str):
    from transformers import AutoImageProcessor, AutoModel
    print("Cargando DINOv2...")
    proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    return model, proc


def _encode_img(model, preprocess, img: Image.Image, device: str) -> torch.Tensor:
    t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model.encode_image(t)
        return e / e.norm(dim=-1, keepdim=True)


def _encode_texts(model, tokenizer, texts: list, device: str) -> torch.Tensor:
    batch_size = 256
    all_embs = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i:i+batch_size]).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0).to(device)


def _encode_dino(model, proc, img: Image.Image, device: str) -> torch.Tensor:
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        e = out.last_hidden_state[:, 0, :]
        return e / e.norm(dim=-1, keepdim=True)


# ── Etapa 1 ────────────────────────────────────────────────────────────────────

def stage1(query_img: Image.Image, species_db: list, encoders: dict,
           device: str, top_m: int) -> list:
    print(f"\n{'─'*60}")
    print("  ETAPA 1 — Recuperacion cross-modal (BioCLIP + CLIP)")
    print(f"{'─'*60}")

    descriptions = [e["description"] for e in species_db]
    all_scores = {}

    for name, (model, preprocess, tokenizer) in encoders.items():
        q_emb  = _encode_img(model, preprocess, query_img, device)
        t_embs = _encode_texts(model, tokenizer, descriptions, device)
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
        print(f"  {r['rank']}. {r['species']['name']:<40} ensemble={r['score_ensemble']:.4f}{subrep}")
    return results


# ── Etapa 2 ────────────────────────────────────────────────────────────────────

def stage2(query_img: Image.Image, candidates: list, dino_model, dino_proc,
           device: str, lam: float, top_k: int) -> list:
    print(f"\n{'─'*60}")
    print(f"  ETAPA 2 — Re-ranking visual DINOv2  (lambda={lam})")
    print(f"{'─'*60}")

    q_dino = _encode_dino(dino_model, dino_proc, query_img, device)

    reranked = []
    for cand in candidates:
        sp = cand["species"]
        anchor_scores = []
        for img_path in sp.get("image_anchors", [sp["image_url"]]):
            try:
                a_dino = _encode_dino(dino_model, dino_proc,
                                      Image.open(img_path).convert("RGB"), device)
                anchor_scores.append((q_dino @ a_dino.T).item())
            except Exception:
                pass
        s_dino  = max(anchor_scores) if anchor_scores else 0.0
        s_final = lam * cand["score_ensemble"] + (1 - lam) * s_dino
        reranked.append({**cand, "score_dino": s_dino, "score_final": s_final})

    reranked.sort(key=lambda x: x["score_final"], reverse=True)
    for i, r in enumerate(reranked):
        r["rank_final"] = i + 1

    print(f"\n  {'Especie':<40} {'s_cross':>8} {'s_dino':>8} {'s_final':>8}  Cambio")
    print("  " + "-" * 72)
    for r in reranked:
        delta = r["rank"] - r["rank_final"]
        ch    = f"+{delta}" if delta > 0 else (str(delta) if delta < 0 else "=")
        top   = " <- TOP1" if r["rank_final"] == 1 else ""
        print(f"  {r['species']['name']:<40} "
              f"{r['score_ensemble']:>8.4f} {r['score_dino']:>8.4f} "
              f"{r['score_final']:>8.4f}  {ch:>5}{top}")
    return reranked[:top_k]


# ── Etapa 3 ────────────────────────────────────────────────────────────────────

def stage3(top_k: list, image_url: str = "") -> dict:
    print(f"\n{'─'*60}")
    print("  ETAPA 3 — Razonamiento LMM (DeepSeek)")
    print(f"{'─'*60}")

    # Construir contexto con los top-k candidatos y sus scores
    candidates_text = ""
    for i, r in enumerate(top_k, 1):
        sp = r["species"]
        subrep = f" [SUBREPRESENTADA: {sp['dataset_records']} registros]" if sp["subrepresented"] else ""
        candidates_text += (
            f"\nCandidato {i} (score visual={r['score_final']:.4f}):\n"
            f"  Especie: {sp['name']}{subrep}\n"
            f"  Familia: {sp['family']} | Orden: {sp['order']}\n"
            f"  Descripcion: {sp['description']}\n"
        )

    img_ref = f"\nURL de la imagen de consulta: {image_url}" if image_url else ""

    prompt = (
        "Eres un entomólogo experto en insectos amazónicos y andinos de Perú y países vecinos.\n"
        f"{img_ref}\n"
        "Se te proporciona una lista de especies candidatas ordenadas por similitud visual "
        "(score más alto = más similar a la imagen):\n"
        f"{candidates_text}\n"
        "Basándote en los scores de similitud visual, las descripciones taxonómicas "
        "y tu conocimiento de entomología neotropical, responde:\n"
        "1. ¿Cuál es la especie más probable? Justifica con rasgos morfológicos clave.\n"
        "2. ¿Qué tan confiable es la identificación (alta/media/baja)? ¿Por qué?\n"
        "3. Si la especie está marcada como subrepresentada, ¿qué implicaciones tiene "
        "para la biodiversidad amazónica?\n"
        "Responde en español, de forma concisa."
    )

    print(f"\n  Llamando a {LMM_MODEL}...")
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY,
        )
        completion = client.chat.completions.create(
            model=LMM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            top_p=0.95,
            max_tokens=1024,
            extra_body={
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "high",
                }
            },
            stream=False,
        )
        reasoning = (
            getattr(completion.choices[0].message, "reasoning", None) or
            getattr(completion.choices[0].message, "reasoning_content", None)
        )
        response  = completion.choices[0].message.content

        if reasoning:
            print(f"\n  [Razonamiento interno]")
            for line in reasoning.strip().split("\n")[:8]:
                print(f"  {line}")
            print("  ...")

        print(f"\n  [Respuesta LMM]")
        for line in response.strip().split("\n"):
            print(f"  {line}")

        # Especie predicha por el LMM: tomar la del top-1 por score
        # (el LMM puede confirmar o rechazar, pero la métrica se basa en ranking)
        predicted = top_k[0]["species"]["name"]

        return {
            "predicted":  predicted,
            "lmm_response": response,
            "reasoning":  reasoning or "",
        }

    except Exception as e:
        print(f"\n  ERROR en LMM: {e}")
        print("  Usando top-1 por score como fallback.")
        best = top_k[0]["species"]
        return {
            "predicted":    best["name"],
            "lmm_response": f"ERROR: {e}",
            "reasoning":    "",
        }


# ── Metricas ───────────────────────────────────────────────────────────────────

def mrr_at_k(results: list, target: str, k: int) -> float:
    for i, r in enumerate(results[:k]):
        if r["species"]["name"].strip().lower() == target.strip().lower():
            return 1.0 / (i + 1)
    return 0.0


def taxon_accuracy_at_k(results: list, target_taxon: str, level: str, k: int) -> float:
    """Fraccion de los top-k candidatos que comparten order/family con target_taxon."""
    matches = sum(
        1 for r in results[:k]
        if r["species"].get(level, "").strip().lower() == target_taxon.strip().lower()
    )
    return matches / min(k, len(results))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-url", help="URL de imagen de prueba")
    group.add_argument("--image",     help="Ruta local de imagen de prueba")
    parser.add_argument("--target-species", default=None)
    parser.add_argument("--top-m", type=int, default=30)   # Etapa 1 → top-30 (paper)
    parser.add_argument("--top-k", type=int, default=10)   # Etapa 2 → top-10 (paper)
    parser.add_argument("--lam",   type=float, default=0.7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  VR-RAG Insecta — Test")
    print(f"  Dispositivo: {device}")
    print(f"{'='*60}\n")

    species_db = load_species_db()

    if args.image_url:
        query_img = download_test_image(args.image_url)
    else:
        query_img = Image.open(args.image).convert("RGB")
        print(f"Imagen de prueba: {args.image}")

    encoders   = load_encoders(device)
    dino_model, dino_proc = load_dino(device)

    candidates  = stage1(query_img, species_db, encoders, device, args.top_m)
    top_k       = stage2(query_img, candidates, dino_model, dino_proc,
                         device, args.lam, args.top_k)
    lmm_result  = stage3(top_k, image_url=args.image_url or args.image or "")
    prediction  = lmm_result["predicted"]

    if args.target_species:
        print(f"\n{'─'*60}")
        print("  METRICAS")
        print(f"{'─'*60}")
        mrr1    = mrr_at_k(top_k, args.target_species, k=1)
        mrr10   = mrr_at_k(top_k, args.target_species, k=10)
        mrr_e1  = mrr_at_k(candidates, args.target_species, k=args.top_m)
        ok      = "CORRECTO" if mrr1 == 1.0 else "incorrecto"
        print(f"  Objetivo      : {args.target_species}")
        print(f"  Top-1         : {prediction}  ({ok})")
        print(f"  MRR@1         : {mrr1:.4f}")
        print(f"  MRR@10        : {mrr10:.4f}")
        print(f"  MRR@{args.top_m} (E1) : {mrr_e1:.4f}")

        target_entry = next(
            (e for e in species_db if e["name"].strip().lower() == args.target_species.strip().lower()),
            None,
        )
        if target_entry:
            order_acc  = taxon_accuracy_at_k(top_k, target_entry["order"],  "order",  k=args.top_k)
            family_acc = taxon_accuracy_at_k(top_k, target_entry["family"], "family", k=args.top_k)
            print(f"  Order-acc@{args.top_k}  : {order_acc:.2%}  ({target_entry['order']})")
            print(f"  Family-acc@{args.top_k} : {family_acc:.2%}  ({target_entry['family']})")
        else:
            print(f"  AVISO: '{args.target_species}' no esta en species_db, no se puede calcular order/family-acc")

        if mrr1 > mrr_e1:
            print("  -> Re-ranking mejoro la posicion del resultado correcto")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
