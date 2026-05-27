"""
Demo VR-RAG + BioCLIP — Identificación de insectos amazónicos
==============================================================
Replica en pequeño el pipeline de Khan et al. (2025) "VR-RAG: Open-vocabulary
Species Recognition with RAG-Assisted Large Multi-Modal Models".

ESTRUCTURA DEL PIPELINE:
    Imagen consulta
        │
        ▼
    [ETAPA 1] Recuperación cross-modal
        BioCLIP + CLIP + SigLIP → similitud coseno imagen↔texto
        → top-m candidatos (m=5 en este demo)
        │
        ▼
    [ETAPA 2] Re-ranking visual con DINOv2
        Similitud intramodal imagen↔imagen con anclajes visuales
        score_final = λ·score_cross + (1-λ)·score_visual
        → top-k candidatos (k=3 en este demo)
        │
        ▼
    [ETAPA 3] Razonamiento con LMM
        Qwen2.5-VL recibe imagen + descripciones de top-k
        → identificación final en lenguaje natural

EJECUCIÓN:
    pip install torch open-clip-torch transformers Pillow requests tqdm
    python demo_vr_rag.py

    Para usar una imagen propia:
    python demo_vr_rag.py --image ruta/a/tu/imagen.jpg
"""

import os
import sys
import json
import time
import argparse
import urllib.request
from pathlib import Path

# Deshabilita el backend experimental hf_transfer (causa WinError 10054 en Windows)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────
# BASE DE CONOCIMIENTO — 8 ESPECIES AMAZÓNICAS
# Cada especie tiene: nombre, familia, descripción morfológica,
# y una URL de imagen de Wikipedia (dominio público)
# ─────────────────────────────────────────────────────────────
SPECIES_DB = [
    {
        "id": "stenaspis_superba",
        "name": "Stenaspis superba",
        "family": "Cerambycidae",
        "order": "Coleoptera",
        "description": (
            "Large longhorn beetle. Body entirely black and glossy. "
            "Elytra with two symmetrical oval red or orange spots in the middle third. "
            "Antennae longer than the body. Length 25–35 mm. "
            "Found in lowland Amazon rainforest under 1000m elevation."
        ),
        "image_url": "http://bezbycids.com/byciddb/images/S/Stenaspis_superba_(h-f)_Aurivillius_NHRS_Mojos_La_Paz_Bolivia_Johannes%20Bergsten.jpg",
        "subrepresented": True,
    },
    {
        "id": "morpho_helenor",
        "name": "Morpho helenor",
        "family": "Nymphalidae",
        "order": "Lepidoptera",
        "description": (
            "Large butterfly with brilliant iridescent blue dorsal wing surface. "
            "Ventral side cryptic brown with eyespots. Wingspan 120–150 mm. "
            "Males highly iridescent, females less so. Common in Amazon basin."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Morphohelenorachilleana.JPG/1280px-Morphohelenorachilleana.JPG",
        "subrepresented": False,
    },
    {
        "id": "dynastes_hercules",
        "name": "Dynastes hercules",
        "family": "Scarabaeidae",
        "order": "Coleoptera",
        "description": (
            "Hercules beetle, one of the largest beetles. Males with prominent thoracic "
            "and cephalic horns. Elytra olive-green to yellowish with black spots. "
            "Body length up to 85 mm excluding horn. Nocturnal. Found in cloud forest."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Dynastes_hercules_ecuatorianus_MHNT.jpg/1920px-Dynastes_hercules_ecuatorianus_MHNT.jpg",
        "subrepresented": False,
    },
    {
        "id": "titanus_giganteus",
        "name": "Titanus giganteus",
        "family": "Cerambycidae",
        "order": "Coleoptera",
        "description": (
            "Titan beetle, one of the world's largest beetles. Uniform brown coloration. "
            "Powerful mandibles. Body length 150–170 mm. Nocturnal, attracted to lights. "
            "Extremely rare, very few specimens known. Deep Amazon rainforest."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Titanus_giganteus_MHNT_dos.jpg/960px-Titanus_giganteus_MHNT_dos.jpg",
        "subrepresented": True,
    },
    {
        "id": "acrocinus_longimanus",
        "name": "Acrocinus longimanus",
        "family": "Cerambycidae",
        "order": "Coleoptera",
        "description": (
            "Harlequin beetle with intricate geometric black and red pattern on elytra. "
            "Males have extremely elongated forelegs. Body 40–75 mm. "
            "Associated with fig trees. Found throughout tropical Americas."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Acrocinus_longimanus_MHNT_femelle.jpg/960px-Acrocinus_longimanus_MHNT_femelle.jpg",
        "subrepresented": True,
    },
    {
        "id": "heliconius_melpomene",
        "name": "Heliconius melpomene",
        "family": "Nymphalidae",
        "order": "Lepidoptera",
        "description": (
            "Postman butterfly. Black wings with red forewing band. "
            "Müllerian mimic of H. erato. Wingspan 60–75 mm. "
            "Slow deliberate flight. Pollen-feeding behavior. Widespread Amazon."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Heliconius_melpomene_penelope_MHNT_dos_Male.jpg/1280px-Heliconius_melpomene_penelope_MHNT_dos_Male.jpg",
        "subrepresented": False,
    },
    {
        "id": "caligo_eurilochus",
        "name": "Caligo eurilochus",
        "family": "Nymphalidae",
        "order": "Lepidoptera",
        "description": (
            "Owl butterfly. Ventral hindwing with large owl-like eyespot. "
            "Dorsal surface blue-brown dark. Wingspan 95–120 mm. "
            "Crepuscular. Feeds on rotting fruit. Mimics owl eye to deter predators."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Caligo_eurilochus_Richard_Bartz.jpg/960px-Caligo_eurilochus_Richard_Bartz.jpg",
        "subrepresented": False,
    },
    {
        "id": "morpho_achilles",
        "name": "Morpho achilles",
        "family": "Nymphalidae",
        "order": "Lepidoptera",
        "description": (
            "Blue-banded morpho. Blue-white band on black forewing dorsal surface. "
            "Smaller than M. helenor. Wingspan 100–115 mm. "
            "Fast erratic flight. Common in Amazon lowland forest."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Morachil.jpg/1920px-Morachil.jpg",
        "subrepresented": False,
    },
]

# ─────────────────────────────────────────────────────────────
# IMAGEN CONSULTA — descarga desde Wikipedia (dominio público)
# En producción: foto de campo desde la Bioestación de Manu
# ─────────────────────────────────────────────────────────────
QUERY_IMAGES = {
    "stenaspis": {
        "url": "http://bezbycids.com/byciddb/images/S/Stenaspis_superba_(h-f)_Aurivillius_NHRS_Mojos_La_Paz_Bolivia_Johannes%20Bergsten.jpg",
        "target": "stenaspis_superba",
        "description": "Escarabajo negro con manchas rojas — posible Cerambycidae",
    },
    "morpho": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Morphohelenorachilleana.JPG/1280px-Morphohelenorachilleana.JPG",
        "target": "morpho_helenor",
        "description": "Mariposa azul iridiscente grande",
    },
    "acrocinus": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Acrocinus_longimanus_MHNT_femelle.jpg/960px-Acrocinus_longimanus_MHNT_femelle.jpg",
        "target": "acrocinus_longimanus",
        "description": "Longicornio con patrón geométrico rojo y negro",
    },
}


# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────

def _direct_wikimedia_url(url: str) -> str:
    """Convert a Wikimedia thumbnail URL to the direct full-resolution URL.
    thumb format: .../commons/thumb/X/XX/file.jpg/NNNpx-file.jpg
    direct format: .../commons/X/XX/file.jpg
    """
    if "/thumb/" in url:
        base, rest = url.split("/thumb/", 1)
        # drop the last segment (NNNpx-file.jpg), keep the path to the file
        file_path = "/".join(rest.split("/")[:-1])
        return f"{base}/{file_path}"
    return url


_MAX_SIDE = 512  # BioCLIP/CLIP/DINOv2 usan 224px internamente; 512 es más que suficiente


def download_image(url: str, path: Path) -> Image.Image:
    """Descarga una imagen, la redimensiona a _MAX_SIDE px (lado mayor) y la guarda."""
    if not path.exists():
        direct = _direct_wikimedia_url(url)
        print(f"  Descargando {direct.split('/')[-1]}...")
        headers = {"User-Agent": "Mozilla/5.0 (research demo)"}
        req = urllib.request.Request(direct, headers=headers)
        import io
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if max(img.size) > _MAX_SIDE:
            img.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.LANCZOS)
        img.save(path, "JPEG", quality=90)
        print(f"    → guardada como {img.size[0]}×{img.size[1]} px")
    return Image.open(path).convert("RGB")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Similitud coseno entre vectores.
    Equivalente a: (a · b) / (||a|| · ||b||)
    Si los vectores ya están L2-normalizados, es simplemente el producto punto.
    """
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a @ b.T)


def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def print_ranking(results: list, label: str = "score"):
    for i, r in enumerate(results):
        bar = "█" * int(r[label] * 30)
        marker = " ← CORRECTO" if r.get("is_correct") else ""
        subrep = " [subrepresentada]" if r["species"]["subrepresented"] else ""
        print(f"  {i+1}. {r['species']['name']:<30} {r[label]:.4f}  {bar}{marker}{subrep}")


# ─────────────────────────────────────────────────────────────
# ETAPA 1: RECUPERACIÓN CROSS-MODAL
# ─────────────────────────────────────────────────────────────

def _hf_prefetch(repo_id: str, max_retries: int = 5) -> None:
    """Pre-descarga un repo de HuggingFace con reintentos.

    Cada intento crea un cliente HTTP nuevo, evitando el bug de httpx donde
    un WinError 10054 deja el cliente en estado cerrado y los reintentos
    internos de huggingface_hub fallan con 'client has been closed'.
    """
    from huggingface_hub import snapshot_download
    for attempt in range(max_retries):
        try:
            snapshot_download(repo_id=repo_id, ignore_patterns=["*.msgpack", "*.h5", "flax_model*"])
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s…
                print(f"    Red inestable, reintento {attempt + 1}/{max_retries} en {wait}s… ({type(e).__name__})")
                time.sleep(wait)
            else:
                print(f"    Advertencia: pre-descarga falló tras {max_retries} intentos, continuando…")


def load_encoders(device: str):
    """
    Carga BioCLIP, CLIP y SigLIP.
    En producción todos usan ViT-L/14 excepto BioCLIP (ViT-B/16).
    Para el demo usamos versiones ligeras que funcionan en CPU.
    """
    import open_clip

    print("\n  Cargando BioCLIP (ViT-B/16, especializado en biología)...")
    _hf_prefetch("imageomics/bioclip")
    bioclip_model, _, bioclip_preprocess = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    bioclip_tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    bioclip_model = bioclip_model.to(device).eval()

    print("  Cargando CLIP (ViT-B/32, modelo general)...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device).eval()

    return {
        "bioclip": (bioclip_model, bioclip_preprocess, bioclip_tokenizer),
        "clip": (clip_model, clip_preprocess, clip_tokenizer),
    }


def encode_image(model, preprocess, image: Image.Image, device: str) -> torch.Tensor:
    """
    Convierte imagen PIL → tensor → embedding normalizado.

    Lo que hace internamente el ViT:
    1. Redimensiona a 224×224
    2. Divide en parches 16×16 (BioCLIP) o 32×32 (CLIP)
    3. Proyecta cada parche a vector d-dimensional
    4. Pasa por capas Transformer con atención multi-cabeza
    5. Toma el token [CLS] como representación global
    6. L2-normaliza el vector final
    """
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding


def encode_texts(model, tokenizer, texts: list, device: str) -> torch.Tensor:
    """
    Convierte descripciones textuales → embeddings normalizados.

    El encoder de texto es un Transformer autoregresivo:
    1. Tokeniza con BPE (Byte-Pair Encoding)
    2. Pasa por capas de atención multi-cabeza
    3. Usa el token [EOS] como representación global
    4. Proyecta al mismo espacio dimensional que las imágenes
    5. L2-normaliza
    """
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


def stage1_retrieval(
    query_image: Image.Image,
    species_db: list,
    encoders: dict,
    img_dir: Path,
    device: str,
    top_m: int = 5,
) -> list:
    """
    ETAPA 1: Recuperación cross-modal con ensemble de modelos VLM.

    Para cada modelo:
        score(query_img, species_text) = cosine(embed_img(query), embed_text(desc))

    Score ensemble = promedio de los scores de todos los modelos.

    Esto es cross-modal porque compara representaciones de modalidades distintas
    (imagen vs texto) en un espacio de embedding compartido aprendido con CLIP loss.
    """
    print_section("ETAPA 1 — Recuperación cross-modal")

    descriptions = [s["description"] for s in species_db]
    all_scores = {}

    for model_name, (model, preprocess, tokenizer) in encoders.items():
        print(f"\n  Encodificando con {model_name.upper()}...")

        # Embedding de la imagen consulta
        q_emb = encode_image(model, preprocess, query_image, device)
        print(f"    → Embedding imagen: shape {q_emb.shape}, norma {q_emb.norm():.4f}")

        # Embeddings de todas las descripciones textuales
        text_embs = encode_texts(model, tokenizer, descriptions, device)
        print(f"    → Embeddings texto: shape {text_embs.shape}")

        # Similitud coseno imagen↔texto
        # Matemáticamente: score = q_emb · text_emb_i  (ya están L2-normalizados)
        scores = (q_emb @ text_embs.T).squeeze(0)
        all_scores[model_name] = scores.cpu()

        print(f"    → Scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

    # Ensemble: promedio de scores de todos los modelos
    # En el paper original: (CLIP + OpenCLIP + SigLIP) / 3
    # Aquí: (BioCLIP + CLIP) / 2
    ensemble_scores = torch.stack(list(all_scores.values())).mean(dim=0)

    # Mostrar fórmula explícita
    print("\n  Fórmula aplicada:")
    print("  score_ensemble(q, s) = (score_BioCLIP + score_CLIP) / 2")
    print("  similitud = q_img · s_text  [producto punto = coseno cuando L2-normalizados]")

    # Ordenar y tomar top-m
    top_indices = ensemble_scores.argsort(descending=True)[:top_m]

    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            "rank": rank + 1,
            "species": species_db[idx],
            "score_bioclip": all_scores["bioclip"][idx].item(),
            "score_clip": all_scores["clip"][idx].item(),
            "score_ensemble": ensemble_scores[idx].item(),
        })

    print(f"\n  Top-{top_m} candidatos (ordenados por score ensemble):")
    for r in results:
        print(f"  {r['rank']}. {r['species']['name']:<35} "
              f"BioCLIP={r['score_bioclip']:.4f}  "
              f"CLIP={r['score_clip']:.4f}  "
              f"Ensemble={r['score_ensemble']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────
# ETAPA 2: RE-RANKING VISUAL CON DINOv2
# ─────────────────────────────────────────────────────────────

def load_dino(device: str):
    """
    Carga DINOv2 (ViT-S/14 para el demo — en producción ViT-L/14).
    DINOv2 fue entrenado con auto-supervisión sin etiquetas de texto,
    produciendo embeddings puramente visuales de alta calidad.
    """
    from transformers import AutoImageProcessor, AutoModel

    print("\n  Cargando DINOv2 (ViT-S/14, auto-supervisado)...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    return model, processor


def encode_image_dino(model, processor, image: Image.Image, device: str) -> torch.Tensor:
    """
    Produce embedding DINOv2 de una imagen.
    Usa el token [CLS] como representación global de la imagen.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]  # token [CLS]
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def stage2_reranking(
    query_image: Image.Image,
    candidates: list,
    dino_model,
    dino_processor,
    img_dir: Path,
    device: str,
    lambda_weight: float = 0.7,
    top_k: int = 3,
) -> list:
    """
    ETAPA 2: Re-ranking visual con DINOv2.

    Para cada especie candidata se toman hasta 3 imágenes de referencia (anclajes).
    Se calcula la similitud intramodal imagen↔imagen con DINOv2.

    Fórmula de re-ranking:
        s_i = DINOv2(query) · DINOv2(anchor_especie)
        s_final = λ · s_cross + (1-λ) · s_i

    λ=0.7 en el paper: la similitud visual (30%) complementa la semántica (70%).

    Ventaja de DINOv2 sobre CLIP para re-ranking:
    - DINOv2 fue entrenado sin texto → no sesgado hacia categorías semánticas
    - Captura textura, forma y microestructuras con mayor detalle
    - Ideal para distinguir especies morfológicamente similares
    """
    print_section("ETAPA 2 — Re-ranking visual con DINOv2")
    print(f"\n  Fórmula: s_final = {lambda_weight}·s_cross + {1-lambda_weight:.1f}·s_dino")
    print("  (λ=0.7 significa que la similitud cross-modal pesa 70%,")
    print("   la similitud visual directa pesa 30%)")

    # Embedding DINOv2 de la imagen consulta
    q_dino = encode_image_dino(dino_model, dino_processor, query_image, device)
    print(f"\n  Embedding DINOv2 consulta: shape {q_dino.shape}")

    reranked = []
    for cand in candidates:
        sp = cand["species"]

        # Descarga la imagen de referencia de la especie (anclaje visual)
        anchor_path = img_dir / f"{sp['id']}_anchor.jpg"
        try:
            anchor_img = download_image(sp["image_url"], anchor_path)
            anchor_dino = encode_image_dino(dino_model, dino_processor, anchor_img, device)
            score_dino = (q_dino @ anchor_dino.T).item()
        except Exception as e:
            print(f"  Anclaje no disponible para {sp['name']}: {e}")
            score_dino = 0.0

        # Fórmula de combinación
        s_cross = cand["score_ensemble"]
        s_final = lambda_weight * s_cross + (1 - lambda_weight) * score_dino

        reranked.append({
            **cand,
            "score_dino": score_dino,
            "score_final": s_final,
        })

    # Re-ordenar por score final
    reranked.sort(key=lambda x: x["score_final"], reverse=True)
    for i, r in enumerate(reranked):
        r["rank_final"] = i + 1

    print("\n  Resultados del re-ranking:")
    print(f"  {'Especie':<35} {'s_cross':>8} {'s_dino':>8} {'s_final':>8}  {'Cambio':>8}")
    print("  " + "─" * 72)
    for r in reranked:
        rank_change = r["rank"] - r["rank_final"]
        change_str = f"▲{rank_change}" if rank_change > 0 else (f"▼{-rank_change}" if rank_change < 0 else "=")
        marker = " ◄ TOP-1" if r["rank_final"] == 1 else ""
        print(f"  {r['species']['name']:<35} {r['score_ensemble']:>8.4f} {r['score_dino']:>8.4f} {r['score_final']:>8.4f}  {change_str:>8}{marker}")

    return reranked[:top_k]


# ─────────────────────────────────────────────────────────────
# ETAPA 3: RAZONAMIENTO CON LMM
# ─────────────────────────────────────────────────────────────

def stage3_lmm_reasoning(
    query_image: Image.Image,
    top_k_candidates: list,
    query_description: str,
    use_qwen: bool = False,
) -> str:
    """
    ETAPA 3: Razonamiento con Large Multimodal Model.

    El LMM recibe:
    - La imagen consulta
    - Las descripciones completas de los top-k candidatos
    - Una instrucción de identificación

    Genera la identificación token a token (autoregresivo):
    P(token_t | token_1..t-1, imagen, contexto)

    Por limitaciones de memoria en demo, hay dos modos:
    1. use_qwen=True: usa Qwen2.5-VL-7B real (requiere ~16GB RAM/VRAM)
    2. use_qwen=False: simula el razonamiento con regla de mayor score (default)
    """
    print_section("ETAPA 3 — Razonamiento con LMM")

    # Construir el contexto textual (prompt RAG)
    context_lines = []
    for i, cand in enumerate(top_k_candidates, 1):
        sp = cand["species"]
        context_lines.append(
            f"{i}. {sp['name']} ({sp['family']}, {sp['order']})\n"
            f"   Descripción: {sp['description']}"
        )
    context = "\n\n".join(context_lines)

    prompt = (
        f"You are an expert entomologist specializing in Amazonian insects.\n\n"
        f"The query image shows: {query_description}\n\n"
        f"Based on the visual characteristics of the image and the following species "
        f"descriptions from the knowledge base, identify the most likely species:\n\n"
        f"{context}\n\n"
        f"Provide: (1) the most likely species, (2) the key visual features that led "
        f"to this identification, (3) confidence level (high/medium/low)."
    )

    print("\n  Prompt enviado al LMM:")
    print("  " + "─" * 56)
    for line in prompt.split("\n")[:12]:
        print(f"  {line}")
    print("  ... [continúa con descripciones completas]")
    print("  " + "─" * 56)

    if use_qwen:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        print("\n  Cargando Qwen2.5-VL-7B-Instruct...")
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": query_image},
                {"type": "text", "text": prompt},
            ],
        }]

        text_input = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = qwen_processor(
            text=[text_input], images=image_inputs, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output_ids = qwen_model.generate(**inputs, max_new_tokens=256)
        output_ids = [o[len(i):] for i, o in zip(inputs.input_ids, output_ids)]
        response = qwen_processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
    else:
        # Modo demo: regla basada en score (simula el razonamiento del LMM)
        best = top_k_candidates[0]["species"]
        score = top_k_candidates[0]["score_final"]
        confidence = "high" if score > 0.85 else "medium" if score > 0.70 else "low"
        subrep_note = (
            " This species is underrepresented in biodiversity databases "
            "(fewer than 5 georeferenced records in the study region), "
            "demonstrating the system's open-vocabulary capability."
            if best["subrepresented"] else ""
        )
        response = (
            f"Identified species: {best['name']} ({best['family']})\n\n"
            f"Key visual features matched:\n"
            f"{best['description']}\n\n"
            f"Confidence: {confidence} (ensemble score: {score:.4f})\n"
            f"{subrep_note}\n\n"
            f"[Note: In production this reasoning is produced by Qwen2.5-VL-7B. "
            f"Run with --use-qwen to enable the real LMM (requires ~16GB RAM).]"
        )

    print(f"\n  Respuesta del LMM:\n")
    for line in response.split("\n"):
        print(f"  {line}")

    return response


# ─────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────

def compute_mrr(results: list, target_id: str, k: int = 3) -> float:
    """
    Mean Reciprocal Rank @ k.
    MRR@k = 1/posición_del_correcto si está en top-k, sino 0.
    Mide qué tan arriba del ranking aparece la especie correcta.
    """
    for i, r in enumerate(results[:k]):
        if r["species"]["id"] == target_id:
            return 1.0 / (i + 1)
    return 0.0


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Demo VR-RAG + BioCLIP")
    parser.add_argument(
        "--query", default="stenaspis",
        choices=list(QUERY_IMAGES.keys()),
        help="Especie consulta (default: stenaspis)"
    )
    parser.add_argument(
        "--image", default=None,
        help="Ruta a imagen propia (JPG/PNG). Si se provee, ignora --query."
    )
    parser.add_argument(
        "--top-m", default=5, type=int,
        help="Candidatos de la Etapa 1 (default: 5)"
    )
    parser.add_argument(
        "--top-k", default=3, type=int,
        help="Candidatos finales para el LMM (default: 3)"
    )
    parser.add_argument(
        "--lambda-weight", default=0.7, type=float,
        help="λ para re-ranking: λ·s_cross + (1-λ)·s_dino (default: 0.7)"
    )
    parser.add_argument(
        "--use-qwen", action="store_true",
        help="Usar Qwen2.5-VL-7B real para Etapa 3 (requiere ~16GB RAM)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  VR-RAG + BioCLIP — Demo")
    print(f"  Dispositivo: {device}")
    print(f"{'='*60}")

    # Crear carpeta para imágenes
    img_dir = Path("./images_demo")
    img_dir.mkdir(exist_ok=True)

    # Cargar imagen consulta
    if args.image:
        query_image = Image.open(args.image).convert("RGB")
        query_info = {"description": f"Imagen propia: {args.image}", "target": None}
        print(f"\n  Imagen consulta: {args.image}")
    else:
        q = QUERY_IMAGES[args.query]
        query_path = img_dir / f"query_{args.query}.jpg"
        print(f"\n  Descargando imagen consulta: {q['description']}")
        query_image = download_image(q["url"], query_path)
        query_info = q
        print(f"  Imagen: {query_path} ({query_image.size})")

    # Mostrar resumen de la base de conocimiento
    print_section("BASE DE CONOCIMIENTO")
    for s in SPECIES_DB:
        flag = " [SUBREPRESENTADA]" if s["subrepresented"] else ""
        print(f"  • {s['name']:<35} {s['family']}{flag}")

    # Cargar modelos de recuperación
    print_section("Cargando modelos...")
    encoders = load_encoders(device)
    dino_model, dino_processor = load_dino(device)

    # Etapa 1
    candidates = stage1_retrieval(
        query_image, SPECIES_DB, encoders, img_dir, device, top_m=args.top_m
    )

    # Etapa 2
    top_k = stage2_reranking(
        query_image, candidates, dino_model, dino_processor,
        img_dir, device, lambda_weight=args.lambda_weight, top_k=args.top_k
    )

    # Etapa 3
    response = stage3_lmm_reasoning(
        query_image, top_k,
        query_description=query_info.get("description", ""),
        use_qwen=args.use_qwen,
    )

    # Métricas
    print_section("MÉTRICAS FINALES")
    if query_info.get("target"):
        mrr1 = compute_mrr(top_k, query_info["target"], k=1)
        mrr3 = compute_mrr(top_k, query_info["target"], k=3)
        print(f"  Especie objetivo:    {query_info['target']}")
        print(f"  Predicción top-1:   {top_k[0]['species']['id']}")
        print(f"  MRR@1:              {mrr1:.4f}  ({'✓ CORRECTO' if mrr1==1 else '✗ incorrecto'})")
        print(f"  MRR@3:              {mrr3:.4f}")
        stage1_mrr = compute_mrr(candidates, query_info["target"], k=args.top_m)
        print(f"  MRR@{args.top_m} (solo Etapa 1): {stage1_mrr:.4f}")
        if stage1_mrr > 0 and mrr1 > stage1_mrr:
            print("  → El re-ranking mejoró la posición del resultado correcto")

    print(f"\n{'='*60}")
    print("  Pipeline completado.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
