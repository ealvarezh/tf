import json
import os
import sys
import urllib.request
import io
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, BooleanType, ArrayType, MapType, LongType
import os 

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
from PIL import Image
from openai import OpenAI

KB_DIR          = Path(__file__).parent.parent / "data" / "bronze" / "kb_insecta"
KB_JSON         = KB_DIR / "knowledge_base.json"
SPECIES_DB_JSON = KB_DIR / "species_db.json"
TEST_IMG_PATH   = KB_DIR / "test_query.jpg"
_MAX_SIDE       = 512

# Funciones 

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


def load_dino(device: str):
    from transformers import AutoImageProcessor, AutoModel
    print("Cargando DINOv2...")
    proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    return model, proc

def _encode_dino(model, proc, img: Image.Image, device: str) -> torch.Tensor:
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        e = out.last_hidden_state[:, 0, :]
        return e / e.norm(dim=-1, keepdim=True)
# ── Main ─────────────────────────────────────────────────────────────────────

def main(muestra=-1,partitions=5):
    #cargamos el device y la base de datos de especies
    device = "cuda" if torch.cuda.is_available() else "cpu"
    species_db = load_species_db()

    #invocamos a los encoders para obtener los embeddings textuales
    dino_model, dino_proc = load_dino(device)
    ID_INT = []
    ID_SUB_IMG = []
    IDXS = []
    all_embs = []
    i=0
    if muestra > 0:
        species_db = species_db[:muestra]
    else:
        species_db = species_db


    for especie in species_db:
        
        print(f"Procesando especie: {especie['name']}")
        j=0
        for img_path in especie.get("image_anchors", [especie["image_url"]]):
            try:
                ID_INT.append(i)
                ID_SUB_IMG.append(j)
                IDXS.append(especie["id"])
                #print(f"Procesando imagen: {img_path}")
                dino_emb = _encode_dino(dino_model, dino_proc, Image.open(img_path).convert("RGB"), device)
                all_embs.append(dino_emb)
            except Exception as e:
                print(f"Error procesando imagen {img_path}: {e}")
                pass
            j += 1
        i += 1
    dino_embs = torch.cat(all_embs, dim=0).to(device).cpu().numpy()
    # para evitar problemas con winutils y pyspark
    
    os.environ["HADOOP_HOME"] = rf"C:\Users\jc.ruedah\hadoop"
    os.environ["PATH"] += r";C:\Users\jc.ruedah\hadoop\bin"

    #Spark session
    #Spark session
    spark = (
        SparkSession.builder
        .appName("EMBEDDINGS_IMAGENES")
        .master("local[*]")
        # .config("spark.hadoop.hadoop.native.lib","false")
        # .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        # .config("spark.hadoop.fs.file.impl.disable.cache","true")
        .getOrCreate()
    )


    data = []
    for i in range(len(species_db)):
        try:
            data.append((
                ID_INT[i], ID_SUB_IMG[i],IDXS[i], dino_embs[i].tolist()
            ))
        except Exception as e:
            print(f"Error creando fila para especie {species_db[i]['name']}: {e}")
            pass
    
    schema = StructType([
    StructField("id_int", IntegerType(), False),
    StructField("id_sub_img", IntegerType(), False),
    StructField("id_str", StringType(), False),
    StructField("embedding", ArrayType(FloatType()), False)
    ])
    df = spark.createDataFrame(data, schema)

    df = df.repartition(partitions)

    ruta_silver_distribuido = Path(__file__).parent.parent / "data" / "silver" / "embeddings_imagenes_distribuidas" / "dino"
    os.makedirs(ruta_silver_distribuido, exist_ok=True) 
    (df.write
        .mode("overwrite")
        .parquet(str(ruta_silver_distribuido))
    )


if __name__ == "__main__":
    main(muestra=-1, partitions=100)
