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


# ── Main ─────────────────────────────────────────────────────────────────────

def main(muestra=-1,partitions=5):
    #cargamos el device y la base de datos de especies
    device = "cuda" if torch.cuda.is_available() else "cpu"
    species_db = load_species_db()

    #extraemos las descripciones y los índices
    idxs = [e["id"] for e in species_db]
    descriptions = [e["description"] for e in species_db]

    #invocamos a los encoders para obtener los embeddings textuales
    encoders   = load_encoders(device)
    T_EMBS = {}
    for name, (model, preprocess, tokenizer) in encoders.items():
            descripciones_a_procesar = descriptions[:muestra] if muestra > 0 else descriptions
            t_embs = _encode_texts(model, tokenizer, descripciones_a_procesar, device)
            T_EMBS[name] = t_embs.cpu()

    # para evitar problemas con winutils y pyspark
    
    os.environ["HADOOP_HOME"] = rf"C:\Users\jc.ruedah\hadoop"
    os.environ["PATH"] += r";C:\Users\jc.ruedah\hadoop\bin"

    #Spark session
    spark = (
        SparkSession.builder
        .appName("EMBEDDINGS_TEXTUALES")
        .master("local[*]")
        # .config("spark.hadoop.hadoop.native.lib","false")
        # .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        # .config("spark.hadoop.fs.file.impl.disable.cache","true")
        .getOrCreate()
    )
    modelos = list(encoders.keys())
    for modelo in modelos:
        print(f"Embeddings para {modelo}: {T_EMBS[modelo].shape}")

        #elegimos la cantidad de descripciones a procesar 
        if muestra > 0:
            cantidad_descripciones = len(descriptions[:muestra])
        else:
            cantidad_descripciones = len(descriptions)

        # ---cargando embeddings a silver
        emb_np = T_EMBS[modelo].numpy() 
        data = [(
            int(i), idxs[i],descriptions[i], emb_np[i].tolist()
        ) for i in range(cantidad_descripciones)]
        #diseñamos el esquema del dataframe
        schema = StructType([
            StructField("id_int", IntegerType(), False),
            StructField("id_str", StringType(), False),
            StructField("description", StringType(), False),
            StructField("embedding", ArrayType(FloatType()), False)
        ]) 
        df = spark.createDataFrame(data, schema)
        #particionamos el dataframe para mejorar el rendimiento de escritura
        df = df.repartition(partitions)

        ruta_silver_distribuido = Path(__file__).parent.parent / "data" / "silver" / "embeddings_textuales_distribuidos" / modelo
        os.makedirs(ruta_silver_distribuido, exist_ok=True)        
        (df.write
            .mode("overwrite")
            .parquet(str(ruta_silver_distribuido))
        )



if __name__ == "__main__":
    main(muestra=-1, partitions=100)



