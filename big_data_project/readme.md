# Big Data Project: VR-RAG + BioCLIP para Identificación Multimodal de Artrópodos

## 📋 Descripción General

Este proyecto implementa un pipeline **VR-RAG (Visual Retrieval-Augmented Generation)** aplicado a la identificación de insectos y arañas amazónicos subrepresentados.

**Referencia:** Khan et al. (2025) - "VR-RAG: Open-vocabulary Species Recognition with RAG-Assisted Large Multi-Modal Models" ([arXiv:2505.05635](https://arxiv.org/abs/2505.05635))

### Motivación
- Muchas especies amazónicas son **subrepresentadas** en datasets públicos (menos registros en GBIF, pocas imágenes en iNaturalist)
- El pipeline opera en modo **open-vocabulary**: identifica especies no vistas en entrenamiento
- Usa **BioCLIP** (entrenado con taxonomía biológica) en lugar de CLIP estándar para mejor performance en artrópodos

---

## Arquitectura del Pipeline

El sistema opera en **3 etapas secuenciales**:

```
┌─────────────────────────────────────────────┐
│ IMAGEN DE CONSULTA (de especie subrepresentada)
└──────────────┬──────────────────────────────┘
               │
               ▼
    ╔══════════════════════════════════════╗
    ║  ETAPA 1: Recuperación Cross-Modal  ║
    ║  BioCLIP + CLIP + SigLIP             ║
    ║  → Similitud coseno imagen↔texto    ║
    ║  → Top-m candidatos (m=5)           ║
    ╚═════────────────┬────────────────────╝
                      │
                      ▼
    ╔══════════════════════════════════════╗
    ║  ETAPA 2: Re-ranking Visual DINOv2  ║
    ║  Similitud intramodal imagen↔imagen ║
    ║  → score_final = 0.7·cross + 0.3·dino
    ║  → Top-k candidatos (k=3)           ║
    ╚════────────────┬─────────────────────╝
                     │
                     ▼
    ╔══════════════════════════════════════╗
    ║  ETAPA 3: Razonamiento con LMM      ║
    ║  (Qwen2.5-VL o simulado)            ║
    ║  → Imagen + descripciones top-k     ║
    ║  → Identificación en lenguaje natural║
    ╚════────────────┬─────────────────────╝
                     │
                     ▼
          ┌──────────────────────┐
          │ RESULTADO FINAL      │
          │ (Especie + confianza)│
          └──────────────────────┘
```

---

## Estructura del Proyecto

### Carpeta `src/` - Scripts Principales

#### 1. **Construcción de Knowledge Bases**

**`build_kb_insecta.py`**
- Descarga insectos de GBIF (polígono Perú + alrededores)
- Filtra por órdenes (default: Lepidoptera, Coleoptera)
- Identifica especies subrepresentadas (< 50 registros)
- Descarga imágenes localmente
- **Output:** `data/bronze/kb_insecta/knowledge_base.json` + imágenes

**`build_kb_araneae.py`**
- Similar a anterior, pero para arañas (orden Araneae)
- Filtrado a Perú únicamente
- **Output:** `data/bronze/kb_araneae/`

#### 2. **Pipeline de Embeddings (Medallion Architecture)**

**Bronze → Silver (Procesamiento Distribuido con Spark)**

**`bronze_to_silver_text.py`**
- Carga descripciones de especies desde KB
- Codifica con BioCLIP + CLIP (modelos multimodales)
- Tokenización batch (256 textos/batch)
- Distribuye cálculo con Spark (5 particiones)
- **Output:** `data/silver/embeddings_textuales_distribuidos/{bioclip,clip}/*.parquet`

**`bronze_to_silver_img.py`**
- Carga imágenes de anclaje de especies
- Codifica con DINOv2-small (ViT-S/14)
- Normalizaciónde embeddings L2
- Particionado para paralelismo
- **Output:** `data/silver/embeddings_imagenes_distribuidas/dino/*.parquet`

**Silver → Gold (Optimización)**

**`silver_to_gold_text.py`**
- Lee embeddings textuales desde silver
- Re-particiona a 100 particiones para mejor performance
- **Output:** `data/gold/embeddings_textuales_distribuidos/{bioclip,clip}/*.parquet`

**`silver_to_gold_img.py`**
- Lee embeddings de imágenes desde silver
- Re-particiona a 100 particiones
- **Output:** `data/gold/embeddings_imagenes_distribuidas/dino/*.parquet`

#### 3. **Inferencia y Testing**

**`inference.py`**
- Pipeline principal de prueba
- Carga KB de especies
- Codifica texto con BioCLIP + CLIP
- Codifica imágenes con DINOv2
- Ejecuta las 3 etapas del VR-RAG
- **Requiere:** NVIDIA_API_KEY para LMM (DeepSeek via NVIDIA)

**`test_vr_rag_insecta.py`**
- Suite de tests con 14 especies candidatas
- Descarga imágenes de prueba desde URLs
- Calcula métricas: hit rate, MRR@k, precision
- Evaluación taxonómica por nivel (familia, orden)
- **Output:** `kb_insecta/batch_results.json`

---

## Flujo de Datos (Medallion Architecture)

```
GBIF Parquets
    ↓
[BRONZE LAYER]
  kb_insecta/
  ├── knowledge_base.json (metadata + rutas locales)
  ├── species_db.json (descripciones)
  └── images/ (imágenes descargadas)
    ↓
[SILVER LAYER] - Embeddings Distribuidos (Spark)
  data/silver/
  ├── embeddings_textuales_distribuidos/
  │   ├── bioclip/*.parquet
  │   └── clip/*.parquet
  └── embeddings_imagenes_distribuidas/
      └── dino/*.parquet
    ↓
[GOLD LAYER] - Embeddings Optimizados
  data/gold/
  ├── embeddings_textuales_distribuidos/
  │   ├── bioclip/*.parquet
  │   └── clip/*.parquet
  └── embeddings_imagenes_distribuidas/
      └── dino/*.parquet
```

---

## Dependencias

```
torch>=2.0.0                    # Deep learning
open-clip-torch>=2.24.0         # BioCLIP + CLIP
transformers>=4.40.0            # DINOv2 + modelos HuggingFace
Pillow>=10.0.0                  # Procesamiento de imágenes
numpy>=1.24.0                   # Operaciones numéricas
requests>=2.31.0                # Descargas HTTP
tqdm>=4.66.0                    # Barras de progreso
pyspark>=3.0.0                  # Procesamiento distribuido
pandas>=1.3.0                   # Manipulación de datos
```

**Instalación:**
```bash
pip install -r ../../requirements.txt
```

---

## Cómo Ejecutar

### Paso 1: Construir Knowledge Base (Primera vez)

```bash
# Insectos (Lepidoptera + Coleoptera)
python src/build_kb_insecta.py

# O todos los órdenes (~65k imágenes)
python src/build_kb_insecta.py --all-orders

# Arañas (Perú)
python src/build_kb_araneae.py --max-images 5
```

**Argumentos disponibles:**
- `--orders Lepidoptera,Coleoptera` - Órdenes específicas
- `--max-images N` - Máximo de imágenes por especie
- `--force-rebuild` - Ignora caché, reescribe todo
- `--all-orders` - Todos los órdenes de insectos

**Output esperado:**
- `kb_insecta/knowledge_base.json` (~100 KB)
- `kb_insecta/species_db.json` - Descripciones textuales
- `kb_insecta/images/` - Carpeta con imágenes por especie

### Paso 2: Procesar Embeddings (Pipelines de Big Data)

**Bronze → Silver (Genera embeddings distribuidos):**
```bash
# Embeddings textuales con BioCLIP + CLIP
python src/bronze_to_silver_text.py

# Embeddings de imágenes con DINOv2
python src/bronze_to_silver_img.py
```

**Silver → Gold (Optimiza para inferencia):**
```bash
# Consolidar embeddings textuales
python src/silver_to_gold_text.py

# Consolidar embeddings de imágenes
python src/silver_to_gold_img.py
```

### Paso 3: Ejecutar Inferencia

```bash
# Test del pipeline completo
python src/test_vr_rag_insecta.py \
    --image-url "https://example.com/image.jpg" \
    --top-m 5 \
    --top-k 3

# Con imagen local
python src/test_vr_rag_insecta.py \
    --image ruta/a/imagen.jpg \
    --target-species "Morpho helenor"
```

### Paso 4: Batch de Pruebas

```bash
# Ejecutar 14 especies candidatas (requiere esperar ~30min)
python ../prueba_20.py

# Solo mostrar resumen (sin re-correr)
python ../prueba_20.py --summary-only

# Personalizar parámetros
python ../prueba_20.py --top-m 30 --top-k 10 --rebuild-cache
```

---

## 📈 Decisiones de Diseño: Demo vs Producción

| Aspecto | Este Proyecto (Demo) | Producción (Paper) |
|---------|---------------------|-------------------|
| Encoder imagen | BioCLIP + CLIP (ViT-B) | CLIP + OpenCLIP + SigLIP (ViT-L) |
| Re-ranker visual | DINOv2-small (ViT-S/14) | DINOv2 ViT-L/14 |
| LMM | Simulado con scores | Qwen2.5-VL-7B real |
| Base de conocimiento | ~100-1000 especies | 11,202 especies |
| Anclajes visuales | 1-5 imágenes/especie | hasta 10 por especie |
| Requisitos RAM | 4-8 GB | 16+ GB |
| Velocidad | ~500ms/imagen | ~1s/imagen |


---

## Métricas de Evaluación

El pipeline genera métricas en `kb_insecta/batch_results.json`:

```json
{
  "species": "Argyractis zamoralis",
  "top_m_hits": [0, 1, 2, 4, 5],
  "top_k_recall": 2,
  "mrr_at_5": 0.5,
  "taxonomic_hits": {
    "family": "exact",
    "order": "exact"
  }
}
```

**Interpretación:**
- **top_m_hits** - Posición de hit en top-m candidatos (-1 = no encontrado)
- **MRR@5** - Mean Reciprocal Rank (penaliza si está muy abajo en ranking)
- **Taxonomic hits** - Precisión en niveles taxonómicos (family, order)

---

## Solución de Problemas Frecuentes

### Error: `OOM (Out of Memory)`
```
CUDA out of memory while trying to allocate...
```
**Solución:**
- Usar `--max-images 2` en build_kb
- Reducir batch_size en bronze_to_silver
- Usar DINOv2-base en lugar de ViT-L

### Error: `open_clip not found`
```bash
pip install open-clip-torch
```

### Error: `NVIDIA_API_KEY not found`
```bash
export NVIDIA_API_KEY="nvapi-..."  # Linux/Mac
set NVIDIA_API_KEY=nvapi-...       # Windows
```

### Lentitud en descargas de GBIF
- Las imágenes se cachean localmente tras primera ejecución
- Aumentar `DOWNLOAD_DELAY` si se rechazan conexiones
- Usar `--force-rebuild` solo cuando sea necesario

### Errores de Spark con Windows (Hadoop)
```
No such file or directory: C:\{path}\hadoop\bin\winutils.exe
```
**Solución:** Los scripts ya configuran `HADOOP_HOME` automáticamente, pero si persiste:
```bash
# Descargar winutils.exe desde github.com/steveloughran/winutils
# Colocar en C:\Users\{tu_usuario}\hadoop\bin\
```

---

## Referencia de Funciones Principales

### Encoders

**`load_encoders(device) → dict`**
- Carga BioCLIP + CLIP preentrenados
- Retorna modelos, preprocessadores, tokenizadores

**`load_dino(device) → tuple`**
- Carga DINOv2-small desde HuggingFace
- Retorna modelo e image processor

### Encoding

**`_encode_texts(model, tokenizer, texts, device) → torch.Tensor`**
- Batch embedding de textos (256/batch)
- Normalización L2
- Retorna tensor GPU/CPU

**`_encode_img(model, preprocess, img, device) → torch.Tensor`**
- Embedding de imagen CLIP
- Input: PIL Image
- Output: vector normalizado L2

**`_encode_dino(model, proc, img, device) → torch.Tensor`**
- Embedding de imagen DINOv2
- Extrae CLS token
- Normalización L2

### Pipeline

**`stage1(query_embedding, text_embeddings, top_m) → list`**
- Recuperación cross-modal
- Similitud coseno
- Retorna top-m índices + scores

**`stage2(img_embedding, dino_embeddings, top_m_indices) → list`**
- Re-ranking visual
- Combina: 0.7*cross + 0.3*dino
- Retorna top-k reordenados

**`stage3(query_img, descriptions, lmm_client) → str`**
- Razonamiento con LMM
- Input: imagen + descripciones top-k
- Output: identificación textual

---

## 🎯 Próximos Pasos / Mejoras Futuras

1. **Aumentar cobertura de KB**
   - Todos los órdenes de insectos (~65k imágenes)
   - Expandir a otros artrópodos (Arachnida, Myriapoda, Crustacea)

2. **Mejorar modelos**
   - Cambiar a DINOv2 ViT-L/14 (producción)
   - Usar Qwen2.5-VL-72B en lugar de simulación

3. **Validación en campo**
   - Pruebas con imágenes capturadas en Bioestación Manu
   - Evaluar con expertos en taxonomía

4. **Optimización de performance**
   - Cachear embeddings en FAISS
   - Compilar modelos con torch.compile
   - Quantización de embeddings (fp16)

5. **Documentación**
   - Agregar notebooks jupyter con ejemplos
   - Crear dashboard de métricas
   - Tutorial paso a paso

---

## Notas Técnicas

### Thresholds de Subrepresentación

- **Bronze (build_kb):** Especies con < 50 registros en polígono Perú
- **Global (build_kb_araneae):** Especies con < 100 registros globales
- **Criterios GBIF:** No identificadas, sin coordenadas, sin imágenes

### Configuración de Spark

- **Master:** local[*] (todos los cores)
- **Particiones Silver:** 5 (ingesta inicial)
- **Particiones Gold:** 100 (optimizado para query)
- **Modo write:** overwrite (sobrescribe cache)

### URLs de Imágenes

- **GBIF:** multimedia.parquet → identifier
- **Validación:** URLs deben comenzar con "http"
- **Descarga:** timeout=10s, delay=0.08s entre requests
- **Caché:** Imágenes existentes no se re-descargan

---

## 📖 Referencias Adicionales

- **Repo original:** https://github.com/faixan-khan/vr-rag
- **BioCLIP:** https://imageomics.github.io/bioclip/
- **GBIF API:** https://www.gbif.org/developer/
- **DINOv2:** https://github.com/facebookresearch/dinov2
- **Qwen2.5-VL:** https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

---

## Data utilizada
- Tener en cuenta al momento de replicar los resultados del proyecto:
    - GBIF.org (14 June 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.wrdgt4


