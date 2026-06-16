# VR-RAG + BioCLIP — Contexto del proyecto

## Qué es este proyecto

Demo del pipeline VR-RAG (Khan et al., 2025) aplicado a identificación de insectos 
amazónicos subrepresentados. Es parte de una tesis sobre identificación multimodal 
automatizada de artrópodos en la Amazonía peruana.

Paper base: "VR-RAG: Open-vocabulary Species Recognition with RAG-Assisted Large 
Multi-Modal Models" — arXiv:2505.05635
Repo original: https://github.com/faixan-khan/vr-rag
BioCLIP: https://imageomics.github.io/bioclip/

## El pipeline tiene 3 etapas

1. **RAG cross-modal**: Ensemble de BioCLIP + CLIP (simula CLIP+OpenCLIP+SigLIP del paper)
   - BioCLIP: ViT-B/16 entrenado en TreeOfLife-10M con etiquetas taxonómicas
   - Similitud coseno imagen↔texto en espacio compartido
   - Recupera top-m=5 candidatos

2. **Re-ranking visual con DINOv2**: 
   - DINOv2-small (ViT-S/14) para el demo, producción usa ViT-L/14
   - Similitud intramodal imagen↔imagen con anclajes visuales de referencia
   - s_final = 0.7*s_cross + 0.3*s_dino

3. **Razonamiento LMM**:
   - Qwen2.5-VL-7B-Instruct (modo demo simula con regla de score)
   - Recibe imagen + descripciones top-k, genera identificación

## Cómo ejecutar

```bash
# Instalar dependencias (solo la primera vez)
pip install -r requirements.txt

# Ejecutar con imagen de escarabajo Stenaspis (subrepresentada)
python demo_vr_rag.py --query stenaspis

# Con mariposa Morpho
python demo_vr_rag.py --query morpho

# Con imagen propia
python demo_vr_rag.py --image mi_foto.jpg

# Con Qwen2.5-VL real (necesitas ~16GB RAM)
python demo_vr_rag.py --query stenaspis --use-qwen
```

## Decisiones de diseño del demo vs producción

| Aspecto | Demo (este código) | Producción (paper) |
|---------|-------------------|-------------------|
| Encoder imagen | BioCLIP + CLIP (ViT-B) | CLIP+OpenCLIP+SigLIP (ViT-L) |
| Re-ranker | DINOv2-small | DINOv2 ViT-L/14 |
| LMM | Simulado con scores | Qwen2.5-VL-7B real |
| Base de conocimiento | 8 especies | 11,202 especies |
| Anclajes visuales | 1 imagen por especie | hasta 3 por especie |

## Por qué BioCLIP

BioCLIP fue entrenado específicamente con taxonomía biológica (450K+ especies en 
TreeOfLife-10M). Para insectos, supera a CLIP estándar en 25+ puntos porcentuales 
(34.8% vs 9.1% en Insects task). Lo incorporamos como primer encoder del ensemble.

## Estructura del proyecto

```
vr_rag_demo/
├── demo_vr_rag.py      ← código principal, léelo completo
├── requirements.txt    ← dependencias
├── LEEME.md            ← este archivo
└── images_demo/        ← se crea automáticamente con las imágenes descargadas
```

## Si hay errores frecuentes

- **OOM (out of memory)**: el demo usa modelos pequeños, debería funcionar con 4GB RAM
- **Descarga lenta**: las imágenes de Wikipedia se descargan una vez y se cachean en images_demo/
- **Error open_clip**: hacer `pip install open-clip-torch` (no `openclip`)
- **Error transformers**: hacer `pip install transformers>=4.40.0`

## Contexto de la tesis

- Tesis: identificación automatizada de artrópodos subrepresentados en Amazonía peruana
- Criterios de subrepresentación: espacial (< registros GBIF), visual (< imágenes iNat), textual
- Datos: GBIF, iNaturalist, WWF Ecoregions 2017, Bioestación Manu
- El sistema opera en modo open-vocabulary: identifica especies no vistas en entrenamiento
