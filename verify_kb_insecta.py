"""
Verifica la integridad de kb_insecta/species_db.json y kb_insecta/knowledge_base.json
en la maquina destino: estructura, campos, imagenes locales presentes en disco.

Correr desde la carpeta que contiene kb_insecta/ (ej. E:\\TF1_bichos\\tf).

Uso:
    python verify_kb_insecta.py
"""
import json
import os
from pathlib import Path
from collections import Counter

KB_DIR = Path("kb_insecta")
KB_JSON = KB_DIR / "knowledge_base.json"
SPECIES_DB_JSON = KB_DIR / "species_db.json"
IMG_DIR = KB_DIR / "images"

REQUIRED_SPECIES_FIELDS = [
    "id", "name", "family", "order", "description",
    "image_url", "image_anchors", "subrepresented", "dataset_records",
]


def fail(msg):
    print(f"  [ERROR] {msg}")


def warn(msg):
    print(f"  [WARN]  {msg}")


def main():
    print("=== Verificacion de knowledge base de insectos ===\n")

    if not KB_JSON.exists():
        fail(f"No existe {KB_JSON}")
        return
    if not SPECIES_DB_JSON.exists():
        fail(f"No existe {SPECIES_DB_JSON}")
        return

    with open(KB_JSON, encoding="utf-8") as f:
        kb_index = json.load(f)
    with open(SPECIES_DB_JSON, encoding="utf-8") as f:
        species_db = json.load(f)

    print(f"  knowledge_base.json : {len(kb_index)} especies en el indice")
    print(f"  species_db.json     : {len(species_db)} especies en SPECIES_DB\n")

    # Especies del indice que tienen imagenes locales (deberian estar en species_db)
    expected_in_db = {
        k: v for k, v in kb_index.items() if v.get("local_images")
    }
    print(f"  Especies con local_images en el indice: {len(expected_in_db)}")

    if len(expected_in_db) != len(species_db):
        warn(
            f"Discrepancia de cantidad: indice con imagenes={len(expected_in_db)} "
            f"vs species_db={len(species_db)}"
        )

    errors = 0
    missing_files = 0
    missing_fields = 0
    duplicate_ids = Counter()

    db_ids = set()
    for i, entry in enumerate(species_db):
        # 1. Campos requeridos presentes
        for field in REQUIRED_SPECIES_FIELDS:
            if field not in entry:
                fail(f"[{i}] falta campo '{field}' (species={entry.get('name', '?')})")
                missing_fields += 1
                errors += 1

        # 2. IDs duplicados
        eid = entry.get("id")
        if eid:
            duplicate_ids[eid] += 1
            db_ids.add(eid)

        # 3. image_anchors no vacio y coincide con image_url
        anchors = entry.get("image_anchors", [])
        if not anchors:
            fail(f"[{i}] image_anchors vacio (species={entry.get('name')})")
            errors += 1
        elif entry.get("image_url") not in anchors:
            warn(f"[{i}] image_url no esta en image_anchors (species={entry.get('name')})")

        # 4. Verificar que los archivos de imagen existan en disco
        for img_path in anchors:
            p = Path(img_path)
            if not p.is_absolute():
                p = KB_DIR / img_path
            if not p.exists():
                fail(f"[{i}] imagen no encontrada en disco: {img_path} (species={entry.get('name')})")
                missing_files += 1
                errors += 1

    dupes = {k: c for k, c in duplicate_ids.items() if c > 1}
    if dupes:
        fail(f"IDs duplicados en species_db.json: {dupes}")
        errors += len(dupes)

    # 5. Verificar que cada especie del indice con imagenes este representada en species_db
    missing_in_db = set(expected_in_db.keys()) - {
        e["name"] for e in species_db
    }
    if missing_in_db:
        warn(f"{len(missing_in_db)} especies con imagenes en el indice pero ausentes de species_db.json")
        for name in list(missing_in_db)[:10]:
            print(f"      - {name}")
        if len(missing_in_db) > 10:
            print(f"      ... y {len(missing_in_db) - 10} mas")

    print("\n── Resumen ─────────────────────────────")
    print(f"  Campos faltantes        : {missing_fields}")
    print(f"  Archivos de imagen perdidos: {missing_files}")
    print(f"  IDs duplicados           : {len(dupes)}")
    print(f"  Especies faltantes en DB : {len(missing_in_db)}")
    print(f"  Total errores            : {errors}")

    if errors == 0 and not missing_in_db:
        print("\n  OK: species_db.json esta consistente y todas las imagenes existen.")
    else:
        print("\n  Se encontraron problemas. Revisa los detalles arriba.")


if __name__ == "__main__":
    main()
