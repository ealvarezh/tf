"""
select_test_species.py
-----------------------
Selecciona 20 especies variadas (no solo Lepidoptera) desde
species_db_classified.json para usarlas como set de prueba leave-one-out
del pipeline VR-RAG, y calcular MRR@3, MRR@10, MRR@30.

Estrategia de seleccion:
  1. Solo especies con >=2 imagenes locales (1 se deja afuera como query,
     el resto sigue sirviendo de ancla para esa especie).
  2. Se reparte el cupo de 20 entre tantos ordenes distintos como sea posible
     (maximo MAX_PER_ORDER por orden), priorizando dist_tier
     "critically_underrepresented" > "underrepresented" > "sufficient".
  3. Seleccion determinista (semilla fija) para que sea reproducible.

Uso (en la maquina con los datos, junto a kb_insecta/):
    python select_test_species.py
    python select_test_species.py --input "C:\\ruta\\species_db_classified.json" --n 20

Genera kb_insecta/test_candidates_loo.json con, por especie:
    species, genus, family, order, dist_tier, dataset_records,
    query_image (la imagen que se deja afuera), anchor_images (el resto)
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

MAX_PER_ORDER = 3
TIER_PRIORITY = {
    "critically_underrepresented": 0,
    "underrepresented": 1,
    "sufficient": 2,
}
SEED = 42


def select(data: list, n: int) -> list:
    eligible = [d for d in data if len(d.get("local_images", [])) >= 2]

    by_order = defaultdict(list)
    for d in eligible:
        by_order[d["order"]].append(d)

    rng = random.Random(SEED)
    for order in by_order:
        by_order[order].sort(
            key=lambda d: (TIER_PRIORITY.get(d.get("dist_tier"), 9), d["species"])
        )

    orders = sorted(by_order.keys(), key=lambda o: len(by_order[o]))  # ordenes raros primero
    selected = []
    used_species = set()

    # Ronda 1: un representante por orden, empezando por los ordenes con menos especies
    # (asi se garantiza variedad antes de que Lepidoptera/Coleoptera llenen el cupo)
    round_idx = 0
    while len(selected) < n and round_idx < MAX_PER_ORDER:
        progressed = False
        for order in orders:
            if len(selected) >= n:
                break
            pool = by_order[order]
            if round_idx >= min(len(pool), MAX_PER_ORDER):
                continue
            cand = pool[round_idx]
            if cand["species"] in used_species:
                continue
            selected.append(cand)
            used_species.add(cand["species"])
            progressed = True
        round_idx += 1
        if not progressed:
            break

    return selected[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="kb_insecta/species_db_classified.json")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", default="kb_insecta/test_candidates_loo.json")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: no existe {in_path}")
        return

    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)

    chosen = select(data, args.n)

    print(f"\n── Seleccion de {len(chosen)} especies de prueba (leave-one-out) ──────\n")
    by_order_count = defaultdict(int)
    out_records = []
    for d in chosen:
        by_order_count[d["order"]] += 1
        query_img = d["local_images"][-1]
        anchors = d["local_images"][:-1]
        out_records.append({
            "species":        d["species"],
            "genus":          d.get("genus", ""),
            "family":         d.get("family", ""),
            "order":          d["order"],
            "dist_tier":      d.get("dist_tier", ""),
            "dataset_records": d.get("dataset_records", 0),
            "query_image":    query_img,
            "anchor_images":  anchors,
        })
        print(f"  {d['species']:<35} {d['order']:<13} {d.get('dist_tier',''):<28} "
              f"registros={d.get('dataset_records', 0):<4} imgs={len(d['local_images'])}")

    print(f"\n  Distribucion por orden: {dict(by_order_count)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    print(f"\n  Guardado en: {out_path}")


if __name__ == "__main__":
    main()
