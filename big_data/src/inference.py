

import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image

from image.model import get_encoder, get_hashcoder
from image.args import args


# ============================================================
# HAMMING DISTANCE
# ============================================================

def hamming_distance(query_hash, db_hashes):
    """
    query_hash: (16,)
    db_hashes: (N,16)

    Returns:
        distances: (N,)
    """
    return np.sum(query_hash != db_hashes, axis=1)


# ============================================================
# IMAGE -> HASH
# ============================================================

def image_to_hash(
    image_path,
    encoder,
    hashcoder,
    preprocess,
    device
):
    """
    Converts an image into a binary hash.
    """

    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    encoder.eval()
    hashcoder.eval()

    with torch.no_grad():

        feat = encoder.encode_image(image)

        logits = hashcoder(feat)

        binary_hash = (logits > 0).int()

    return binary_hash.squeeze(0).cpu().numpy()


# ============================================================
# LOAD MODEL
# ============================================================

def load_models():
    args.device = "cpu"
    args.hashcoder = "small"
    args.bitdim = 16

    encoder, _, preprocess_eval, tokenizer, dim = get_encoder(args)

    hashcoder = get_hashcoder(dim, args)

    checkpoint = torch.load(
        "../model/bioclip2_16dim_1epochs.pth",
        map_location=args.device
    )

    encoder.load_state_dict(checkpoint["encoder"])
    hashcoder.load_state_dict(checkpoint["hashcoder_img"])

    encoder.eval()
    hashcoder.eval()

    return encoder, hashcoder, preprocess_eval


# ============================================================
# MAIN
# ============================================================

def main(image_path):

    print("Loading model...")

    encoder, hashcoder, preprocess = load_models()

    print("Loading embeddings parquet...")

    df = pd.read_parquet(
        "../data/functional/tiny_embeddings.parquet"
    )

    db_hashes = np.vstack(
        df["hash_16"].apply(
            lambda x: np.array(x)
        ).values
    )

    print(f"{len(df)} embeddings loaded")

    query_hash = image_to_hash(
        image_path=image_path,
        encoder=encoder,
        hashcoder=hashcoder,
        preprocess=preprocess,
        device=args.device
    )

    distances = hamming_distance(
        query_hash,
        db_hashes
    )

    best_idx = np.argmin(distances)

    best_row = df.iloc[best_idx]

    print("\n==============================")
    print("BEST MATCH")
    print("==============================")

    print(f"Distance: {distances[best_idx]}")

    if "caption" in best_row:
        print(f"Caption: {best_row['caption']}")

    if "label" in best_row:
        print(f"Label: {best_row['label']}")

    if "image_path" in best_row:
        print(f"Image: {best_row['image_path']}")

    print("==============================\n")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    image_path = "../data/landing/Felis_catus_0.jpg"

    main(image_path)