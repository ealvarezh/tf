import os
import torch
import pandas as pd
from tqdm import tqdm

from image.model import get_encoder, get_hashcoder
from image.dataset import get_dataloaders
from image.args import args


def generate_embeddings(
    encoder,
    hashcoder_img,
    loader,
    device="cpu"
):
    encoder.eval()
    hashcoder_img.eval()

    rows = []

    with torch.no_grad():

        for images, captions, labels in tqdm(loader):

            images = images.to(device)

            feats = encoder.encode_image(images)          # [B,768]
            hashes = hashcoder_img(feats)                # [B,16]
            binary_hashes = (hashes > 0).int()

            feats = feats.cpu()
            binary_hashes = binary_hashes.cpu()

            for i in range(images.size(0)):

                rows.append(
                    {
                        "label": int(labels[i]),
                        "caption": captions[i],
                        "embedding_768": feats[i].tolist(),
                        "hash_16": binary_hashes[i].tolist()
                    }
                )

    return pd.DataFrame(rows)


def main(args):

    # ------------------------
    # Models
    # ------------------------

    encoder, _, preprocess_eval, tokenizer, dim = get_encoder(args)

    hashcoder_img = get_hashcoder(dim, args)
    hashcoder_txt = get_hashcoder(dim, args)

    # ------------------------
    # Checkpoint
    # ------------------------

    checkpoint = torch.load(
        args.ckpt,
        map_location=args.device
    )

    encoder.load_state_dict(checkpoint["encoder"])
    hashcoder_img.load_state_dict(checkpoint["hashcoder_img"])
    hashcoder_txt.load_state_dict(checkpoint["hashcoder_txt"])

    print("Checkpoint loaded.")

    # ------------------------
    # Dataset
    # ------------------------

    _, database_loader, _, _ = get_dataloaders(
        args,
        train_transform=None,
        eval_transform=preprocess_eval,
    )

    # ------------------------
    # Generate embeddings
    # ------------------------

    df = generate_embeddings(
        encoder,
        hashcoder_img,
        database_loader,
        args.device
    )

    # ------------------------
    # Save parquet
    # ------------------------

    output_file = "../data/functional/tiny_embeddings.parquet"

    df.to_parquet(
        output_file,
        index=False
    )

    print(f"Saved {len(df)} embeddings to {output_file}")


if __name__ == "__main__":
    main(args)

    