import torch
from tqdm import tqdm
from collections import defaultdict


def get_features(encoder, images, texts, tokenizer, device):
    """
    Extract image and text features using BioCLIP.
    """
    img_feats = encoder.encode_image(images)
    txt_tokens = tokenizer(texts).to(device)
    txt_feats = encoder.encode_text(txt_tokens)
    return img_feats, txt_feats


def average_by_class(features, class_ids):
    """
    Average features by class_id.
    
    Args:
        features: [N, D] tensor
        class_ids: [N] tensor/list with class labels
    Returns:
        avg_feats: [C, D] tensor (one per class in this batch)
        unique_classes: list of unique class ids
    """
    grouped = defaultdict(list)
    for feat, cid in zip(features, class_ids):
        grouped[cid.item()].append(feat)

    avg_feats, unique_classes = [], []
    for cid, feats in grouped.items():
        avg_feats.append(torch.stack(feats, dim=0).mean(dim=0))
        unique_classes.append(cid)

    avg_feats = torch.stack(avg_feats, dim=0)
    return avg_feats, unique_classes


def train_epoch(encoder, hashcoder_img, hashcoder_txt, train_loader, tokenizer, optimizer, criterion_inv, criterion_reg, scaler, scheduler, epoch, args):
    """
    Train for one epoch with cross-modal learning.
    If args.supervised is True, average features per class before loss computation.
    """
    freeze_encoder = epoch < args.frozenepochs

    encoder.eval() if freeze_encoder else encoder.train()
    hashcoder_img.train()
    hashcoder_txt.train()

    total_loss, total_inv, total_reg = 0.0, 0.0, 0.0
    n_batches = 0

    for idx, (images, texts, text_ids) in enumerate(tqdm(train_loader, desc="Training Iteration")):
        images = images.to(args.device)
        text_ids = text_ids.to(args.device)  # class IDs

        # --- Scheduler update ---
        it = len(train_loader) * epoch + idx
        for g in optimizer.param_groups:
            g["lr"] = scheduler["lr"][it]

        optimizer.zero_grad()

        with torch.autocast(device_type=args.device, dtype=torch.float16):
            # --- Encode features ---
            if freeze_encoder:
                with torch.no_grad():
                    img_feats, txt_feats = get_features(encoder, images, texts, tokenizer, args.device)
            else:
                img_feats, txt_feats = get_features(encoder, images, texts, tokenizer, args.device)

            # --- Project into hash space ---
            img_proj = hashcoder_img(img_feats)
            txt_proj = hashcoder_txt(txt_feats)

            img_codes = (img_proj > 0).to(img_proj.dtype)
            txt_codes = (txt_proj > 0).to(txt_proj.dtype)

            # --- Cross-modal loss (symmetrized) ---
            loss_inv = 0.5 * (criterion_inv(img_proj, txt_codes) + criterion_inv(txt_proj, img_codes))

            # --- Regularization loss ---
            loss_reg = 0.5 * (criterion_reg(img_proj) + criterion_reg(txt_proj))

            # --- Total loss ---
            loss = args.coeff_inv * loss_inv + args.coeff_reg * loss_reg

        # --- Backward + Step ---
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # --- Logging accumulators ---
        total_loss += loss.item()
        total_inv += loss_inv.item()
        total_reg += loss_reg.item()
        n_batches += 1

    # --- Averages ---
    avg_loss = total_loss / n_batches
    avg_inv = total_inv / n_batches
    avg_reg = total_reg / n_batches

    return avg_loss, avg_inv, avg_reg