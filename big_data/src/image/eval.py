import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple, Dict, Iterable


# -------------------------
# Helpers
# -------------------------
def _cat_cpu(lst: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate list of tensors on CPU (assumes tensors are already same shape on dim0)."""
    if not lst:
        return torch.empty(0)
    return torch.cat([t.cpu() for t in lst], dim=0)


def collect_first_text_per_label(loader: DataLoader) -> Tuple[List[str], torch.Tensor]:
    """
    Scan the loader and collect the first text found for each class label.
    Returns (texts, labels_tensor) where:
      - texts: list of strings sorted by ascending label id
      - labels_tensor: torch.LongTensor of the corresponding label ids
    Assumes each batch yields (images, texts, labels) or (texts, labels).
    Simple, deterministic, single-pass scan.
    """
    label2text: Dict[int, str] = {}
    for batch in loader:
        # Expect (images, texts, labels) or (texts, labels)
        if len(batch) == 3:
            _, texts_batch, labels_batch = batch
        else:
            texts_batch, labels_batch = batch

        for t, l in zip(texts_batch, labels_batch):
            lab = int(l.item()) if isinstance(l, torch.Tensor) else int(l)
            if lab not in label2text:
                # accept str or bytes
                if isinstance(t, bytes):
                    label2text[lab] = t.decode("utf-8", errors="ignore")
                else:
                    label2text[lab] = str(t)

        # stop early if we've seen a lot (no heuristics here - single pass)
    cat_ids = sorted(label2text.keys())
    texts = [label2text[c] for c in cat_ids]
    return texts, torch.tensor(cat_ids, dtype=torch.long)


# -------------------------
# Feature extraction
# -------------------------
def extract_image_features(encoder, hashcoder, loader: DataLoader, device: str = "cuda"):
    """
    Extract image features and hash logits for every sample in loader.
    Assumes loader yields (images, texts, labels) or (images, labels).
    Returns (feats_cpu, logits_cpu, labels_cpu).
    """
    encoder.eval()
    hashcoder.eval()

    feats, logits, labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting DB features"):
            if len(batch) == 3:
                images, _, lbls = batch
            else:
                images, lbls = batch

            images = images.to(device, non_blocking=True)
            f = encoder.encode_image(images)      # (B, D) on device
            l = hashcoder(f)                      # (B, H) on device

            feats.append(f.cpu())
            logits.append(l.cpu())
            labels.append(lbls.cpu())

    return _cat_cpu(feats), _cat_cpu(logits), _cat_cpu(labels)


def encode_texts(encoder, hashcoder, texts: List[str], tokenizer, labels: torch.Tensor = None,
                 device: str = "cuda", batch_size: int = 256):
    """
    Encode list of texts to features + logits. If labels is None, labels are 0..N-1.
    Returns (feats_cpu, logits_cpu, labels_tensor).
    """
    encoder.eval()
    hashcoder.eval()

    if labels is None:
        labels = torch.arange(len(texts), dtype=torch.long)

    feats, logits = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Query (txt) Features"):
            batch_texts = texts[i:i + batch_size]
            tokens = tokenizer(batch_texts).to(device, non_blocking=True)
            f = encoder.encode_text(tokens)
            l = hashcoder(f)
            feats.append(f.cpu())
            logits.append(l.cpu())

    return _cat_cpu(feats), _cat_cpu(logits), labels


# -------------------------
# Retrieval (simple, readable)
# -------------------------
def _chunked_cosine_topk(q_feats: torch.Tensor, db_feats: torch.Tensor, topk: int, device: str,
                         chunk_db: int = 16384) -> torch.LongTensor:
    """
    Compute topk indices by cosine similarity using DB-chunks to avoid OOM.
    Returns (num_q, topk) CPU LongTensor.
    """
    q = F.normalize(q_feats, dim=-1).to(device)
    db = F.normalize(db_feats, dim=-1).to(device)
    num_q = q.size(0)
    candidates_idx = []
    candidates_scores = []

    # process db in chunks
    for start in range(0, db.size(0), chunk_db):
        db_chunk = db[start:start + chunk_db]  # (C, D)
        scores = q @ db_chunk.t()              # (num_q, C)
        # store chunk indices relative to whole DB
        base_idx = torch.arange(start, start + db_chunk.size(0), device=device, dtype=torch.long)
        candidates_idx.append(base_idx.unsqueeze(0).expand(num_q, -1).cpu())
        candidates_scores.append(scores.cpu())

    # concatenate and take topk
    all_idx = torch.cat(candidates_idx, dim=1)    # (num_q, total_db)
    all_scores = torch.cat(candidates_scores, dim=1)
    k = min(topk, all_scores.size(1))
    _, pos = all_scores.topk(k, dim=1)
    topk_idx = all_idx.gather(1, pos)
    return topk_idx


def _chunked_asymhamming_topk(q_logits: torch.Tensor, db_logits: torch.Tensor, topk: int, device: str,
                              chunk_db: int = 16384) -> torch.LongTensor:
    """
    Asymmetric hamming: queries are soft (sigmoid), db is binary (db_logits>0).
    Use chunking similar to cosine.
    """
    q = torch.sigmoid(q_logits).to(device)
    db_bin = (db_logits > 0).float().to(device)
    num_q = q.size(0)

    candidates_idx = []
    candidates_scores = []
    for start in range(0, db_bin.size(0), chunk_db):
        db_chunk = db_bin[start:start + chunk_db]          # (C, H)
        # L1 distance -> smaller is better; we negate to treat as scores
        dist = torch.cdist(q, db_chunk, p=1)               # (num_q, C)
        scores = -dist
        base_idx = torch.arange(start, start + db_chunk.size(0), device=device, dtype=torch.long)
        candidates_idx.append(base_idx.unsqueeze(0).expand(num_q, -1).cpu())
        candidates_scores.append(scores.cpu())

    all_idx = torch.cat(candidates_idx, dim=1)
    all_scores = torch.cat(candidates_scores, dim=1)
    k = min(topk, all_scores.size(1))
    _, pos = all_scores.topk(k, dim=1)
    topk_idx = all_idx.gather(1, pos)
    return topk_idx


# -------------------------
# Metrics
# -------------------------
def mean_average_precision_at_k(retr_idx: torch.LongTensor, db_labels: torch.Tensor,
                                q_labels: torch.Tensor, k: int) -> float:
    """
    Compute mAP@k (returned as percentage 0..100).
    retr_idx: (num_q, topk) indices into db_labels
    """
    k = min(k, retr_idx.size(1))
    db_labels = db_labels.to(retr_idx.device)
    q_labels = q_labels.to(retr_idx.device)
    retrieved = db_labels[retr_idx[:, :k]]            # (num_q, k)
    relevant = (retrieved == q_labels.unsqueeze(1))   # bool
    cumsum = relevant.cumsum(1).float()
    denom = (torch.arange(1, k + 1, device=retr_idx.device).float()).unsqueeze(0)
    precision = cumsum / denom
    AP = (precision * relevant.float()).sum(dim=1) / relevant.sum(dim=1).clamp(min=1)
    return (AP.mean().item() * 100.0)


def evaluate_t2i_continuous(
    bioclip,
    db_loader: DataLoader,
    query_loader: DataLoader,
    tokenizer,
    topk: int = 1000,
    device: str = "cuda",
) -> Dict[str, float]:

    db_feats, _, db_labels = extract_image_features(bioclip, nn.Identity(), db_loader, device=device)
    query_texts, q_label_ids = collect_first_text_per_label(query_loader)
    q_feats, _, q_labels = encode_texts(bioclip, nn.Identity(), query_texts, tokenizer, labels=q_label_ids, device=device)

    topk_idx_cos = _chunked_cosine_topk(q_feats, db_feats, topk, device)

    metrics = {}
    metrics[f"mAP@{topk}_cosine"] = mean_average_precision_at_k(topk_idx_cos, db_labels, q_labels, topk)

    return metrics


def evaluate_t2i_binary(
    encoder,
    hashcoder_img,
    hashcoder_txt,
    db_loader: DataLoader,
    query_loader: DataLoader,
    tokenizer,
    topk: int = 1000,
    device: str = "cuda",
) -> Dict[str, float]:

    db_feats, db_logits, db_labels = extract_image_features(encoder, hashcoder_img, db_loader, device=device)
    query_texts, q_label_ids = collect_first_text_per_label(query_loader)
    q_feats, q_logits, q_labels = encode_texts(encoder, hashcoder_txt, query_texts, tokenizer, labels=q_label_ids, device=device)

    topk_idx_asym = _chunked_asymhamming_topk(q_logits, db_logits, topk, device)

    metrics = {}
    metrics[f"mAP@{topk}_asymhamming"] = mean_average_precision_at_k(topk_idx_asym, db_labels, q_labels, topk)

    return metrics


def run_evaluation(
    bioclip, encoder, hashcoder_img, hashcoder_txt, db_loader: DataLoader, 
    query_loader: DataLoader, tokenizer, topk: int = 1000, device: str = "cuda"
) -> Dict[str, float]:
    print("BioCLIP2 Retrieval:")
    continuous_metrics = evaluate_t2i_continuous(bioclip, db_loader, query_loader, tokenizer, topk, device)
    print(continuous_metrics)

    print("BioCLIP2-FT (Hashing) Retrieval:")
    binary_metrics = evaluate_t2i_binary(encoder, hashcoder_img, hashcoder_txt, db_loader, query_loader, tokenizer, topk, device)
    print(binary_metrics)

    metrics = {**continuous_metrics, **binary_metrics}
    return metrics