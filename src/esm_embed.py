import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def trim_ends(seq, max_len):
    """Keep the first and last max_len/2 residues of a long sequence."""
    if len(seq) <= max_len:
        return seq
    half = max_len // 2
    return seq[:half] + seq[-half:]


@torch.no_grad()
def embed_sequences(seqs, model_name, device, max_len=1000, batch_size=16):
    """Mean-pooled ESM-2 embedding of each sequence (special tokens excluded)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    seqs = [trim_ends(s, max_len) for s in seqs]

    order = np.argsort([len(s) for s in seqs])        # similar lengths together -> less padding
    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        batch = tokenizer([seqs[j] for j in idx], return_tensors="pt", padding=True).to(device)
        hidden = model(**batch).last_hidden_state                 # (batch, tokens, dim)

        mask = batch["attention_mask"].clone()
        mask[:, 0] = 0                                            # <cls> token
        last = batch["attention_mask"].sum(dim=1) - 1
        mask[torch.arange(len(idx)), last] = 0                    # <eos> token
        m = mask.unsqueeze(-1).to(hidden.dtype)
        out[idx] = ((hidden * m).sum(dim=1) / m.sum(dim=1)).cpu().numpy()
    return out
