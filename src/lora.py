import time

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, get_linear_schedule_with_warmup

from .esm_embed import trim_ends
from .models import EmbeddingClassifier


class LoraESM(nn.Module):
    """ESM-2 with LoRA adapters on query/value, mean pooling and an MLP head."""

    def __init__(self, model_name, n_classes, r=8, alpha=16, dropout=0.05):
        super().__init__()
        base = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                         target_modules=["query", "value"])
        self.encoder = get_peft_model(base, cfg)
        self.head = EmbeddingClassifier(base.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.clone()
        mask[:, 0] = 0                                             # <cls>
        mask[torch.arange(len(mask)), attention_mask.sum(dim=1) - 1] = 0   # <eos>
        m = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(dim=1) / m.sum(dim=1)
        return self.head(pooled)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n])


def make_batch(tokenizer, seqs, max_len, device):
    seqs = [trim_ends(s, max_len) for s in seqs]
    enc = tokenizer(seqs, return_tensors="pt", padding=True)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def predict_lora(model, tokenizer, seqs, device, max_len=510, batch_size=32, amp=True):
    model.eval()
    order = np.argsort([len(s) for s in seqs])
    preds = np.zeros(len(seqs), dtype=np.int64)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        ids, att = make_batch(tokenizer, [seqs[j] for j in idx], max_len, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(ids, att)
        preds[idx] = logits.argmax(dim=1).cpu().numpy()
    return preds


def train_lora(model, tokenizer, train_seqs, y_train, val_seqs, y_val, device, seed,
               epochs=4, batch_size=16, lr_lora=3e-4, lr_head=1e-3,
               max_len=510, amp=True):
    model.to(device)
    lora_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": lora_params, "lr": lr_lora},
                             {"params": head_params, "lr": lr_head}], weight_decay=0.01)
    steps_per_epoch = int(np.ceil(len(train_seqs) / batch_size))
    total_steps = epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)
    loss_fn = nn.CrossEntropyLoss()
    y_t = torch.from_numpy(y_train)
    g = torch.Generator().manual_seed(seed)

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(train_seqs), generator=g)
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size].tolist()
            ids, att = make_batch(tokenizer, [train_seqs[j] for j in idx], max_len, device)
            yb = y_t[idx].to(device)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(ids, att)
            loss = loss_fn(logits.float(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)

        val_pred = predict_lora(model, tokenizer, val_seqs, device, max_len, amp=amp)
        f1 = f1_score(y_val, val_pred, average="macro")
        acc = accuracy_score(y_val, val_pred)
        history.append({"epoch": epoch + 1, "train_loss": total / len(train_seqs),
                        "val_acc": acc, "val_macro_f1": f1})
        print(f"epoch {epoch + 1}  loss {total / len(train_seqs):.4f}  val acc {acc:.3f}  "
              f"val macro-F1 {f1:.3f}  ({(time.time() - t0) / 60:.1f} min)")
        if f1 > best_f1:
            best_f1, best_state = f1, trainable_state(model)

    load_trainable_state(model, best_state)
    return model, history
