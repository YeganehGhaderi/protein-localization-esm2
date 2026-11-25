import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, X, device, batch_size=256):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        preds.append(model(xb).argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def train_cnn(model, X_tr, y_tr, X_val, y_val, device, seed,
              epochs=30, patience=5, lr=1e-3, batch_size=64):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    X_t, y_t = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    g = torch.Generator().manual_seed(seed)

    best_f1, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t), generator=g)
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx].to(device), y_t[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        val_pred = predict(model, X_val, device)
        f1 = f1_score(y_val, val_pred, average="macro")
        acc = accuracy_score(y_val, val_pred)
        history.append({"epoch": epoch + 1, "train_loss": total / len(X_t),
                        "val_acc": acc, "val_macro_f1": f1})
        print(f"epoch {epoch + 1:2d}  loss {total / len(X_t):.4f}  val acc {acc:.3f}  val macro-F1 {f1:.3f}")

        if f1 > best_f1:
            best_f1, best_state, bad_epochs = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    return model, history