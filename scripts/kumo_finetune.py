"""Fine-tune Kumo-Forecast's head on NBA series, then forecast 2025-26 again.

Kumo-Forecast keeps its MOMENT encoder frozen and trains only the linear
head, so the encoder's output for each window is computed once and the
next-game output of the head (its first of 72 steps, the one used) is
retrained on those cached features: the windows ``kumo_compare.py
export-train`` drew from team and player series before 2025-26, the same
recipe as TimesFM's fine-tune (8,000 windows, 15% held out, best epoch
kept). Windows are z-scored by their own mean and spread with the padded
part at the mean (the ``own_scale`` variant of ``kumo_infer.py``).

Runs in Kumo-TS's environment:

    cd /Volumes/ExtremeSSD/Kumo-TS/Kumo-Forecast
    .venv/bin/python /Volumes/ExtremeSSD/nbaPredicts/scripts/kumo_finetune.py \\
        /Volumes/ExtremeSSD/nbaPredicts/cache.nosync/kumo

It adds ``team_finetuned`` and ``player_finetuned`` to ``kumo_preds.npz``,
which ``kumo_compare.py score`` then reports beside the zero-shot rows.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())  # Kumo-Forecast/: the SDK's absolute imports

from backbone import Masking  # noqa: E402
from sdk.forecasting import (  # noqa: E402
    CHECKPOINT_CROSS_CHANNEL,
    DEFAULT_BACKBONE_NAME,
    DEVICE,
    _load_cached_model,
    download_model_weights,
)

BATCH = 32
EPOCHS = 30
PATIENCE = 4
LR = 1e-4


def own_scale(ctx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    n = m.sum(axis=1)
    mean = np.where(m, ctx, 0).sum(axis=1) / n
    sd = np.sqrt(np.where(m, (ctx - mean[:, None]) ** 2, 0).sum(axis=1) / n) + 1e-6
    return np.where(m, (ctx - mean[:, None]) / sd[:, None], 0.0).astype(np.float32)


def encode(model, x: np.ndarray, mask: np.ndarray, device):
    """The head's input for each window, and RevIN's mean and spread.

    Mirrors ``BackboneModel.forecast`` up to the head.
    """
    feats, means, stds = [], [], []
    with torch.no_grad():
        for i in range(0, len(x), BATCH):
            xb = torch.from_numpy(x[i : i + BATCH, None, :]).float().to(device)
            mb = torch.from_numpy(mask[i : i + BATCH]).long().to(device)
            b = xb.shape[0]
            z = model.normalizer(x=xb, mask=mb, mode="norm")
            z = torch.nan_to_num(z, nan=0, posinf=0, neginf=0)
            z = model.tokenizer(x=z)
            enc = model.patch_embedding(z, mask=torch.ones_like(mb))
            n_patches = enc.shape[2]
            enc = enc.reshape((b, n_patches, model.config.d_model))
            att = Masking.convert_seq_to_patch_view(mb, model.patch_len)
            out = model.encoder(inputs_embeds=enc, attention_mask=att).last_hidden_state
            out = model._apply_cross_channel(
                out.reshape((-1, 1, n_patches, model.config.d_model))
            )
            feats.append(out.flatten(start_dim=-2)[:, 0].half().cpu())
            means.append(model.normalizer.mean[:, 0, 0].cpu())
            stds.append(model.normalizer.stdev[:, 0, 0].cpu())
    return torch.cat(feats), torch.cat(means).numpy(), torch.cat(stds).numpy()


def main() -> None:
    work = Path(sys.argv[1])
    device = torch.device(os.environ.get("KUMO_DEVICE", str(DEVICE)))
    _, ckpt = download_model_weights(ckpt=CHECKPOINT_CROSS_CHANNEL)
    model = _load_cached_model(
        model_name=DEFAULT_BACKBONE_NAME,
        ckpt=ckpt,
        seq_len=512,
        model_horizon=72,
        device=device,
    )
    linear = model.head.linear
    w0 = linear.weight[0].detach().float().cpu().clone()
    b0 = linear.bias[0].detach().float().cpu().clone()

    tr = np.load(work / "train_windows.npz")
    x, mask, y = own_scale(tr["ctx"], tr["mask"]), tr["mask"], tr["target"]
    feats, mean, std = encode(model, x, mask, device)
    # Window targets in the window's own z-units, then RevIN's.
    m = mask.astype(bool)
    n = m.sum(axis=1)
    raw_mean = np.where(m, tr["ctx"], 0).sum(axis=1) / n
    raw_sd = (
        np.sqrt(np.where(m, (tr["ctx"] - raw_mean[:, None]) ** 2, 0).sum(axis=1) / n)
        + 1e-6
    )
    z = ((y - raw_mean) / raw_sd - mean) / std
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    n_val = int(0.15 * len(y))
    val, fit_idx = order[:n_val], order[n_val:]

    w = torch.nn.Parameter(w0.clone())
    b = torch.nn.Parameter(b0.clone())
    opt = torch.optim.Adam([w, b], lr=LR)
    f = feats.float()
    zt = torch.from_numpy(z).float()

    def val_mae() -> float:
        with torch.no_grad():
            p = (f[val] @ w + b).numpy()
        pred = (p * std[val] + mean[val]) * raw_sd[val] + raw_mean[val]
        return float(np.abs(pred - y[val]).mean())

    history = [{"epoch": 0, "val_mae": round(val_mae(), 4)}]
    best, best_state, bad = history[0]["val_mae"], (w0.clone(), b0.clone()), 0
    for epoch in range(1, EPOCHS + 1):
        perm = fit_idx[rng.permutation(len(fit_idx))]
        for i in range(0, len(perm), 256):
            idx = perm[i : i + 256]
            loss = torch.nn.functional.huber_loss(f[idx] @ w + b, zt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        mae = val_mae()
        history.append({"epoch": epoch, "val_mae": round(mae, 4)})
        print(json.dumps(history[-1]), flush=True)
        if mae < best - 1e-4:
            best, best_state, bad = mae, (w.detach().clone(), b.detach().clone()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    wb, bb = best_state

    ev = np.load(work / "windows.npz")
    preds = dict(np.load(work / "kumo_preds.npz"))
    for part in ("team", "player"):
        ctx, msk = ev[f"{part}_ctx"], ev[f"{part}_mask"]
        f_ev, mu, sd = encode(model, own_scale(ctx, msk), msk, device)
        mm = msk.astype(bool)
        nn_ = mm.sum(axis=1)
        rm = np.where(mm, ctx, 0).sum(axis=1) / nn_
        rs = np.sqrt(np.where(mm, (ctx - rm[:, None]) ** 2, 0).sum(axis=1) / nn_) + 1e-6
        with torch.no_grad():
            p = (f_ev.float() @ wb + bb).numpy()
            p0 = (f_ev.float() @ w0 + b0).numpy()
        preds[f"{part}_finetuned"] = (p * sd + mu) * rs + rm
        # The cached-feature path with the released head must reproduce
        # kumo_infer.py's own_scale forecasts.
        check = float(
            np.abs((p0 * sd + mu) * rs + rm - preds[f"{part}_own_scale"]).max()
        )
        print(
            part,
            {
                "mean": float(preds[f"{part}_finetuned"].mean()),
                "zero_shot_max_diff": check,
            },
            flush=True,
        )
    np.savez_compressed(work / "kumo_preds.npz", **preds)
    meta = json.loads((work / "kumo_meta.json").read_text())
    meta["finetune"] = {
        "windows": int(len(y)),
        "held_out": n_val,
        "best_val_mae": round(best, 4),
        "zero_shot_val_mae": history[0]["val_mae"],
        "history": history,
    }
    (work / "kumo_meta.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
