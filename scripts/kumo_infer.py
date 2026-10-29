"""Kumo-Forecast one-step forecasts for the windows ``kumo_compare.py`` exported.

Runs in Kumo-TS's environment, not this project's:

    cd /Volumes/ExtremeSSD/Kumo-TS/Kumo-Forecast
    .venv/bin/python /Volumes/ExtremeSSD/nbaPredicts/scripts/kumo_infer.py \\
        /Volumes/ExtremeSSD/nbaPredicts/cache.nosync/kumo

It follows the SDK's direct path (``sdk.forecasting.perform_forecasting``):
standardize with the released standardizer, one forward pass of the
cross-channel checkpoint on a 512-point window, first step of the 72-step
forecast, inverse transform; but batched, with the padded part of a short
history masked out. A second variant z-scores each window by its own mean
and spread instead of the released standardizer, in case that one (fit on
Kumo's training data) suits points-per-game series badly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())  # Kumo-Forecast/: the SDK's absolute imports

from sdk.forecasting import (  # noqa: E402
    CHECKPOINT_CROSS_CHANNEL,
    DEFAULT_BACKBONE_NAME,
    DEVICE,
    _load_cached_model,
    _load_standardizer_artifact,
    download_model_weights,
)

MODEL_HORIZON = 72  # the released head's native horizon
SEQ_LEN = 512
BATCH = 64


def forecast(model, x: np.ndarray, mask: np.ndarray, device) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(x), BATCH):
            xb = torch.from_numpy(x[i : i + BATCH, None, :]).float().to(device)
            mb = torch.from_numpy(mask[i : i + BATCH]).long().to(device)
            out.append(model(x_enc=xb, input_mask=mb).forecast[:, 0, 0].cpu().numpy())
    return np.concatenate(out)


def released(model, std, ctx, mask, device) -> np.ndarray:
    mean, scale = float(std.mean.ravel()[0]), float(std.std.ravel()[0]) + std.eps
    z = np.where(mask == 1, (ctx - mean) / scale, 0.0).astype(np.float32)
    return forecast(model, z, mask, device) * scale + mean


def own_scale(model, ctx, mask, device) -> np.ndarray:
    m = mask.astype(bool)
    n = m.sum(axis=1)
    mean = np.where(m, ctx, 0).sum(axis=1) / n
    sd = np.sqrt(np.where(m, (ctx - mean[:, None]) ** 2, 0).sum(axis=1) / n) + 1e-6
    z = np.where(m, (ctx - mean[:, None]) / sd[:, None], 0.0).astype(np.float32)
    return forecast(model, z, mask, device) * sd + mean


def main() -> None:
    work = Path(sys.argv[1])
    device = torch.device(os.environ.get("KUMO_DEVICE", str(DEVICE)))
    std_path, ckpt = download_model_weights(ckpt=CHECKPOINT_CROSS_CHANNEL)
    std = _load_standardizer_artifact(std_path)
    meta = {
        "checkpoint": CHECKPOINT_CROSS_CHANNEL,
        "backbone": DEFAULT_BACKBONE_NAME,
        "device": str(device),
        "standardizer_mean": np.asarray(std.mean).ravel().round(4).tolist()[:8],
        "standardizer_std": np.asarray(std.std).ravel().round(4).tolist()[:8],
        "standardizer_size": int(np.asarray(std.mean).size),
    }
    print(json.dumps(meta), flush=True)
    model = _load_cached_model(
        model_name=DEFAULT_BACKBONE_NAME,
        ckpt=ckpt,
        seq_len=SEQ_LEN,
        model_horizon=MODEL_HORIZON,
        device=device,
    )
    w = np.load(work / "windows.npz")
    preds = {}
    for part in ("team", "player"):
        ctx, mask = w[f"{part}_ctx"], w[f"{part}_mask"]
        # A per-channel standardizer from multichannel training data has no
        # single value for a one-channel series; then only own_scale runs.
        if meta["standardizer_size"] == 1:
            preds[f"{part}_released"] = released(model, std, ctx, mask, device)
        preds[f"{part}_own_scale"] = own_scale(model, ctx, mask, device)
        print(
            part,
            {k: float(np.nanmean(v)) for k, v in preds.items() if k.startswith(part)},
            flush=True,
        )
    np.savez_compressed(work / "kumo_preds.npz", **preds)
    (work / "kumo_meta.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
