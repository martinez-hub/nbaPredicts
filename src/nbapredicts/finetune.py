"""Fine-tune TimesFM 2.5's point head on NBA series.

The 20 transformer layers stay frozen; only ``output_projection_point`` (about
5M parameters, which emits the mean and nine deciles for the next 128 steps)
is trained, on embeddings the frozen stack computes once. That keeps a daily
fine-tune within minutes on a CPU runner.

Training mirrors inference exactly: each window is left-padded to the model's
context, normalized by its own mean and standard deviation (TimesFM's
``normalize_inputs``), then by the per-patch running statistics ``decode``
uses. Every patch position is a training example whose targets are the next
``K`` values of the series after that patch, as in TimesFM's own decoder-only
pretraining. Each window is also used negated, matching the flip-invariance
averaging applied at inference.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

log = logging.getLogger(__name__)

K = 4  # steps ahead supervised per position; forecasts here use 1-3
QUANTILES = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
_TOL = 1e-6


@dataclass
class FinetuneConfig:
    # From the 2025-26 backtest: lr 1e-4 for 12 epochs on 8,000 windows beat
    # 3e-5/6/4,000 (undertrained) and 3e-4 (lower training loss, worse
    # forecasts of player points).
    windows: int = 8000
    epochs: int = 12
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 512
    val_frac: float = 0.15
    seed: int = 0


def _windows(
    series: Sequence[np.ndarray],
    ends: Sequence[np.ndarray],
    n: int,
    context: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample (series index, end index) pairs; ``ends[i]`` lists allowed ends."""
    pool = [(i, int(e)) for i, es in enumerate(ends) for e in es]
    if not pool:
        return []
    pick = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[j] for j in pick]


def _build(
    series: Sequence[np.ndarray],
    picks: list[tuple[int, int]],
    context: int,
    patch: int,
    limit: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Padded contexts, masks, per-position targets and target masks.

    ``limit[i]`` is the first index of series ``i`` a target may not use (the
    training cutoff), so no window ever learns from held-out games.
    """
    num_patches = context // patch
    x = np.zeros((len(picks), context), dtype=np.float32)
    m = np.ones((len(picks), context), dtype=bool)
    y = np.zeros((len(picks), num_patches, K), dtype=np.float32)
    ym = np.zeros((len(picks), num_patches, K), dtype=bool)
    for row, (i, end) in enumerate(picks):
        s = series[i]
        start = end - context  # series index of padded position 0
        lo = max(start, 0)
        x[row, lo - start :] = s[lo:end]
        m[row, lo - start :] = False
        for p in range(num_patches):
            if start + patch * p + patch - 1 < 0:
                continue  # patch is all padding
            first = start + patch * (p + 1)
            for k in range(K):
                t = first + k
                if 0 <= t < min(len(s), limit[i]):
                    y[row, p, k] = s[t]
                    ym[row, p, k] = True
    return (
        torch.from_numpy(x),
        torch.from_numpy(m),
        torch.from_numpy(y),
        torch.from_numpy(ym),
    )


@torch.no_grad()
def _embed(module, x: torch.Tensor, m: torch.Tensor, batch: int = 64):
    """Frozen-stack embeddings plus the normalization needed for the targets."""
    from timesfm.torch import util

    p = module.p
    out_emb, out_mu, out_sig, out_gmu, out_gsig = [], [], [], [], []
    for b in range(0, len(x), batch):
        xb, mb = x[b : b + batch], m[b : b + batch]
        # normalize_inputs: over the padded context, padding included, as
        # compiled_decode does.
        gmu = xb.mean(dim=-1, keepdim=True)
        gsig = xb.std(dim=-1, keepdim=True)
        xb = util.revin(xb, gmu, gsig, reverse=False)
        bsz = xb.shape[0]
        px, pm = xb.reshape(bsz, -1, p), mb.reshape(bsz, -1, p)
        n = mu = sig = torch.zeros(bsz)
        mus, sigs = [], []
        for i in range(px.shape[1]):
            (n, mu, sig), _ = util.update_running_stats(n, mu, sig, px[:, i], pm[:, i])
            mus.append(mu)
            sigs.append(sig)
        cmu, csig = torch.stack(mus, 1), torch.stack(sigs, 1)
        normed = torch.where(pm, 0.0, util.revin(px, cmu, csig, reverse=False))
        (_, emb, _, _), _ = module(normed, pm)
        out_emb.append(emb)
        out_mu.append(cmu)
        out_sig.append(csig)
        out_gmu.append(gmu)
        out_gsig.append(gsig)
    return (
        torch.cat(out_emb),
        torch.cat(out_mu),
        torch.cat(out_sig),
        torch.cat(out_gmu),
        torch.cat(out_gsig),
    )


def _targets(y, ym, cmu, csig, gmu, gsig):
    """Targets in the doubly normalized space the point head predicts in."""
    g_sig = torch.where(gsig < _TOL, 1.0, gsig)
    yt = (y - gmu[..., None]) / g_sig[..., None]
    c_sig = torch.where(csig < _TOL, 1.0, csig)
    yt = (yt - cmu[..., None]) / c_sig[..., None]
    return torch.where(ym, yt, 0.0)


def _loss(head, emb, yt, ym, o: int, q: int) -> torch.Tensor:
    out = head(emb).reshape(*emb.shape[:2], o, q)[:, :, :K, :]
    mean, quant = out[..., 0], out[..., 1:]
    w = ym.float()
    mse = ((mean - yt) ** 2 * w).sum() / w.sum().clamp(min=1)
    err = yt[..., None] - quant
    pin = torch.maximum(QUANTILES * err, (QUANTILES - 1) * err)
    pin = (pin * w[..., None]).sum() / (w.sum().clamp(min=1) * len(QUANTILES))
    return pin + 0.1 * mse


def finetune_point_head(
    module,
    series: Sequence[np.ndarray],
    ends: Sequence[np.ndarray],
    limit: Sequence[int],
    config: FinetuneConfig | None = None,
) -> dict[str, float]:
    """Train ``module.output_projection_point`` in place; keep the best epoch.

    ``ends[i]`` are the window end indices to sample for series ``i`` and
    ``limit[i]`` the first index no target may reach. A slice of the windows
    is held out for early stopping; the head is restored to the epoch with the
    lowest held-out loss (possibly the untouched original, epoch 0).
    """
    config = config or FinetuneConfig()
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    context = 512
    picks = _windows(series, ends, config.windows, context, rng)
    x, m, y, ym = _build(series, picks, context, module.p, limit)
    log.info("fine-tune: %d windows, %d target values", len(picks), int(ym.sum()))

    # Original and negated windows (flip invariance at inference).
    emb, cmu, csig, gmu, gsig = _embed(module, torch.cat([x, -x]), torch.cat([m, m]))
    y2, ym2 = torch.cat([y, -y]), torch.cat([ym, ym])
    yt = _targets(y2, ym2, cmu, csig, gmu, gsig)

    n = len(picks)
    order = torch.from_numpy(rng.permutation(n))
    n_val = max(1, int(n * config.val_frac))
    val = torch.cat([order[:n_val], order[:n_val] + n])
    trn = torch.cat([order[n_val:], order[n_val:] + n])

    head = module.output_projection_point
    o, q = module.o, module.q
    for prm in module.parameters():
        prm.requires_grad_(False)
    for prm in head.parameters():
        prm.requires_grad_(True)

    def val_loss() -> float:
        with torch.no_grad():
            return float(_loss(head, emb[val], yt[val], ym2[val], o, q))

    best = val_loss()
    start = best
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    opt = torch.optim.AdamW(
        head.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    for epoch in range(1, config.epochs + 1):
        perm = trn[torch.randperm(len(trn))]
        for b in range(0, len(perm), config.batch_size):
            idx = perm[b : b + config.batch_size]
            opt.zero_grad()
            loss = _loss(head, emb[idx], yt[idx], ym2[idx], o, q)
            loss.backward()
            opt.step()
        current = val_loss()
        log.info("fine-tune epoch %d: held-out loss %.4f", epoch, current)
        if current < best:
            best, best_epoch = current, epoch
            best_state = copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    for prm in head.parameters():
        prm.requires_grad_(False)
    module.eval()
    return {
        "windows": len(picks),
        "val_loss_before": start,
        "val_loss_after": best,
        "best_epoch": best_epoch,
    }


# --------------------------------------------------------------------------
# Daily use: fine-tune, check against the latest games, keep only if better
# --------------------------------------------------------------------------


def training_series(
    tf, pf, cutoff, first_start, player_stride: int = 16
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Team-points and player-stat series with sampling ranges and cutoffs.

    Window ends are drawn from games since ``first_start``; no target may use
    a game at or after ``cutoff``. Player series are thinned by
    ``player_stride`` so teams are not drowned out (there are ~100 player
    series per team series).
    """
    from .forecast import MIN_GAMES, PLAYER_STATS

    series, ends, limits = [], [], []

    def add(values: np.ndarray, starts, stride: int) -> None:
        lim = int((starts < cutoff).sum())
        first = max(int((starts < first_start).sum()), MIN_GAMES)
        if lim - first < 2:
            return
        series.append(values)
        limits.append(lim)
        ends.append(np.arange(first, lim)[::stride])

    finals = tf[tf["final"]].sort_values(["start", "game_id"])
    for _, rows in finals.groupby("team_id"):
        add(rows["pts"].to_numpy(float), rows["start"], 1)
    for stat in PLAYER_STATS:
        for _, rows in pf.groupby("athlete_id"):
            rows = rows.dropna(subset=[stat]).sort_values("start")
            add(rows[stat].to_numpy(float), rows["start"], player_stride)
    return series, ends, limits


def holdout_jobs(tf, pf, start, max_context: int, player_sample: int, seed: int = 0):
    """One-step jobs for every game since ``start`` (players sampled)."""
    from .forecast import MIN_GAMES, PLAYER_STATS, _job

    jobs, groups, actual = [], [], []
    finals = tf[tf["final"]].sort_values(["start", "game_id"])
    for _, rows in finals.groupby("team_id"):
        rows = rows.reset_index(drop=True)
        for i in np.flatnonzero(rows["start"].to_numpy() >= start):
            if i < MIN_GAMES:
                continue
            target = rows.iloc[[i]]
            jobs.append(_job(rows.iloc[max(0, i - max_context) : i], target, "pts"))
            groups.append("team_pts")
            actual.append(float(target["pts"].iloc[0]))
    pf = pf.sort_values(["start", "game_id"]).reset_index(drop=True)
    pf["n"] = pf.groupby("athlete_id").cumcount()
    cand = pf[(pf["start"] >= start) & (pf["n"] >= MIN_GAMES)]
    picks = cand.sample(min(player_sample, len(cand)), random_state=seed)
    history = dict(tuple(pf.groupby("athlete_id", sort=False)))
    for p in picks.itertuples():
        rows = history[p.athlete_id]
        hist, target = rows[rows["n"] < p.n], rows[rows["n"] == p.n]
        for stat in PLAYER_STATS:
            h = hist.dropna(subset=[stat]).tail(max_context)
            if len(h) < MIN_GAMES or np.isnan(getattr(p, stat)):
                continue
            jobs.append(_job(h, target, stat))
            groups.append(stat)
            actual.append(float(getattr(p, stat)))
    return jobs, groups, np.array(actual)


def _score(tfm, jobs, groups, actual) -> dict[str, float]:
    pred = np.array([float(point[0]) for point, _ in tfm.run(jobs)])
    err = np.abs(pred - actual)
    g = np.array(groups)
    return {name: float(err[g == name].mean()) for name in sorted(set(groups))}


def daily_finetune(
    tfm,
    tf,
    pf,
    holdout_games: int = 150,
    history_days: int = 3 * 365,
    player_sample: int = 600,
    config: FinetuneConfig | None = None,
) -> dict:
    """Fine-tune ``tfm``'s point head; keep it only if it wins on recent games.

    The most recent ``holdout_games`` finals are held out: the head trains on
    everything before them, and is kept only if its one-step forecasts of
    those games beat the untouched model's (mean of per-stat MAE ratios
    below 1). Otherwise the original head is restored.
    """
    import pandas as pd

    module = tfm.model.model
    finals = tf[tf["final"]].drop_duplicates("game_id").sort_values("start")
    if len(finals) < holdout_games + 100:
        return {"used": False, "reason": "not enough games"}
    cutoff = finals["start"].iloc[-holdout_games]
    jobs, groups, actual = holdout_jobs(tf, pf, cutoff, tfm.max_context, player_sample)
    base = _score(tfm, jobs, groups, actual)

    original = copy.deepcopy(module.output_projection_point.state_dict())
    series, ends, limits = training_series(
        tf, pf, cutoff, cutoff - pd.Timedelta(days=history_days)
    )
    info = finetune_point_head(module, series, ends, limits, config)
    tuned = _score(tfm, jobs, groups, actual)

    ratio = float(np.mean([tuned[k] / base[k] for k in base]))
    used = ratio < 1.0
    if not used:
        module.output_projection_point.load_state_dict(original)
    log.info("daily fine-tune: ratio %.4f, %s", ratio, "kept" if used else "reverted")
    return {
        "used": used,
        "mae_ratio": ratio,
        "holdout_games": holdout_games,
        "holdout_from": cutoff.strftime("%Y-%m-%d"),
        "mae_base": base,
        "mae_tuned": tuned,
        **info,
    }
