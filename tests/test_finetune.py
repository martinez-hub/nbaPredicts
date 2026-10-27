from __future__ import annotations

import numpy as np
import torch

from nbapredicts.finetune import K, _build, _targets


def test_targets_never_reach_the_cutoff():
    s = np.arange(200, dtype=float)
    limit = 150
    # Windows ending at, before and after the cutoff.
    x, m, y, ym = _build([s], [(0, 120), (0, 149), (0, 180)], 64, 32, [limit])
    # Every supervised target value is a series value strictly before the cutoff.
    assert y[ym].max().item() < limit
    # The window ending at 180 still learns from its earlier patches.
    assert ym[2].any()


def test_targets_follow_each_patch_and_padding_is_masked():
    s = np.arange(100, dtype=float) + 1000
    x, m, y, ym = _build([s], [(0, 40)], 64, 32, [100])
    # Context = s[0:40] right-aligned in 64 slots: 24 padded positions.
    assert m[0, :24].all() and not m[0, 24:].any()
    assert torch.equal(x[0, 24:], torch.tensor(s[:40], dtype=torch.float32))
    # Patch 0 holds s[0:8] (its last real value is s[7]), so its targets are
    # s[8:8+K]; patch 1 ends at s[39], so its targets are s[40:40+K].
    np.testing.assert_array_equal(y[0, 0].numpy(), s[8 : 8 + K])
    np.testing.assert_array_equal(y[0, 1].numpy(), s[40 : 40 + K])
    assert ym[0].all()


def test_all_padding_patch_is_not_supervised():
    s = np.arange(20, dtype=float)
    x, m, y, ym = _build([s], [(0, 10)], 64, 32, [20])
    # 54 padded slots: patch 0 is all padding, patch 1 holds s[0:10].
    assert not ym[0, 0].any() and ym[0, 1].any()


def test_targets_are_normalized_like_decode():
    y = torch.tensor([[[110.0, 120.0, 100.0, 90.0]]])
    ym = torch.ones_like(y, dtype=torch.bool)
    cmu, csig = torch.tensor([[0.5]]), torch.tensor([[2.0]])
    gmu, gsig = torch.tensor([[100.0]]), torch.tensor([[10.0]])
    out = _targets(y, ym, cmu, csig, gmu, gsig)
    # ((110 - 100) / 10 - 0.5) / 2 = 0.25
    assert torch.allclose(out[0, 0], torch.tensor([0.25, 0.75, -0.25, -0.75]))
