from __future__ import annotations

import pandas as pd

from nbapredicts.collect import Store
from nbapredicts.text_log import _append


def test_append_keeps_only_new_notes(tmp_path):
    store = Store(tmp_path)
    first = pd.DataFrame(
        {
            "season": [2027, 2027],
            "name": ["A", "B"],
            "note_date": ["d1", "d1"],
            "short_comment": ["out", "probable"],
        }
    )
    assert _append(store, "notes", first, ["name", "note_date", "short_comment"]) == 2
    again = pd.DataFrame(
        {
            "season": [2027, 2027],
            "name": ["A", "B"],
            "note_date": ["d1", "d2"],
            "short_comment": ["out", "available"],
        }
    )
    assert _append(store, "notes", again, ["name", "note_date", "short_comment"]) == 1
    assert len(store.load("notes")) == 3
