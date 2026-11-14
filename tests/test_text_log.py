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


class FakeESPN:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get(self, path, **params):
        if path == "injuries":
            athlete = {"id": "7", "displayName": "A Player"}
            note = {"athlete": athlete, "date": "d1", "shortComment": "Out tonight."}
            return {"injuries": [{"id": "1", "injuries": [note]}]}
        return {"articles": [{"id": 9, "headline": "News", "description": "Text"}]}


def test_text_log_command_keeps_espn_text_out_of_the_public_data(tmp_path, monkeypatch):
    from nbapredicts import cli

    monkeypatch.setattr(cli, "ESPNClient", FakeESPN)
    cli.main(["--root", str(tmp_path), "text-log"])
    local = Store(tmp_path / "data_local")
    assert len(local.load("text_injuries")) == 1
    assert len(local.load("text_news")) == 1
    assert not (tmp_path / "data" / "text_news").exists()
    assert not (tmp_path / "data" / "text_injuries").exists()
