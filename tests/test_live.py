from __future__ import annotations

import csv
import gzip

import pandas as pd

from nbapredicts import live

NOW = pd.Timestamp("2026-11-14T20:00:00Z")


def test_a_task_runs_once_per_cadence():
    state: dict = {}
    assert live.due(state, "rss", NOW, 2)
    live.mark(state, "rss", NOW)
    assert not live.due(state, "rss", NOW + pd.Timedelta(minutes=1), 2)
    # launchd's 2-minute interval drifts a little: a few seconds early counts.
    assert live.due(state, "rss", NOW + pd.Timedelta(seconds=110), 2)


RSS = """<?xml version="1.0"?><rss><channel>
<item><title><![CDATA[Star ruled out]]></title><link>https://x.test/a</link>
<guid>a1</guid><pubDate>Sat, 14 Nov 2026 19:50:09 GMT</pubDate></item>
<item><title>Guard upgraded</title><link>https://x.test/b</link>
<pubDate>Sat, 14 Nov 2026 19:58:00 GMT</pubDate></item>
</channel></rss>"""


def test_rss_items_read_title_link_and_time():
    items = live.rss_items(RSS)
    assert [i["title"] for i in items] == ["Star ruled out", "Guard upgraded"]
    assert items[0]["key"] == "a1"
    assert items[1]["key"] == "https://x.test/b"  # no guid: the link is the key
    assert items[0]["published"] == "Sat, 14 Nov 2026 19:50:09 GMT"


def test_bluesky_posts_keep_time_text_and_reposts():
    feed = {
        "feed": [
            {
                "post": {
                    "uri": "at://p1",
                    "indexedAt": "2026-11-14T19:50:10Z",
                    "record": {"text": "Scoop", "createdAt": "2026-11-14T19:50:09Z"},
                }
            },
            {
                "post": {
                    "uri": "at://p2",
                    "indexedAt": "2026-11-14T19:00:00Z",
                    "record": {"text": "Old", "createdAt": "2026-11-14T18:59:59Z"},
                },
                "reason": {"$type": "app.bsky.feed.defs#reasonRepost"},
            },
        ]
    }
    posts = live.bluesky_posts(feed, "shamsbot.bsky.social")
    assert [p["key"] for p in posts] == ["at://p1", "at://p2"]
    assert posts[0]["created_at"] == "2026-11-14T19:50:09Z"
    assert posts[0]["text"] == "Scoop" and posts[1]["repost"] is True


def test_append_new_records_first_seen_once(tmp_path):
    path = tmp_path / "rss" / "2026-11.csv"
    rows = [{"key": "a", "title": "A"}, {"key": "b", "title": "B"}]
    assert live.append_new(path, rows, "2026-11-14T20:00:00Z") == 2
    later = [{"key": "b", "title": "B"}, {"key": "c", "title": "C"}]
    assert live.append_new(path, later, "2026-11-14T20:02:00Z") == 1
    with open(path, newline="") as f:
        got = {r["key"]: r["first_seen"] for r in csv.DictReader(f)}
    assert got == {
        "a": "2026-11-14T20:00:00Z",
        "b": "2026-11-14T20:00:00Z",
        "c": "2026-11-14T20:02:00Z",
    }


def test_snapshots_are_kept_only_when_they_change(tmp_path):
    state: dict = {}
    folder = tmp_path / "lineups" / "2026-11-14"
    assert live.save_if_changed(folder, "200000", '{"a": 1}', state, "lineups")
    assert not live.save_if_changed(folder, "200200", '{"a": 1}', state, "lineups")
    assert live.save_if_changed(folder, "200400", '{"a": 2}', state, "lineups")
    files = sorted(p.name for p in folder.iterdir())
    assert files == ["200000.json.gz", "200400.json.gz"]
    with gzip.open(folder / "200400.json.gz", "rt") as f:
        assert f.read() == '{"a": 2}'


def test_book_rows_take_the_best_bid_and_ask():
    books = [
        {
            "asset_id": "t1",
            "timestamp": "1794669728077",
            "bids": [{"price": "0.82", "size": "5"}, {"price": "0.83", "size": "7"}],
            "asks": [{"price": "0.85", "size": "9"}, {"price": "0.84", "size": "3"}],
            "last_trade_price": "0.84",
        }
    ]
    meta = {"t1": ("nba-den-nop-2026-11-14", "m1", "Nuggets")}
    (row,) = live.book_rows(books, meta, "2026-11-14T20:00:00Z")
    assert (row["best_bid"], row["bid_size"]) == (0.83, 7.0)
    assert (row["best_ask"], row["ask_size"]) == (0.84, 3.0)
    assert row["outcome"] == "Nuggets" and row["last_trade"] == 0.84


def test_sweep_moves_only_finished_days_and_months(tmp_path):
    staging, archive = tmp_path / "staging", tmp_path / "archive"
    for rel in (
        "lineups/2026-11-13/200000.json.gz",
        "lineups/2026-11-14/200000.json.gz",
        "polymarket/2026-11-13.csv",
        "rss/2026-10.csv",
        "rss/2026-11.csv",
        "state.json",
    ):
        (staging / rel).parent.mkdir(parents=True, exist_ok=True)
        (staging / rel).write_text("x")
    moved = live.sweep(staging, archive, pd.Timestamp("2026-11-14").date())
    assert moved == 3
    assert (archive / "lineups/2026-11-13/200000.json.gz").exists()
    assert (archive / "polymarket/2026-11-13.csv").exists()
    assert (archive / "rss/2026-10.csv").exists()
    assert (staging / "lineups/2026-11-14/200000.json.gz").exists()
    assert (staging / "rss/2026-11.csv").exists() and (staging / "state.json").exists()
