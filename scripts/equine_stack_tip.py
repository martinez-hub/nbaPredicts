"""Find the tip of EQUINE's open-PR stack and pin the project to it.

The open pull requests on mit-ll-responsible-ai/equine form a stack: each PR's
base branch is the previous PR's head, and the first one targets ``main``.
Installing the tip gets every open change at once. This script walks the
stack from ``main``, checks that every branch contains its base (so the tip
really includes everything), writes ``equine_stack.json`` and rewrites the
SHA in ``pyproject.toml``. Run ``uv lock`` afterwards.

    uv run python scripts/equine_stack_tip.py [--write]

Set GITHUB_TOKEN to avoid the unauthenticated rate limit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import requests

REPO = "mit-ll-responsible-ai/equine"
ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"


def gh(path: str, **params: object) -> object:
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(f"{API}{path}", params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="update the pins")
    args = parser.parse_args()

    prs = gh(f"/repos/{REPO}/pulls", state="open", per_page=100)
    same_repo = [p for p in prs if p["head"]["repo"]["full_name"] == REPO]
    by_base: dict[str, list[dict]] = {}
    for p in same_repo:
        by_base.setdefault(p["base"]["ref"], []).append(p)

    chain, branch = [], "main"
    while True:
        # Dependabot PRs also target main; the stack is the branch that other
        # PRs build on, so prefer a child that is itself a base.
        children = by_base.get(branch, [])
        children.sort(key=lambda p: (p["head"]["ref"] not in by_base, p["number"]))
        if not children:
            break
        nxt = children[0]
        chain.append(nxt)
        branch = nxt["head"]["ref"]

    if not chain:
        # Everything merged: the newest code is main itself.
        main_sha = gh(f"/repos/{REPO}/branches/main")["commit"]["sha"]
        tip_branch, sha = "main", main_sha
        print(f"no open PR stack on main; tip: main @ {sha}")
    else:
        base = "main"
        for p in chain:
            cmp = gh(f"/repos/{REPO}/compare/{base}...{p['head']['ref']}")
            behind = cmp["behind_by"]
            status = "ok" if behind == 0 else f"BEHIND {base} by {behind}"
            print(f"#{p['number']:<4} {p['head']['ref']:<40} {status}  {p['title']}")
            if behind:
                # Mid-rebase: the tip would miss commits below it. Keep the
                # current pin and try again next run.
                print(
                    "::warning::the EQUINE PR stack is not linear "
                    f"(#{p['number']} is behind {base}); keeping the current pin"
                )
                return 0
            base = p["head"]["ref"]
        tip_branch, sha = chain[-1]["head"]["ref"], chain[-1]["head"]["sha"]
        print(f"\ntip: {tip_branch} @ {sha}")
    left_out = sorted(
        p["number"] for p in prs if p["number"] not in {c["number"] for c in chain}
    )
    if left_out:
        print(f"open PRs outside the stack (not included): {left_out}")

    stack_file = ROOT / "equine_stack.json"
    prs_in_stack = [p["number"] for p in chain]
    if stack_file.exists():
        current = json.loads(stack_file.read_text())
        if current.get("commit") == sha and current.get("prs") == prs_in_stack:
            print("pin is already at the tip; nothing to do")
            return 0

    if args.write:
        stack_file.write_text(
            json.dumps(
                {
                    "repo": REPO,
                    "branch": tip_branch,
                    "commit": sha,
                    "prs": prs_in_stack,
                    "outside_stack": left_out,
                    "resolved_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                indent=1,
            )
            + "\n"
        )
        pyproject = ROOT / "pyproject.toml"
        text = pyproject.read_text()
        new = re.sub(r"(equine\.git@)[0-9a-f]{40}", rf"\g<1>{sha}", text)
        pyproject.write_text(new)
        print("wrote equine_stack.json and pyproject.toml; now run `uv lock`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
