#!/bin/bash
# Backup trigger for the daily workflow, run by launchd on the owner's Mac.
#
# GitHub's scheduler runs the workflow hours late or not at all (2026-10-28:
# the first scheduled runs came about six hours after their slots). At each
# launchd slot this starts a run unless one is queued, running, or was
# started in the last WINDOW minutes (GitHub's own slot then fired on time).
# Runs are idempotent, so a late scheduled run that follows does no harm.
#
#   nbapredicts-dispatch.sh [--dry-run]
#
# Installed to ~/.local/bin; see com.martinez-hub.nbapredicts-dispatch.plist.
set -euo pipefail

REPO=martinez-hub/nbaPredicts
WINDOW=75 # minutes; catches GitHub's slot before each launchd slot, EDT or EST
GH=/opt/homebrew/bin/gh

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
since=$(date -u -v-"${WINDOW}"M +%Y-%m-%dT%H:%M:%SZ)
recent=$("$GH" api "repos/$REPO/actions/workflows/daily.yml/runs?per_page=10" \
  --jq "[.workflow_runs[] | select(.status != \"completed\" or .created_at >= \"$since\")] | length")

if [ "$recent" -gt 0 ]; then
  echo "$(stamp) skip: $recent daily run(s) queued, running or started since $since"
elif [ "${1:-}" = "--dry-run" ]; then
  echo "$(stamp) would dispatch: no daily run since $since"
else
  "$GH" workflow run daily -R "$REPO"
  echo "$(stamp) dispatched: no daily run since $since"
fi
