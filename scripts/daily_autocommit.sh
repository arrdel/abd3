#!/usr/bin/env bash
# ==============================================================================
# daily_autocommit.sh
#
# Runs once per day (via cron) to commit any tracked-path changes in the ABD3
# working tree and push them to GitHub.
#
# - Respects .gitignore, so checkpoints/, outputs/, logs/, report/, the
#   non-README .md files, and virtualenvs are never committed.
# - No-op (exit 0) when there is nothing to commit.
# - All output goes to logs/autocommit.log for audit.
# ==============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/achinda1/projects/abd3}"
LOG_DIR="${REPO_DIR}/logs"
LOG_FILE="${LOG_DIR}/autocommit.log"

mkdir -p "${LOG_DIR}"
exec >>"${LOG_FILE}" 2>&1

echo
echo "=========================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] daily_autocommit starting in ${REPO_DIR}"
echo "=========================================================================="

cd "${REPO_DIR}"

# Make gh / git available under cron's stripped PATH.
export PATH="${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"

# Sanity: must be inside a git repo on branch main.
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[ERROR] not inside a git work tree; aborting."
    exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${BRANCH}" != "main" ]]; then
    echo "[WARN] expected branch 'main' but on '${BRANCH}'; continuing anyway."
fi

# Stage every tracked-path change that is not ignored.
git add -A

# If the index matches HEAD, nothing to do — exit cleanly.
if git diff --cached --quiet; then
    echo "[$(date '+%H:%M:%S')] no changes to commit."
    exit 0
fi

# Summarise what will be committed.
echo
echo "--- files staged ---"
git --no-pager diff --cached --stat
echo

STAMP="$(date '+%Y-%m-%d %H:%M:%S %z')"
SHORT_SUMMARY="$(git --no-pager diff --cached --stat | tail -1 | sed 's/^ *//')"

git commit -m "chore(autocommit): daily snapshot ${STAMP%% *}" \
           -m "${SHORT_SUMMARY}" \
           --quiet

echo "[$(date '+%H:%M:%S')] committed $(git --no-pager log -1 --format='%h %s')"

# Push to origin/main with a short retry in case of a transient network hiccup.
for attempt in 1 2 3; do
    if git push origin main; then
        echo "[$(date '+%H:%M:%S')] pushed to origin/main (attempt ${attempt})."
        exit 0
    fi
    echo "[$(date '+%H:%M:%S')] push failed on attempt ${attempt}; retrying in 15s..."
    sleep 15
done

echo "[ERROR] could not push to origin/main after 3 attempts."
exit 2
