#!/usr/bin/env bash
# Install muck for an evaluator: the CLI (a command on PATH) + the Claude skill (a directory
# Claude Code loads). These are two separate things — this installs both.
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"

# 1) The muck CLI  ->  puts `muck` on PATH. Clean, non-editable install (no dev-env churn,
#    no `--no-editable` gymnastics — that quirk only affects editable dev installs).
if command -v uv >/dev/null 2>&1; then
  uv tool install --force "$REPO"
elif command -v pipx >/dev/null 2>&1; then
  pipx install --force "$REPO"
else
  python3 -m pip install --user "$REPO"
fi

# 2) The Claude Agent Skill  ->  Claude Code discovers it by location, not by package install.
SKILLS_DIR="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"
mkdir -p "$SKILLS_DIR/muck"
cp -R "$REPO/skill/." "$SKILLS_DIR/muck/"

echo
echo "✓ CLI:   $(command -v muck >/dev/null 2>&1 && muck --version || echo 'installed (restart shell for PATH)')"
echo "✓ skill: $SKILLS_DIR/muck"
echo
echo "Next — build an index over the corpus, then investigate:"
echo "  mkdir -p run/corpus/.muck && cp $REPO/configs/corpus_q1.toml run/corpus/.muck/config.toml"
echo "  cp -R <the challenge corpus> run/corpus/data"
echo "  cd run/corpus && muck build data          # ~90s: 35,987 docs / 110,359 chunks"
