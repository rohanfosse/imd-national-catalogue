#!/usr/bin/env bash
# Refresh the overleaf-paper branch from the current state of paper/ on main.
#
# Usage (from anywhere inside the repository):
#     bash paper/sync_overleaf.sh
#
# What it does:
#   1. checkout main
#   2. re-split paper/ into a temporary branch via git subtree split
#   3. fast-forward overleaf-paper to that branch
#   4. push overleaf-paper to origin with --force-with-lease
#
# Safe to run repeatedly. Does not touch main's history.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# 1. Make sure main is up to date.
git checkout main
git pull --ff-only origin main

# 2. Compute the new subtree split.
git branch -D overleaf-paper-new 2>/dev/null || true
git subtree split --prefix=paper -b overleaf-paper-new

# 3. Fast-forward overleaf-paper.
git checkout overleaf-paper 2>/dev/null || git checkout -b overleaf-paper
git reset --hard overleaf-paper-new
git branch -D overleaf-paper-new

# 4. Push to origin.
git push --force-with-lease origin overleaf-paper

git checkout main
echo "overleaf-paper branch refreshed and pushed."
