# Linking Overleaf to this repository

This repository ships the manuscript in two places that you can pick from
depending on your Overleaf plan and personal preference :

| Location | Branch | Content layout |
| --- | --- | --- |
| `paper/` on `main` | `main` | `paper/imd_national.tex`, `paper/references_imd.bib`, `paper/figures/` |
| Root of `overleaf-paper` branch | `overleaf-paper` | `imd_national.tex`, `references_imd.bib`, `figures/` |

Overleaf expects the LaTeX files at the project root. **Always sync the
`overleaf-paper` branch, not `main`.**

The `overleaf-paper` branch is derived from `paper/` via `git subtree
split` ; it preserves the commit history of paper-only changes and is
auto-updated whenever `paper/` evolves on `main` (see the *Updating* section
below).

---

## Option 1 — Overleaf Premium (GitHub auto-sync)

If you have Overleaf Premium or an institutional licence :

1. Open your Overleaf project.
2. **Menu → GitHub → Link to GitHub**.
3. Authorise Overleaf to read `rohanfosse/imd-national-catalogue`.
4. In the branch picker, select **`overleaf-paper`** (not `main`).
5. Set **`imd_national.tex`** as the main document.

From now on, every push to `overleaf-paper` triggers an Overleaf re-sync,
and every Overleaf edit creates a commit on the branch.

---

## Option 2 — Free plan (manual git remote)

The Overleaf project for this paper is
<https://www.overleaf.com/project/6a01c1954b0135ab254c83ca>, with the
git endpoint at
`https://git.overleaf.com/6a01c1954b0135ab254c83ca`.

1. From a terminal :

   ```bash
   # Clone the overleaf-paper branch into a local working folder
   git clone -b overleaf-paper \
       https://github.com/rohanfosse/imd-national-catalogue.git overleaf-mirror
   cd overleaf-mirror

   # Add Overleaf as a second remote
   git remote add overleaf \
       https://git.overleaf.com/6a01c1954b0135ab254c83ca

   # Push the paper to Overleaf
   git push overleaf overleaf-paper:master
   ```

   When Overleaf prompts for a password, paste the Git Auth token from
   Overleaf's **Account Settings → Git authentication**.

2. To pull Overleaf-side edits back :

   ```bash
   git pull overleaf master
   git push origin overleaf-paper
   ```

---

## Updating the `overleaf-paper` branch from `main`

After any edit to `paper/` on `main`, refresh the side branch :

```bash
bash paper/sync_overleaf.sh
```

The script runs the four-step subtree split + force-with-lease push.
Under the hood it does :

```bash
cd "$(git rev-parse --show-toplevel)"
git checkout main
git subtree split --prefix=paper -b overleaf-paper-new
git checkout overleaf-paper
git reset --hard overleaf-paper-new
git branch -D overleaf-paper-new
git push --force-with-lease origin overleaf-paper
```

The `--force-with-lease` flag is the safe variant of force-push : it
refuses to overwrite remote changes that Overleaf may have committed
since the last sync. If it fails, do a `git pull overleaf master` first.

---

## Why two branches instead of moving the manuscript to the repo root?

The repository has multiple deliverables : commune-level Parquet catalogue,
pipeline code, documentation, and the manuscript. Promoting the
manuscript to the root would crowd the landing page and confuse non-paper
consumers. The `overleaf-paper` branch gives Overleaf the flat layout it
expects without breaking the main project structure.
