---
name: git-commit
description: Use whenever creating a git commit in this project. Enforces the hard rule that commits are ONLY made after the user explicitly confirms in their own words (e.g. "commit the changes", "go ahead and commit", "yes commit it"), plus the conventional-commit message format and the quick one-pass grouping rule.
---

# Git Commit Skill — Confirm First, Then Conventional Commits

## Rule 1: Never commit without explicit confirmation (non-negotiable)

**NEVER run `git commit` (or `git commit --amend`, `git push`, or commit-adjacent writes like `git stash`, `git restore`, `git checkout -- .`) unless the user explicitly asked for a commit in the current conversation state.**

A commit is authorized ONLY by explicit user phrases such as:
- "commit the changes" / "commit this"
- "go ahead and commit" / "make a commit"
- "yes, commit" (in direct reply to your question)

NOT authorized by: "done", "looks good", "thanks", "next", or anything ambiguous. Finished a task? Tests pass? User said thanks? **That is NOT permission.** Silence is not consent.

**When not authorized:** stop after finishing the work and say the changes are uncommitted, e.g.:
> "Done — changes are in `<files>`, uncommitted. Say the word if you want a commit."

## Rule 2: Group related changes — one pass, no overthinking

When the user authorizes a commit and multiple files are dirty, do this **once, quickly, without deliberating**:

1. Glance at `git status` + `git diff --stat`.
2. Group the changed files by related change (e.g. feature code together, docs together, config together). Use common sense: files serving the same purpose go in the same group.
3. Commit **each group separately** with its own suitable conventional-commit message (Rule 3).

**Do NOT spend time agonizing over the perfect split** — no long analysis, no asking the user how to group unless something is genuinely ambiguous. One pass, group, commit, move on.

Example: you wrote a benchmark script, a training script, and updated docs.
- `feat(benchmark): add zero-shot baseline runner` → benchmark script + its example data
- `feat(training): add laya fine-tuning script` → training script
- `docs: update status checklist` → STATUS.md

## Rule 3: Commit message format — Conventional Commits

```
<type>(<scope>): <subject>

[optional body]
[optional footer(s)]
```

**Type** (pick the closest, don't invent):

| Type | When |
|:---|:---|
| `feat` | New feature / new capability (incl. new scripts, new files with functionality) |
| `fix` | Bug fix |
| `docs` | Documentation only (README, STATUS.md, plans) |
| `refactor` | Code change that neither fixes nor adds |
| `perf` | Performance improvement |
| `test` | Adding/fixing tests |
| `build` | Build system, dependencies, requirements.txt |
| `chore` | Maintenance, .gitignore, housekeeping |

**Scope** (optional but preferred): short lowercase noun for the area — e.g. `feat(benchmark)`, `feat(training)`, `docs(status)`, `build(deps)`.

**Subject rules:**
- Imperative mood, present tense: "add" not "added"/"adds"
- Lowercase start, no trailing period
- ≤ 72 chars, ideally ≤ 50
- No attribution footers / co-authored-by unless the user asks

**Body** (optional, when the "why" isn't obvious): wrapped at 72 chars, explains WHY not WHAT.

Examples:
```
feat(benchmark): add zero-shot baseline runner for laya-multilingual
fix(training): freeze act_head when no action labels in data
docs(status): mark environment phase complete, note benchmark not on hub
build(deps): pin laya==0.3.4 and fine-tuning stack
```

## Commit hygiene

- Inspect first: `git status`, `git diff`, `git log --oneline -10`
- Stage only intended files (`git add <paths>`); never blind `git add -A`; never commit secrets, keys, `venv/`, `.env`, HF caches
- This repo: `.gitignore` excludes `/venv` and `/plans` — never force-add them
- `STATUS.md` is a living status doc; only stage it when the changes belong to the user's requested work or a `docs(status)` group
- If a commit fails or hooks reject it: fix and create a NEW commit — do not amend the failed one
