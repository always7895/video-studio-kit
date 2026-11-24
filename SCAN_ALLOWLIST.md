# Privacy scan allowlist

Scope: working tree and all git history (commit messages, author, committer, diffs).
Patterns (word-bounded where short): project and product codes, drive-letter paths, personal
account names, e-mail addresses, plus the local-only threshold number.

Rulings on hits (each hit was read and judged):

| Pattern family | Hit | Ruling |
|---|---|---|
| e-mail | the GitHub noreply author/committer address on every commit | allowed: de-identified by design, must stay the same on new commits |
| e-mail | `Co-Authored-By: Claude ... <noreply@anthropic.com>` trailers | allowed: public vendor noreply address |
| drive-letter path | `D:\\models` in the usage line of `setup/download_models.py` | allowed: generic example directory, not a real machine path |
| URL scheme (`https://`, `http://127.0.0.1`) | matched by loose drive-letter or host patterns | false positive: allowed |
| person or project names | none in tree or history | no action |

Re-run before every push and update this file if a new hit appears.
