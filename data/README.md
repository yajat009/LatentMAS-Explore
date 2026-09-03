# `data/` — datasets that ship with the repo

Only `medqa.json` lives here. The other nine tasks (`gsm8k`, `mbppplus`,
`humanevalplus`, `arc_easy`, `arc_challenge`, `gpqa`, `aime2024`, `aime2025`) are
pulled from the Hugging Face Hub at runtime by `data.py`, into `$HF_HOME`
(set to `/pub/$USER/hf` by the scripts in `repro/`) — not into this directory.

Dataset sizes, for sizing a run: gsm8k 1319 · mbppplus 378 · humanevalplus 164 ·
arc_easy 2376 · arc_challenge 1172 · medqa 300 · gpqa 198 · aime24/25 30 each.
