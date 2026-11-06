# LatentMAS-Explore

This repo tests whether the results of the paper
[Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639) (LatentMAS) reproduce.
It is a fork of [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS), and it adds the test scripts and their results.

LatentMAS lets language-model agents pass internal state to each other, not text. The paper reports equal or better accuracy, with fewer tokens and faster runs than agents that write text.

## Main results

Every number has a label and evidence in [results/](results/README.md).

| Finding | Evidence | Label |
|---|---|---|
| On Qwen3-4B MBPP+, the paper's 73.5% does not reproduce with the method on and `--think`. | First 84 questions, batch size 1: 47.6%, against 75.0% with the method off (3 wins, 26 losses, p = 0.00002) | CONFIRMED |
| The size of that drop depends on the `--think` flag, and the reason is unknown. | Steps 10 scored 25/45 with the flag and 35/45 without it. Steps 20 scored about the same both ways. | OPEN |
| Qwen3-14B loses almost nothing from the latent steps on MBPP+. | 42/45 with the method off, 41/45 with it on | SUGGESTIVE |
| The paper's 14B MBPP+ LatentMAS number used 0 latent steps, so the method was off. The 4B number probably did too. | All 1,134 entries of the authors' log show 0 steps | CONFIRMED / SUGGESTIVE |
| LatentMAS beats the single agent on 14B GPQA. | 49.0% against 39.9% on 198 questions, p = 0.015, but the padding bug affects both runs | BUG-AFFECTED |
| Batched runs are broken by right padding. | 93.3% of batched LatentMAS sequences start from a pad token | CONFIRMED |
| GSM8K results cannot separate the methods. | All methods score about 95% on the first questions | INCONCLUSIVE |
| On GSM8K, latent messages show no question-specific content with the identity map. | 200 questions, one seed, 4B +0.40 [−0.8, 1.5], 8B −0.57 [−1.5, 0.4]. Not measured with `W_a` on. | PILOT |

## Repo layout

| Folder | What it holds |
|---|---|
| [architecture/](architecture/README.md) | **Part 1.** How the method works, the original Gen-Verse code, the paper's figures, and a list of every change from upstream |
| [results/](results/README.md) | **Part 2.** Every result, labelled by how far to trust it, with its evidence. `results/test_runs/` holds the scripts that produced the results. |

`data/` holds the upstream MedQA dataset. It stays at the root because `architecture/data.py` reads it by that path.

## Quick start

```bash
pip install -r requirements.txt
python -m results.test_runs.test_experiments                       # CPU checks, about 30 s
python results/test_runs/analysis/recompute_result_tables.py       # recompute the paired results, no GPU
python architecture/run.py --method latent_mas --model_name Qwen/Qwen3-4B --task mbppplus \
    --prompt sequential --latent_steps 20 --generate_bs 1 --max_samples 20
```

## License

The upstream code is under its original license. See [LICENSE](LICENSE).
