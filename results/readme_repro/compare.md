| Model | Method | n | Ours % | SE | Numeric rescore % | Paper % | Gap | Gap in SE | Median answer chars | Has </think> % |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | baseline | 100 | 90.0 | 3.0 | 90.0 | 82.4 | +7.6 | +2.5 | 1484 | 98 |
| Qwen3-4B | latent_mas | 100 | 90.0 | 3.0 | 91.0 | 88.2 | +1.8 | +0.6 | 1425 | 98 |
| Qwen3-8B | baseline | 80 | 90.0 | 3.4 | 90.0 | 81.1 | +8.9 | +2.7 | 1618 | 96 |
| Qwen3-8B | latent_mas | 100 | 90.0 | 3.0 | 90.0 | 93.8 | -3.8 | -1.3 | 1039 | 93 |

Paired, LatentMAS against baseline on the same questions:
  Qwen3-4B: n=100, baseline 90, LatentMAS 90, LatentMAS wins 2, losses 2, exact McNemar p = 1.000
  Qwen3-8B: n=80, baseline 72, LatentMAS 72, LatentMAS wins 1, losses 1, exact McNemar p = 1.000

Paired, our LatentMAS run against the authors' released LatentMAS log, same questions:
  Qwen3-4B: n=100, ours 90, authors 90, agree on 94, only ours right 3, only authors right 3
  Qwen3-8B: n=100, ours 90, authors 93, agree on 95, only ours right 1, only authors right 4
