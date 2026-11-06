# 08 Content audit

**Status: PILOT.** One task, 200 questions, one seed, no generated answers. The result has a gap for the method with `W_a` on.

The audit asks whether the latent messages carry information about the specific question.
It gives the Judger different memories and scores the gold answer by teacher forcing. The score is log p(gold), with no accuracy measurement.

## Result

GSM8K, 20 latent steps. Each row is X minus Y on the same questions, with a 95% bootstrap interval. Above zero means X scores the gold answer higher.

| Part | X minus Y | Qwen3-4B | Qwen3-8B |
|---|---|---|---|
| Agent prompts | prompts only − Judger alone | −12.28 [−13.2, −11.4] | −9.12 [−10.0, −8.1] |
| Extra positions | pause tokens − prompts only | +0.70 [0.4, 1.0] | −1.56 [−1.8, −1.3] |
| Any real message | swapped − pause tokens | +1.87 [0.7, 3.0] | +3.01 [2.1, 3.9] |
| **Question-specific content** | **real − swapped** | **+0.40 [−0.8, 1.5]** | **−0.57 [−1.5, 0.4]** |
| Ridge `W_a` against identity | ridge − identity | +0.05 [−0.6, 0.7] | +1.57 [0.9, 2.3] |

What the numbers show:

- Memory from a different question scored as well as the real memory. Both intervals include zero.
- A real message from any question beat blank pause vectors. Both intervals exclude zero.
- On Qwen3-4B, ridge and identity are the same map, so +0.05 shows the noise level of the setup.

Activation patching agrees. Swapping the vectors of one agent changed nothing measurable, and every swap interval includes zero. Replacing them with pause vectors hurt, and every pause interval excludes zero.

A logit lens reads each latent vector as token scores. On 4B, the gold answer's first token reaches a median rank of about 5 to 10 by step 8. On 8B, it stays in the thousands.

## Limits

- **The 8B test used the identity map.** The swapped condition always uses identity. On 8B, that is not the paper's method, and ridge scored higher than identity. There is no swapped-with-ridge condition, so question-specific content with `W_a` on is **not measured**. Earlier summaries stated the null result for 8B without this limit.
- **The score is not accuracy.** The gold answer is scored with no reasoning text, and gold is top-1 only about 50% of the time, where real accuracy is about 95%.
- **An interval that includes zero does not prove zero.** The 4B interval allows up to +1.5.
- **GSM8K only.** The MBPP+ result in `../02_latent_steps_and_think_tag/` has not had this audit.
- 8B ran at batch size 2 after running out of memory at batch size 8. All other runs used batch size 8.

## Evidence

- `content_audit_output_*.txt`, `message_decoding_output_*.txt`: printed output of the runs
- `../../data/content_audit_scores.csv`: per-question scores for every condition
