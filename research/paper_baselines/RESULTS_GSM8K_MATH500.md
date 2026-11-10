# GSM8K and MATH500 baseline results

Checked on October 13, 2026. Both released 0.6B checkpoints completed both
benchmarks: 1,319 GSM8K problems and 500 MATH500 problems, one completion per
problem. All input IDs, problem text, reference answers and aggregate scores
were checked against the saved per-sample records and input manifest. There
were no missing or duplicate problems. The run has a completion marker.

## Final-answer accuracy

The primary score grades text after `</think>` when the response contains a
thinking block. Unfinished reasoning does not count as a final answer.

| Benchmark | Standard-0.6B | TaH+-0.6B | TaH+ gain |
| --- | ---: | ---: | ---: |
| GSM8K | 701/1,319 = **53.15%** | 806/1,319 = **61.11%** | **+7.96 percentage points** |
| MATH500 | 171/500 = **34.20%** | 201/500 = **40.20%** | **+6.00 percentage points** |

On GSM8K, TaH+ answered 225 problems correctly that Standard missed, while
Standard answered 120 correctly that TaH+ missed. On MATH500 those counts
were 79 and 49. This is a single sampled completion per problem; repeated
seeds have not been run.

## Comparison with the paper

| Benchmark/model | Our final-answer score | Our raw repository score | Paper Table 2 |
| --- | ---: | ---: | ---: |
| GSM8K / Standard | 53.15% | 58.38% | 58.2% |
| GSM8K / TaH+ | 61.11% | 64.97% | 67.6% |
| MATH500 / Standard | 34.20% | 37.40% | 39.9% |
| MATH500 / TaH+ | 40.20% | 46.40% | 51.8% |

Paper values are published reference scores, not measurements from this
run: [Think-at-Hard v4, Table 2](https://arxiv.org/html/2511.08577v4#S6.T2).
The raw repository grader parses the entire generated response, including
the thinking block. Its higher scores show that grading differences explain
part of the gap to the paper. Remaining differences have not been diagnosed;
these results are not an exact reproduction of the published scores.

## Compute and completion behavior

| Benchmark/model | Average prediction depth | Continued predictions | Hit token limit | Generation time | Peak allocated GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K / Standard | 1.0000 | 0% | 370/1,319 = 28.05% | 7.69 h | 5.03 GiB |
| GSM8K / TaH+ | 1.0150 | 1.50% | 269/1,319 = 20.39% | 8.55 h | 5.58 GiB |
| MATH500 / Standard | 1.0000 | 0% | 210/500 = 42.00% | 7.62 h | 9.35 GiB |
| MATH500 / TaH+ | 1.0100 | 1.00% | 226/500 = 45.20% | 10.32 h | 17.03 GiB |

Total recorded generation time was **34.17 hours**. This excludes model
loading and grading. It is the sum of actual batch generation runtimes,
not individual response latency. TaH+ took about 11.3% longer on GSM8K
and 35.5% longer on MATH500; sparse extra passes do not imply an equal
percentage increase in wall-clock time.

GSM8K mean output lengths were 2,169 tokens for Standard and 1,819 for TaH+.
MATH500 mean lengths were 5,286 and 5,275 tokens. The high MATH500 truncation
rates are a material limitation at the fixed 8,192-token budget.

## Settings and saved evidence

- Checkpoints: `nics-efc/Standard-0.6B` and `nics-efc/TaH-plus-0.6B`.
- Exact problem text and answers recovered from the authors' public
  evaluation artifacts; generated author solutions were discarded.
- Temperature 0.6, top-p 0.95, top-k 20, min-p 0; base seed 42 with
  deterministic per-batch seeds; batch size 8.
- GSM8K limit 4,096 generated tokens; MATH500 limit 8,192.
- TaH continuation threshold 0.9, maximum depth 2; frozen checkpoints.
- L40S-24Q, BF16, SDPA. No entropy router, forced-depth ablations, or
  trained bandit router was included in this run.

Remote full records: `/root/TaH/research/paper_baselines/focused_math/`.
Local audit, config and summaries: `focused_math_results/` beside this report.

These establish the released Standard-versus-TaH+ comparison for our two
main benchmarks. The next project baselines should force depth 1 or 2 on
the same TaH+ checkpoint and add entropy routing, enabling fair comparison
with the proposed learned router at matched compute budgets.
