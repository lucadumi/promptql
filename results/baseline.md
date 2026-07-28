# Baseline results

| timestamp (UTC) | model | eval set | device | n | exact-match | exec-accuracy |
|---|---|---|---|---|---|---|
| 2026-07-28T08:38:43+00:00 | `Qwen/Qwen2.5-0.5B-Instruct` | text2sql_eval | mps | 20 | 40.0% | 65.0% |
| 2026-07-28T08:38:53+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b` | text2sql_eval | mps | 20 | 100.0% | 100.0% |
| 2026-07-28T08:39:07+00:00 | `Qwen/Qwen2.5-0.5B-Instruct` | text2sql_eval_paraphrase | mps | 20 | 30.0% | 55.0% |
| 2026-07-28T08:39:17+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b` | text2sql_eval_paraphrase | mps | 20 | 70.0% | 75.0% |
