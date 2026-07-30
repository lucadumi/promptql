# Baseline results

| timestamp (UTC) | model | eval set | device | n | exact-match | exec-accuracy |
|---|---|---|---|---|---|---|
| 2026-07-28T08:38:43+00:00 | `Qwen/Qwen2.5-0.5B-Instruct` | text2sql_eval | mps | 20 | 40.0% | 65.0% |
| 2026-07-28T08:38:53+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b` | text2sql_eval | mps | 20 | 100.0% | 100.0% |
| 2026-07-28T08:39:07+00:00 | `Qwen/Qwen2.5-0.5B-Instruct` | text2sql_eval_paraphrase | mps | 20 | 30.0% | 55.0% |
| 2026-07-28T08:39:17+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b` | text2sql_eval_paraphrase | mps | 20 | 70.0% | 75.0% |
| 2026-07-28T11:24:26+00:00 | `Qwen/Qwen2.5-0.5B-Instruct` | text2sql_eval_bookstore | mps | 20 | 55.0% | 70.0% |
| 2026-07-28T11:24:45+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b` | text2sql_eval_bookstore | mps | 20 | 100.0% | 100.0% |
| 2026-07-29T11:15:31+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-aug` | text2sql_eval | mps | 20 | 100.0% | 100.0% |
| 2026-07-29T11:15:42+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-aug` | text2sql_eval_paraphrase | mps | 20 | 80.0% | 90.0% |
| 2026-07-29T11:15:52+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-aug` | text2sql_eval_bookstore | mps | 20 | 100.0% | 100.0% |
| 2026-07-30T11:43:18+00:00 | `Qwen/Qwen2.5-0.5B-Instruct` | text2sql_eval_join | mps | 12 | 0.0% | 8.3% |
| 2026-07-30T11:43:30+00:00 | `Qwen/Qwen2.5-0.5B-Instruct` | text2sql_eval_join_bookstore | mps | 11 | 0.0% | 63.6% |
| 2026-07-30T11:43:39+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-aug` | text2sql_eval_join | mps | 12 | 0.0% | 0.0% |
| 2026-07-30T11:43:48+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-aug` | text2sql_eval_join_bookstore | mps | 11 | 0.0% | 9.1% |
| 2026-07-30T12:36:29+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-join` | text2sql_eval | mps | 20 | 95.0% | 95.0% |
| 2026-07-30T12:36:39+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-join` | text2sql_eval_paraphrase | mps | 20 | 90.0% | 95.0% |
| 2026-07-30T12:36:50+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-join` | text2sql_eval_bookstore | mps | 20 | 100.0% | 100.0% |
| 2026-07-30T12:37:01+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-join` | text2sql_eval_join | mps | 12 | 100.0% | 100.0% |
| 2026-07-30T12:37:12+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-join` | text2sql_eval_join_bookstore | mps | 11 | 90.9% | 100.0% |
| 2026-07-30T13:05:36+00:00 | `Qwen/Qwen2.5-0.5B-Instruct` | text2sql_eval_blind | mps | 24 | 20.8% | 29.2% |
| 2026-07-30T13:05:51+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-join` | text2sql_eval_blind | mps | 24 | 70.8% | 75.0% |
| 2026-07-30T13:24:48+00:00 | `Qwen/Qwen2.5-1.5B-Instruct` | text2sql_eval | mps | 20 | 75.0% | 95.0% |
| 2026-07-30T13:25:16+00:00 | `Qwen/Qwen2.5-1.5B-Instruct` | text2sql_eval_paraphrase | mps | 20 | 40.0% | 85.0% |
| 2026-07-30T13:25:38+00:00 | `Qwen/Qwen2.5-1.5B-Instruct` | text2sql_eval_bookstore | mps | 20 | 85.0% | 100.0% |
| 2026-07-30T13:26:07+00:00 | `Qwen/Qwen2.5-1.5B-Instruct` | text2sql_eval_join | mps | 12 | 8.3% | 66.7% |
| 2026-07-30T13:26:35+00:00 | `Qwen/Qwen2.5-1.5B-Instruct` | text2sql_eval_join_bookstore | mps | 11 | 9.1% | 90.9% |
| 2026-07-30T13:27:20+00:00 | `Qwen/Qwen2.5-1.5B-Instruct` | text2sql_eval_blind | mps | 24 | 45.8% | 75.0% |
| 2026-07-30T13:30:23+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-join` | text2sql_eval_blind | cpu | 24 | 70.8% | 75.0% |
| 2026-07-30T13:30:37+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-join` | text2sql_eval | cpu | 20 | 95.0% | 95.0% |
| 2026-07-30T13:33:11+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-join [int8]` | text2sql_eval | cpu | 20 | 75.0% | 75.0% |
| 2026-07-30T13:33:40+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-join [int8]` | text2sql_eval_blind | cpu | 24 | 16.7% | 20.8% |
