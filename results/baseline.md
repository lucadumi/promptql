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
| 2026-07-31T12:07:20+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs` | text2sql_eval | mps | 20 | 100.0% | 100.0% |
| 2026-07-31T12:07:30+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs` | text2sql_eval_paraphrase | mps | 20 | 100.0% | 100.0% |
| 2026-07-31T12:07:40+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs` | text2sql_eval_bookstore | mps | 20 | 100.0% | 100.0% |
| 2026-07-31T12:07:52+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs` | text2sql_eval_join | mps | 12 | 100.0% | 100.0% |
| 2026-07-31T12:08:03+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs` | text2sql_eval_join_bookstore | mps | 11 | 81.8% | 81.8% |
| 2026-07-31T12:08:18+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs` | text2sql_eval_blind_v1_retired | mps | 24 | 75.0% | 87.5% |
| 2026-07-31T12:27:33+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs-rebalanced` | text2sql_eval | mps | 20 | 95.0% | 95.0% |
| 2026-07-31T12:27:44+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs-rebalanced` | text2sql_eval_paraphrase | mps | 20 | 95.0% | 95.0% |
| 2026-07-31T12:27:55+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs-rebalanced` | text2sql_eval_bookstore | mps | 20 | 95.0% | 95.0% |
| 2026-07-31T12:28:07+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs-rebalanced` | text2sql_eval_join | mps | 12 | 100.0% | 100.0% |
| 2026-07-31T12:28:19+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs-rebalanced` | text2sql_eval_join_bookstore | mps | 11 | 72.7% | 72.7% |
| 2026-07-31T12:28:37+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs-rebalanced` | text2sql_eval_blind_v1_retired | mps | 24 | 83.3% | 91.7% |
| 2026-07-31T12:30:31+00:00 | `Qwen/Qwen2.5-0.5B-Instruct` | text2sql_eval_blind_v2 | mps | 30 | 10.0% | 10.0% |
| 2026-07-31T12:30:58+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-constructs` | text2sql_eval_blind_v2 | mps | 30 | 30.0% | 36.7% |
| 2026-07-31T12:33:02+00:00 | `Qwen/Qwen2.5-1.5B-Instruct` | text2sql_eval_blind_v2 | mps | 30 | 23.3% | 43.3% |
| 2026-07-31T13:21:43+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup` | text2sql_eval | mps | 20 | 100.0% | 100.0% |
| 2026-07-31T13:21:54+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup` | text2sql_eval_paraphrase | mps | 20 | 100.0% | 100.0% |
| 2026-07-31T13:22:04+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup` | text2sql_eval_bookstore | mps | 20 | 95.0% | 100.0% |
| 2026-07-31T13:22:15+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup` | text2sql_eval_join | mps | 12 | 100.0% | 100.0% |
| 2026-07-31T13:22:27+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup` | text2sql_eval_join_bookstore | mps | 11 | 90.9% | 90.9% |
| 2026-07-31T13:22:43+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup` | text2sql_eval_blind_v1_retired | mps | 24 | 75.0% | 87.5% |
| 2026-07-31T13:45:04+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joinorder` | text2sql_eval | mps | 20 | 100.0% | 100.0% |
| 2026-07-31T13:45:15+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joinorder` | text2sql_eval_paraphrase | mps | 20 | 100.0% | 100.0% |
| 2026-07-31T13:45:27+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joinorder` | text2sql_eval_bookstore | mps | 20 | 95.0% | 95.0% |
| 2026-07-31T13:45:40+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joinorder` | text2sql_eval_join | mps | 12 | 100.0% | 100.0% |
| 2026-07-31T13:45:52+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joinorder` | text2sql_eval_join_bookstore | mps | 11 | 18.2% | 18.2% |
| 2026-07-31T13:46:09+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joinorder` | text2sql_eval_blind_v1_retired | mps | 24 | 70.8% | 87.5% |
| 2026-07-31T13:49:25+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup` | text2sql_eval_blind_v2 | mps | 30 | 30.0% | 40.0% |
| 2026-07-31T14:30:28+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup [repair x2]` | text2sql_eval | mps | 20 | 100.0% | 100.0% |
| 2026-07-31T14:30:38+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup [repair x2]` | text2sql_eval_paraphrase | mps | 20 | 100.0% | 100.0% |
| 2026-07-31T14:30:49+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup [repair x2]` | text2sql_eval_bookstore | mps | 20 | 95.0% | 100.0% |
| 2026-07-31T14:31:00+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup [repair x2]` | text2sql_eval_join | mps | 12 | 100.0% | 100.0% |
| 2026-07-31T14:31:13+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup [repair x2]` | text2sql_eval_join_bookstore | mps | 11 | 100.0% | 100.0% |
| 2026-07-31T14:31:29+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup [repair x2]` | text2sql_eval_blind_v1_retired | mps | 24 | 75.0% | 87.5% |
| 2026-07-31T14:32:27+00:00 | `Qwen/Qwen2.5-0.5B-Instruct [repair x2]` | text2sql_eval_blind_v2 | mps | 30 | 13.3% | 13.3% |
| 2026-07-31T14:33:04+00:00 | `Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup [repair x2]` | text2sql_eval_blind_v2 | mps | 30 | 30.0% | 40.0% |
