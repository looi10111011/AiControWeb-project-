# แผน Controlled Benchmark + Ablation (สำหรับรายงานฝึกงาน บทที่ 2.3.4)

เป้าหมาย: แทนภาพก่อน/หลังในรายงาน (ซึ่งต่าง task ต่าง model เทียบกันไม่ได้) ด้วยตัวเลขจากการทดลองที่คุมตัวแปร
และตอบให้ได้ว่า "แต่ละฟีเจอร์ช่วยจริงไหม ช่วยเท่าไร"

ใช้ของที่มีอยู่แล้วทั้งหมด:

- `benchmark_target/` — HRM ในเครื่อง (port 8100) + Control Plane: reset fixture ทุก attempt, ตัดสินผลจาก DB ไม่ใช่จากคำพูด agent
- `benchmark_target/runner/attempt_runner.py::run_attempt()` — วงจร 1 attempt (fixture → precondition → snapshot → agent → verify → integrity)
- `benchmark_target/runner/agent_adapters.py::HermesAgentAdapter` — รัน `Orchestrator.run_task()` จริง และเขียน telemetry เมื่อส่ง `run_id`
- `backend/app/core/hrm_local_eval.py` — `ensure_target_running()` + `allow_internal_navigation_here()` + การรัน Control Plane ในโปรเซส
- `backend/app/core/telemetry.py` — `data/token_usage.jsonl` (1 แถว/task) + `data/step_trace.jsonl` (1 แถว/step, มี `failure_class` และ `timing`) ผูกด้วย `run_id`
- `benchmark_target/runner/stats.py::wilson_interval()` — ช่วงความเชื่อมั่น 95% ของ success rate

## 1. ชุด task (10 ตัว, ตัดสินจาก DB ทั้งหมด)

| กลุ่ม | task_id | หมายเหตุ |
|---|---|---|
| hrm-local เดิม (ผ่าน 6/6 เมื่อ 14 ต.ค.) | PIM-EDIT-VERIFY-01, PIM-CREATE-EMPLOYEE, PIM-DEACTIVATE-EMPLOYEE, ADMIN-CREATE-USER-ACCOUNT, LEAVE-APPLY-01, RECRUITMENT-SHORTLIST-01 | ผ่านการพิสูจน์แล้วว่า stub agent ทำไม่ผ่าน |
| เพิ่ม | PIM-REACTIVATE-EMPLOYEE, TIME-CREATE-SUBMIT-TIMESHEET, TIME-APPROVE-BY-SUPERVISOR, RECRUITMENT-FULL-HIRE-WORKFLOW-02 | actor เดียว, ค่าในเป้าหมายระบุชัด, task ยาว (context management มีผลเมื่อเกิน 4 step) |

ตัดออกโดยตั้งใจ:

- `verification.type` = `http_redirect` / `download_artifact` — `attempt_runner` คืน INCONCLUSIVE เสมอ
- `PIM-SEARCH-*`, `AUTH-LOGIN-SUCCESS-ADMIN` — verifier เป็นจริงเสมอแม้ agent ไม่ทำอะไร (AUTH เช็คแค่ EMP-0001 active)
- `LEAVE-FULL-CYCLE-APPROVE-*` และ task หลาย actor — adapter ส่งรหัสผ่านให้ actor แรกเท่านั้น และเป้าหมายไม่บอกวันที่แต่ verifier ล็อก `2026-04-06` (ปัญหาเดียวกับที่เพิ่งแก้ใน LEAVE-APPLY)
- task `RECOVERY-FAULT-INJECTED-*` / `EXPERT-*` — ใส่ความสุ่มเพิ่ม ไม่เหมาะกับการเทียบฟีเจอร์

## 2. Arm ของการทดลอง (ทุก arm รันบน commit เดียวกัน, โปรเซสเดียวกัน, สลับด้วย `unittest.mock.patch`)

ไม่ต้องแก้ `orchestrator.py` — ทุกจุดที่ปิดเป็นชื่อระดับโมดูลหรือค่าใน `settings` ที่ถูกอ่านตอนรัน

| Arm | ปิดอะไร | patch |
|---|---|---|
| **FULL** | ไม่ปิด (ค่าปัจจุบัน) | — |
| **NO_CONTEXT** | การบีบ context ทั้งกลุ่ม W_token_trim / W_token_cut | `orchestrator._COMPACT_AFTER_STEPS = 10**9`; `orchestrator._compact_stale_user_turns = lambda m, g, **k: (m, 0)`; `orchestrator._dedupe_stale_snapshots = lambda m, **k: m`; `settings.snapshot_max_elements = 10**6`; `settings.read_page_data_max_rows = 10**6` |
| **NO_PROMPT_SECTIONS** | การส่ง system prompt แบบเลือกบล็อก (W_prompt_sections) | `orchestrator._resolve_prompt_sections = lambda prev, **k: llm.ALL_PROMPT_SECTIONS` |
| **NO_VERIFY** | การตรวจหลักฐานก่อนยอมรับ `finish_task` (W5[A], ACC-3, W63[7.2] ฯลฯ) | ตั้ง `_MAX_PREMATURE_TRUE_FINISH_RETRIES`, `_MAX_PREMATURE_FALSE_FINISH_RETRIES`, `_MAX_PREMATURE_ALL_FAILED_RETRIES`, `_MAX_PREMATURE_VALIDATION_ERROR_RETRIES`, `_MAX_PREMATURE_DELETION_INCOMPLETE_RETRIES`, `_MAX_DELETE_ALL_UNVERIFIED_RETRIES`, `_MAX_PREMATURE_TABLE_VERIFY_RETRIES`, `_MAX_COUNT_ANSWER_MISMATCH_RETRIES` = 0 และแทน `orchestrator._scan_created_item_in_table` (เป็น async) ด้วย coroutine ที่คืน `True` เสมอ |
| **BASELINE** | ปิดทั้ง 3 กลุ่มข้างบน = "ก่อนปรับปรุง" | รวม patch ของ 3 arm |

ข้อสังเกตเรื่อง NO_VERIFY: ถ้าตั้งแค่ `_MAX_PREMATURE_TABLE_VERIFY_RETRIES = 0` ยังไม่พอ — โค้ดหลัง retry หมด
(`orchestrator.py` ราวบรรทัด 4505) จะบังคับ `claimed_success = False` ทันที ซึ่งยังเป็นการ verify อยู่
จึงต้อง patch `_scan_created_item_in_table` ด้วย

ใช้รายงานได้สองแบบ:

- **ก่อน/หลัง** (ตารางหลักของบท 2.3.4): BASELINE เทียบ FULL
- **Ablation แบบ leave-one-out**: FULL เทียบ NO_CONTEXT / NO_PROMPT_SECTIONS / NO_VERIFY — ส่วนต่างแต่ละแถว = ผลของฟีเจอร์นั้น

## 3. ตัวแปรที่ต้องคุมให้เท่ากันทุก arm

- **commit เดียว** — ตอนนี้มีไฟล์ค้าง (`task_manager.py`, `L2.yaml`, `generate_catalog.py`) ให้ commit ก่อนเริ่ม แล้วบันทึก hash ลงผล
- **provider/model เดียว** — `.env` ในเครื่องนี้ตั้ง `LLM_PROVIDER=gemini` (gemini-flash-lite-latest, free tier
  โควต้าจำกัดตาม optimize.txt) ไม่ใช่ `openai` ตาม default ของ config.py ให้ใส่ `--provider` ให้ชัดทุกครั้ง
  (ค่าที่ใช้จริงถูกบันทึกใน manifest) ถ้าเลือก `openai` = `gpt-5.5` ชื่อเดียวที่ endpoint ChatGPT OAuth รับ
  ทั้งสองตัวตั้งค่าอุณหภูมิผ่านโค้ดนี้ไม่ได้ จึงต้องรันซ้ำ
- `max_steps=20`, headless, `step_pacing_delay_seconds=3.0` (ห้ามลด — ดู A/B ใน `config.py` W41)
- **ปิด `enable_procedural_memory_capture` ทุก arm** — abstractor ยิง LLM เพิ่มหลัง task สำเร็จแบบ background กินโควต้าและอาจทำให้ attempt ถัดไปโดน throttle
- long-term memory ไม่ปนข้าม attempt อยู่แล้ว (`recall()` คืน `[]` เมื่อไม่มี `session_id` และ adapter ไม่ส่ง)
- nav fast-path ไม่ทำงานบน localhost เพราะยังไม่มี site manual ของ `127.0.0.1:8100` — ยืนยันจาก `fastpath_hit_rate = 0`
- **สลับลำดับ**: วนแบบ `for rep in 1..3: for task in tasks: for arm in shuffled(arms)` ไม่รัน arm เดียวต่อกันยาว ๆ
  เพื่อไม่ให้ช่วงเวลา/สถานะโควต้าไปผูกกับ arm ใด arm หนึ่ง และทุก rep เป็นบล็อกที่ครบสมดุล หยุดกลางทางยังใช้ผลได้

## 4. Metric

ต่อ attempt (บันทึกลง `data/ablation/<run_id>.jsonl`): arm, task_id, rep, `passed` (DB), `claimed_success` (agent),
steps, llm_calls, `input` / `cache_read` / `output` tokens, duration, failure detail

ต่อ arm:

1. **Success rate** จาก DB + Wilson 95% CI
2. **False success** = agent บอกสำเร็จแต่ DB บอกไม่ — metric หลักของ NO_VERIFY
3. **Tokens ต่อ task** (median) แยก input / cached / output
4. **Steps และ LLM calls** (median)
5. **เวลา** (median, p95) แยก `timing.llm` / `action` / `pacing` จาก step_trace
6. **สาเหตุความล้มเหลว** — นับ `failure_class` จาก step_trace (element_not_found, timeout, wrong_action_type, …)
7. **Token ต่อ step ของ task ยาว** — เส้นกราฟ FULL เทียบ NO_CONTEXT ใช้ `tokens` ราย step ใน step_trace (แสดงว่า compaction ทำให้เส้นไม่โตตามจำนวน step)

### ข้อควรระวังเรื่องการนับ token

provider `openai` คืน `input_tokens` ที่ **รวม** cached tokens อยู่แล้ว (`llm.py` ราวบรรทัด 2253 ใช้
`input_tokens_details.cached_tokens` เป็นส่วนย่อย) แต่สูตรใน `hrm_local_eval.py:111`, `evaluation.py:200`,
`miniwob_eval.py:174` บวก `input + output + cache_read + cache_creation` — สำหรับ OpenAI จึงนับส่วน cached ซ้ำ
(ตัวอย่างรัน 14 ต.ค.: 133,267 + 951 = 134,218 แต่ `avg_tokens` นับเป็น 167,882)
สูตรนี้ถูกสำหรับ Anthropic ซึ่ง `input_tokens` ไม่รวม cache read สคริปต์วิเคราะห์จึงต้องคำนวณเองจาก token_usage.jsonl:
`total = input + output` และรายงาน `cache_read / input` เป็นสัดส่วน cache แยก

### การตีความ

benchmark นี้ผันผวน (commit เดียวกันเคยได้ 12/15 และ 15/15 — CLAUDE.md) ด้วย 10 task × 3 rep = 30 attempt ต่อ arm
ส่วนต่าง success rate ต่ำกว่าประมาณ 15–20 จุดเปอร์เซ็นต์ให้ถือว่าแยกไม่ออกจาก noise ส่วน tokens/steps เทียบแบบจับคู่
ราย task (task เดียวกันข้าม arm) แล้วรายงาน median ของส่วนต่าง จะน่าเชื่อถือกว่าเทียบค่าเฉลี่ยรวม

## 5. สคริปต์ (เขียนแล้ว)

```bash
python benchmark_target/run_ablation.py --dry-run --reps 1                      # stub, ไม่ใช้ LLM
python benchmark_target/run_ablation.py --provider openai --arms FULL,BASELINE  # รันจริง
python benchmark_target/run_ablation.py --resume data/ablation/<session>.jsonl  # รันต่อช่องที่ขาด
python benchmark_target/analyze_ablation.py data/ablation/<session>.jsonl
```

dry-run 10 task × 5 arm (2026-10-22): ทั้ง 50 ช่องได้ FUNCTIONAL_FAIL — ไม่มี task ไหนผ่านได้โดย agent ไม่ทำอะไร

1. `benchmark_target/run_ablation.py` (~150 บรรทัด) — นิยาม arm เป็น context manager ของ patch,
   วนแบบสลับลำดับ, เรียก `run_attempt()` + `HermesAgentAdapter(run_id=f"abl-{arm}-{ts}")` ใน
   `allow_internal_navigation_here()` + `TestClient(control_app)` แบบเดียวกับ `hrm_local_eval._run_sync()`,
   ใส่ `--arms`, `--tasks`, `--reps`, `--dry-run` (ใช้ `StubAgentAdapter`)
2. `benchmark_target/analyze_ablation.py` (~100 บรรทัด) — อ่านผล + join token_usage/step_trace ด้วย `run_id`,
   ออกตาราง Markdown/CSV และกราฟ PNG สำหรับใส่เล่ม
3. เทสต์สั้น ๆ: arm แต่ละตัว patch ค่าตามที่ประกาศ และคืนค่าเดิมหลังออกจาก context

## 6. ลำดับการทำ (เหลือถึง 31 ต.ค.)

| วัน | งาน | LLM |
|---|---|---|
| 22–23 ต.ค. | commit ไฟล์ค้าง, เขียน 2 สคริปต์, `--dry-run` ด้วย stub (ยืนยันว่า 4 task ใหม่ stub ทำไม่ผ่าน), smoke 1 task × 5 arm × 1 rep | ~5 attempt |
| 24–26 ต.ค. | รันจริง rep ละ 1 รอบต่อช่วง (50 attempt ≈ 1–1.5 ชม. ต่อ rep ไม่รวม backoff) | 150 attempt |
| 27 ต.ค. | วิเคราะห์, ทำตาราง/กราฟ, เขียน 2.3.4 ใหม่ + บทสรุป | — |
| 28–31 ต.ค. | กันเวลาไว้สำหรับ rerun ที่โดน quota/infra | — |

ถ้าโควต้าไม่พอ: ทำ **FULL + BASELINE ก่อน** (60 attempt) ให้ได้ตารางก่อน/หลัง แล้วค่อยเพิ่ม NO_VERIFY,
NO_CONTEXT, NO_PROMPT_SECTIONS ตามลำดับ — คาดว่า NO_CONTEXT/BASELINE จะกิน token ต่อ attempt มากกว่า FULL
ชัดเจน (ยังไม่ได้วัด ให้ดูจาก smoke run ก่อนกำหนดจำนวนจริง)

attempt ที่ล้มเพราะ infra (429, browser พัง, fixture ไม่ขึ้น) ไม่นับเป็นความผิดของ agent (`AttemptResult.is_infra_failure()`)
ให้รันชดเชยเฉพาะช่องนั้นจนได้ครบ 3 sample ที่ใช้ได้

## 7. ตัวเลือกเพิ่มเติม (ถ้าเหลือเวลา): Fast Path

ต้อง learn site ของ HRM ในเครื่องก่อน (`POST /api/site-manual/learn` ที่ `http://127.0.0.1:8100`) แล้วเทียบ
`enable_nav_fastpath_auto_decide` เปิด/ปิด — วัด `fastpath_hit_rate`, LLM calls และเวลา ไม่รวมในแผนหลัก
เพราะ crawl อาจกดปุ่มที่เปลี่ยน DB (fixture reset ช่วยได้ แต่ต้องตรวจก่อน) และ procedural memory replay
(`enable_procedural_memory`) ยังไม่ผ่านการ validate บนเว็บจริงตาม `config.py`

## 8. ผลที่จะได้ใส่เล่ม

- ตาราง: BASELINE เทียบ FULL — success rate (CI), false success, median tokens, steps, เวลา
- ตาราง: ablation — ส่วนต่างของแต่ละ arm เทียบ FULL
- กราฟแท่ง: tokens ต่อ task ราย arm
- กราฟเส้น: tokens ต่อ step ของ task ยาว 1 ตัว (FULL เทียบ NO_CONTEXT)
- กราฟวงกลม: สัดส่วน `failure_class`
- ย่อหน้าวิธีการทดลอง: เว็บจำลองในเครื่อง + reset fixture + ตัดสินจาก DB + สลับลำดับ + รันซ้ำ 3 ครั้ง
