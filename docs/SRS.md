# Scope / SRS — AI Browser Agent

> หมุดหมาย W1: "ตกลงเว็บเป้าหมาย 2-3 หน้า + เกณฑ์วัดผล" ตาม `roadmap.txt`
> เติมเนื้อหาจริงแล้ว (2026-07-15) — สรุปย้อนหลังจากสิ่งที่ใช้จริงตลอด W2-W8 (ทุก
> test case/demo ใน `run.py` ใช้เว็บเดียวกันนี้มาตั้งแต่ W2 โดยไม่เคยเปลี่ยน)

## 1. เป้าหมายโปรเจกต์

AI ที่ควบคุมหน้าเว็บได้เอง โดย (1) เรียนรู้หน้าเว็บด้วยตัวเอง และ (2) ใช้คู่มือที่ user ป้อนเพิ่ม

## 2. เว็บเป้าหมาย (Target Websites)

**เว็บหลัก: [saucedemo.com](https://www.saucedemo.com/)** — เว็บ e-commerce demo
สาธารณะ (ไม่ต้องใช้ API key/บัญชีจริง ใช้ login สาธิต `standard_user`/`secret_sauce`
ที่เปิดเผยต่อสาธารณะ) เลือกเพราะ:
- ครอบคลุม UI pattern หลักที่ agent ต้องรับมือครบในเว็บเดียว (form กรอกข้อมูล,
  dropdown, icon-only element ที่ไม่มี text ให้อ่าน, multi-page flow, action ที่
  ควรขอยืนยันจาก human เช่น "Remove"/"Checkout")
- เข้าถึงได้ฟรีไม่จำกัด เหมาะกับการรัน automated test ซ้ำๆ ระหว่าง dev โดยไม่ติด
  rate limit/ToS ของเว็บจริง

เลือก 3 หน้า/flow ภายในเว็บนี้สำหรับ dev + eval ตลอดโปรเจกต์:

| # | หน้า/Flow | เหตุผลที่เลือก | ความซับซ้อน |
|---|-----------|----------------|--------------|
| 1 | Login (`/`) | ทดสอบ form fill (username/password) + risky-action type "submit" บนปุ่ม Login | ต่ำ — 2-3 step |
| 2 | Inventory + Product Detail (`/inventory.html`) | ทดสอบ dropdown (sort), click ปุ่ม icon-only (ตะกร้า), navigate เข้าหน้ารายละเอียดสินค้า | กลาง — 3-6 step |
| 3 | Cart → Checkout (`/cart.html` → `/checkout-step-one.html` → `/checkout-step-two.html` → `/checkout-complete.html`) | ทดสอบ multi-step form, RAG-driven policy (คู่มือกำหนดค่าที่ต้องกรอก + action ที่ต้องขออนุมัติ), human-in-the-loop confirmation | สูง — 10-21 step |

## 3. Scope

### 3.1 In-scope
- Perception แบบ indexed elements (ไม่ dump ทั้ง DOM) — W2[A]
- Browser actions ครบ: click/fill/select/check/scroll/goto/go_back/switch_tab/wait
  — W3[A]/W5[A]
- RAG: ingest คู่มือ (PDF/DOCX/TXT) + query ทุก step ของ planner — W3[B]/W6[B]
- Agent loop (Perceive → Plan → Act → Verify) รองรับ 3 LLM provider
  (Anthropic/Gemini/Groq) — W4[A]
- Permission layer: allowlist/blocklist + human-in-the-loop + RAG-based (คู่มือ
  กำหนดเพิ่มได้) — W4[B]/W7[B]
- Verify + Retry: retry action ที่ fail จาก DOM ยังไม่นิ่ง, guard กัน
  finish_task(true/false) ก่อนเวลาอันควร, คืนหลักฐาน DOM สุดท้ายให้ตรวจสอบได้
  — W5[A]
- Memory: short-term (กันทำซ้ำ/refusal memory ภายใน task เดียว) + long-term
  (pattern ข้าม task run ผ่าน ChromaDB) — W7[A]
- Loop-detection: กันวนซ้ำคาบ 1-4 (action เดิม/สลับ 2-4 action ไม่มีความคืบหน้า)
  — W5[A]/W6[B]/W8

### 3.2 Out-of-scope (ของ 8 สัปดาห์แรก — ดู roadmap.txt เฟส 3-4 สำหรับของที่เหลือ)
- Vision fallback (screenshot-based, W9) — nice-to-have ตัดได้ถ้าเวลาไม่พอ
- Multi-website generalization นอกเหนือ saucedemo.com (W9[B] เก็บ edge case จากเว็บ
  อื่นทีหลัง)
- API server + Test Console UI (W10) — ตอนนี้ยังเป็น CLI (`run.py`) ล้วนๆ
- Automated success/fail scoring แบบ goal-agnostic เต็มรูปแบบ (W5[A] ให้แค่
  หลักฐาน DOM สุดท้ายไว้ตรวจสอบ ไม่ได้ auto-judge ว่า "สำเร็จจริงไหม" เพราะเกณฑ์
  สำเร็จผูกกับ goal แต่ละอันไม่เหมือนกัน)

## 4. Functional Requirements

- FR1: Agent ต้องดึง snapshot หน้าเว็บเป็น indexed elements ได้ — ✅ (W2[A])
- FR2: Agent ต้องทำ action พื้นฐานได้ (คลิก/พิมพ์/เลื่อน/dropdown/สลับ tab) —
  ✅ (W3[A]/W5[A], ครบทั้ง 9 action type ใน `backend/app/core/actions.py`)
- FR3: Agent ต้องใช้คู่มือ (RAG) ประกอบการวางแผนได้ — ✅ (W6[B])
- FR4: Action ที่มีความเสี่ยงต้องขอยืนยันจาก user ก่อน (human-in-the-loop) —
  ✅ (W4[B]/W7[B], ทั้ง hardcoded type/label และ RAG-based จากคู่มือ)
- FR5: Agent ต้องจำ pattern ที่เคยล้มเหลว/ถูกปฏิเสธได้ทั้งภายใน 1 task run
  (short-term) และข้าม task run (long-term) — ✅ (W7[A])

## 5. Non-functional Requirements

- NFR1: Token efficiency — perception ต้อง compact (ไม่ dump ทั้ง DOM), token
  ต้องไม่โตแบบไม่มีเพดานตามจำนวน step (context compaction, Gemini) — ✅ (W2[A]/W7[A])
- NFR2: Safety — ต้องมี allowlist/blocklist ป้องกัน action อันตราย —
  ✅ (W4[B]/W7[B])
- NFR3: Robustness — action ที่ fail ต้อง retry อัตโนมัติก่อนส่งกลับ LLM (DOM
  ยังไม่นิ่ง), agent ต้องไม่วนซ้ำไม่มีที่สิ้นสุด (loop-detection) — ✅ (W5[A])

## 6. เกณฑ์วัดผล (Evaluation Criteria)

แนว WebVoyager (ดู roadmap.txt เฟส 4, วัดจริงเป็นงานของ W12[B]) — เป้าหมายตั้งไว้จาก
ข้อมูลจริงที่สังเกตได้ตลอด W4-W8 (ผ่าน provider Gemini เป็นหลัก):

| Metric | นิยาม | เป้าหมาย |
|--------|-------|----------|
| Success rate | % ของ task ที่ทำสำเร็จ end-to-end บน flow ในข้อ 2 | ≥ 80% |
| จำนวน step เฉลี่ย | จำนวน action ต่อ task (จาก `result["steps"]`) | Login เดี่ยว ≤ 6 step, เต็ม flow (login→cart→checkout) ≤ 20 step |
| Token per task | token รวมต่อ 1 task (จาก `result["tokens"]`) | ไม่โตเป็นเส้นตรงไม่มีเพดานตามจำนวน step (compaction ทำงาน) |

หลักฐานประกอบการตัดสิน success/fail: `result["final_page_state"]` (page state
จริงจาก DOM ตอนจบ task — เพิ่มใน W5[A]) เทียบกับ `result["message"]` ที่ LLM
อ้างเอง ไม่ใช่เชื่อคำเคลมของ LLM ลอยๆ อย่างเดียว

## 7. Milestones (อ้างอิง roadmap.txt)

- W1 ★ ตกลงเว็บเป้าหมาย + เกณฑ์วัดผล — ✅ (เอกสารนี้)
- W5 ★ Agent ทำ task end-to-end + มีเบรกความปลอดภัย — ✅
- W8 ★ AI อ่านหน้าเว็บ + ใช้คู่มือ + จำได้ — ✅ (รอบแรก ดู roadmap.txt)
- W11 ★ ระบบครบ ใช้ผ่าน UI ได้ — ยังไม่เริ่ม (รอ W10 API server + Test Console)
- W14 ★ ส่งงาน + นำเสนอ — ยังไม่เริ่ม
