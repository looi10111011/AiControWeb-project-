# Scope / SRS — AI Browser Agent

> Template สำหรับ W1 (หมุดหมาย: "ตกลงเว็บเป้าหมาย 2-3 หน้า + เกณฑ์วัดผล") ตาม `roadmap.txt`
> เติมเนื้อหาจริงร่วมกันระหว่าง [A]/[B]

## 1. เป้าหมายโปรเจกต์

AI ที่ควบคุมหน้าเว็บได้เอง โดย (1) เรียนรู้หน้าเว็บด้วยตัวเอง และ (2) ใช้คู่มือที่ user ป้อนเพิ่ม

## 2. เว็บเป้าหมาย (Target Websites)

เลือก 2-3 หน้าเว็บสำหรับ dev + eval ตลอดโปรเจกต์:

| # | เว็บ | เหตุผลที่เลือก | ความซับซ้อน |
|---|------|----------------|--------------|
| 1 | TBD | | |
| 2 | TBD | | |
| 3 | TBD | | |

## 3. Scope

### 3.1 In-scope
- TBD

### 3.2 Out-of-scope
- TBD

## 4. Functional Requirements

- FR1: Agent ต้องดึง snapshot หน้าเว็บเป็น indexed elements ได้
- FR2: Agent ต้องทำ action พื้นฐานได้ (คลิก/พิมพ์/เลื่อน/dropdown/สลับ tab)
- FR3: Agent ต้องใช้คู่มือ (RAG) ประกอบการวางแผนได้
- FR4: Action ที่มีความเสี่ยงต้องขอยืนยันจาก user ก่อน (human-in-the-loop)
- FR5: TBD

## 5. Non-functional Requirements

- NFR1: Token efficiency — perception ต้อง compact (ไม่ dump ทั้ง DOM)
- NFR2: Safety — ต้องมี allowlist/blocklist ป้องกัน action อันตราย
- NFR3: TBD

## 6. เกณฑ์วัดผล (Evaluation Criteria)

แนว WebVoyager (ดู roadmap.txt เฟส 4):

| Metric | นิยาม | เป้าหมาย |
|--------|-------|----------|
| Success rate | % ของ task ที่ทำสำเร็จ end-to-end | TBD |
| จำนวน step เฉลี่ย | จำนวน action ต่อ task | TBD |
| Token per task | token ที่ใช้ต่อ 1 task | TBD |

## 7. Milestones (อ้างอิง roadmap.txt)

- W1 ★ ตกลงเว็บเป้าหมาย + เกณฑ์วัดผล
- W5 ★ Agent ทำ task end-to-end + มีเบรกความปลอดภัย
- W8 ★ AI อ่านหน้าเว็บ + ใช้คู่มือ + จำได้
- W11 ★ ระบบครบ ใช้ผ่าน UI ได้
- W14 ★ ส่งงาน + นำเสนอ
