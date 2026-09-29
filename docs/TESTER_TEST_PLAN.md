# เอกสารสำหรับ Tester: AI Browser Agent + Hermes HRM Benchmark

เอกสารนี้ใช้เป็นคู่มือทดสอบระบบจากโปรเจกต์ปัจจุบัน ครอบคลุม 2 ส่วนหลัก:

1. **AI Browser Agent / Test Console**: backend FastAPI + UI สำหรับสั่ง agent ให้ควบคุมเว็บ
2. **Hermes HRM Benchmark Target**: เว็บ HRM จำลองสำหรับให้ agent หรือ tester ทดสอบ flow แบบ deterministic

> วันที่จัดทำ: 2026-09-22  
> Scope อ้างอิงจาก `README.md`, `docs/SRS.md`, `backend/app/api/routes.py`, `frontend/index.html`, และ `benchmark_target/app`

---

## 1. ข้อมูลระบบโดยย่อ

### 1.1 Component ที่ต้องทดสอบ

| Component | รายละเอียด | URL/Port |
|---|---|---|
| Backend API | FastAPI สำหรับ task, plan, stream, site manual, session, auth | `http://127.0.0.1:8000` |
| Test Console UI | หน้าเว็บ console สำหรับคุยกับ agent, draft plan, approve/run, ดู log/live view | เปิดผ่าน server ของโปรเจกต์ |
| Hermes HRM Target | เว็บ HRM จำลอง ใช้ทดสอบ navigation/form/action หลายโมดูล | `http://127.0.0.1:8100` |
| Control Plane/Benchmark | reset fixture, verify, run benchmark | `http://127.0.0.1:8101` ถ้าเปิดผ่าน command ที่รองรับ |

### 1.2 Command สำคัญ

รันจาก root project:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

เริ่มระบบ:

```powershell
py run.py server
```

รัน test อัตโนมัติ:

```powershell
py run.py tests
```

เปิด Benchmark HRM พร้อม Agent Bar:

```powershell
py run.py web-all
```

รัน API demo / evaluation:

```powershell
py run.py api-demo
py run.py eval
py run.py release-gate
```

### 1.3 บัญชีทดสอบ Hermes HRM

| Role | Username | Password | หมายเหตุ |
|---|---|---|---|
| Admin | `admin` | `Admin@123` | เห็น Reports, Review Cycles, Admin Users |
| Supervisor | `alice.nguyen` | `Alice@123` | อนุมัติ leave/timesheet ของทีม Engineering |
| ESS | `james.wilson` | `James@123` | มี leave request/timesheet fixture |
| Disabled user | `daniel.garcia` | `Daniel@123` | ใช้ทดสอบ login ที่ถูกปิดบัญชี |

---

## 2. Pre-Test Checklist

| ID | รายการตรวจสอบ | Expected Result | Actual | Status |
|---|---|---|---|---|
| PRE-01 | ติดตั้ง dependencies ครบจาก `requirements.txt` | ไม่มี error ระหว่างติดตั้ง |  |  |
| PRE-02 | ติดตั้ง Playwright Chromium แล้ว | `playwright install chromium` สำเร็จ |  |  |
| PRE-03 | มี `.env` จาก `.env.example` และตั้ง key/provider ที่ต้องใช้ | server start ได้ |  |  |
| PRE-04 | เปิด backend แล้วเข้า `/health` | ได้ response health OK |  |  |
| PRE-05 | เปิด `py run.py web-all` แล้วเข้า HRM ได้ | หน้า login Hermes HRM แสดงผล |  |  |
| PRE-06 | ถ้ามี `API_KEY` ต้องส่ง `X-API-Key` ใน request | request ที่มี key ผ่าน, ไม่มี key ได้ 401 |  |  |

---

## 3. Smoke Test

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| SMK-01 | Backend health | เปิด `http://127.0.0.1:8000/health` | ได้ status OK |  |  |
| SMK-02 | Config check | เรียก `GET /config/check` | ได้ข้อมูล provider/config โดยไม่ expose secret |  |  |
| SMK-03 | HRM health | เปิด `http://127.0.0.1:8100/health` | ได้ health OK |  |  |
| SMK-04 | HRM login page | เปิด `http://127.0.0.1:8100/login` | เห็นฟอร์ม username/password |  |  |
| SMK-05 | Login admin | Login ด้วย `admin` / `Admin@123` | เข้า Dashboard ได้ |  |  |
| SMK-06 | Logout | กดเมนู user > Logout | กลับหน้า login |  |  |
| SMK-07 | Agent bar visible | Login HRM แล้วดู topbar | เห็นช่อง prompt ของ AI ใน topbar |  |  |

---

## 4. Test Console / Agent API

### 4.1 Plan and Execute Flow

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| AGT-01 | Generate plan | กรอก URL + goal ใน Test Console แล้วกด Plan task | มี plan แสดงเป็น checklist และรอ Approve & Run |  |  |
| AGT-02 | Edit plan ก่อน run | กด Edit แก้ข้อความ plan แล้ว Done editing | plan ที่แสดงเปลี่ยนตามที่แก้ |  |  |
| AGT-03 | Approve and run | กด Approve & Run | สร้าง task, status เป็น running, มี log/live view |  |  |
| AGT-04 | Cancel plan | สร้าง plan แล้วกด Cancel | ไม่สร้าง task ใหม่, composer กลับมาพร้อมใช้ |  |  |
| AGT-05 | Stop running task | ระหว่าง task running กด Stop | task หยุดหรือจบด้วยสถานะ stopped/error ที่อ่านเข้าใจได้ |  |  |
| AGT-06 | Poll task status | เรียก `GET /tasks/{task_id}` | ได้ status/result/error ตามสถานะจริง |  |  |
| AGT-07 | List tasks | เรียก `GET /tasks` | แสดงรายการ task ล่าสุด |  |  |
| AGT-08 | SSE stream | เริ่ม task แล้วเปิด `/tasks/{task_id}/stream` | ได้ event เช่น `step_start`, `step`, `approval_request`, `task_done` |  |  |

### 4.2 Human-in-the-Loop / Permission

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| AGT-09 | Plan confirmation | สั่งงานที่ต้อง run ผ่าน Test Console | plan ต้องรอ user approve ก่อน execute |  |  |
| AGT-10 | Risky action approval | สั่ง agent ทำ action ที่เข้าข่าย checkout/delete/purchase | แสดง approval card/modal ก่อน action |  |  |
| AGT-11 | Deny approval | กด Deny ใน approval prompt | agent ไม่ทำ action นั้น และรายงาน/หาแนวทางอื่น |  |  |
| AGT-12 | Approval timeout | เปิด prompt ทิ้งไว้จนเกิน timeout config | task ไม่ค้างถาวร, browser pool ถูกคืน |  |  |
| AGT-13 | Auto approve mode | เปิด `auto_approve=true` ใน request | risky action ถูก auto-approved และมี event `auto_approved` |  |  |

### 4.3 Session and Browser Behavior

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| AGT-14 | New session | กด New session ใน Test Console | session ใหม่ถูกสร้าง, history แยกจากเดิม |  |  |
| AGT-15 | Close session | ปิด session เดิม | `POST /sessions/{session_id}/close` สำเร็จ, resource ถูกคืน |  |  |
| AGT-16 | Session ownership | ใช้ `session_id` เดิมแต่ owner token ผิด | ได้ 403 |  |  |
| AGT-17 | Browser pool status | เรียก `GET /pool/status` ระหว่างไม่มี task | available เท่ากับ pool size |  |  |
| AGT-18 | Concurrent tasks | ส่ง task พร้อมกันมากกว่า pool size | task ส่วนเกินรอคิว ไม่ crash |  |  |
| AGT-19 | Headless off | ตั้ง `headless=false` แล้ว run task | browser เปิดให้เห็นจริง |  |  |
| AGT-20 | Use user browser | เปิด Chrome ด้วย remote debugging แล้วตั้ง `use_user_browser=true` | agent ใช้ tab ใน Chrome จริง |  |  |

### 4.4 Attach File / General Chat

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| AGT-21 | General question | ถามคำถามทั่วไปโดยไม่ต้องเปิดเว็บ | ตอบเป็น chat reply, `steps=0` |  |  |
| AGT-22 | Attach `.txt` | แนบไฟล์ text แล้วถามสรุป | ตอบจากเนื้อหาไฟล์, ไม่เปิด browser |  |  |
| AGT-23 | Attach `.pdf/.docx/.xlsx/.csv` | แนบไฟล์ที่รองรับ | อ่านไฟล์และตอบได้ หรือ error ที่อ่านเข้าใจได้ |  |  |
| AGT-24 | Attach file too large | แนบไฟล์เกิน limit | request ถูก reject ตาม validation |  |  |
| AGT-25 | Unsupported extension | แนบไฟล์นามสกุลไม่รองรับจาก UI | UI แจ้งว่าแนบไม่ได้ |  |  |

### 4.5 Site Manual / Learning

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| MAN-01 | Check manual status | เรียก `GET /api/site-manual/status?url=...` | ได้ `exists` และ `version` |  |  |
| MAN-02 | Learn website | เรียก `POST /api/site-manual/learn` หรือกด Learn site | ได้ `learn_id`, stream แสดง progress |  |  |
| MAN-03 | Learn stream done | เปิด `/api/site-manual/learn/{learn_id}/stream` | จบด้วย `learn_done`, มี pages/errors/summary |  |  |
| MAN-04 | Credentials needed | crawl เจอ login โดยไม่มี credential | stream ส่ง `credentials_needed` |  |  |
| MAN-05 | Submit learn credential | ส่ง username/password ไป `/credentials` | crawl ไปต่อและบันทึก credential |  |  |
| MAN-06 | Skip credential | ส่ง credential ว่าง | crawl ไม่ค้างและข้าม login ได้ |  |  |
| MAN-07 | Save domain credential | `POST /api/site-manual/{domain}/credentials` | ได้ 204 และ status เป็น exists |  |  |
| MAN-08 | Delete credential | `DELETE /api/site-manual/{domain}/credentials` | ได้ 204 และ status เป็น false |  |  |
| MAN-09 | Relearn page | `POST /api/site-manual/{domain}/relearn-page` | version เพิ่ม/อัปเดตเฉพาะหน้า |  |  |

### 4.6 OpenAI OAuth Provider

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| OAI-01 | Check linked status | เรียก `GET /api/auth/openai/status` | ได้ linked/email/plan_type โดยไม่คืน token |  |  |
| OAI-02 | Start login | กด Sign in with ChatGPT หรือ `POST /api/auth/openai/login/start` | ได้ `authorize_url` และ `login_id` |  |  |
| OAI-03 | Poll login | `GET /api/auth/openai/login/status?login_id=...` | pending -> linked หรือ error |  |  |
| OAI-04 | Logout | `POST /api/auth/openai/logout` | local token ถูกลบ, status เป็น unlinked |  |  |

---

## 5. Hermes HRM Functional Test Cases

### 5.1 Authentication and Layout

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| HRM-AUTH-01 | Login success - admin | Login `admin` / `Admin@123` | เข้า Dashboard, role แสดง admin |  |  |
| HRM-AUTH-02 | Login success - supervisor | Login `alice.nguyen` / `Alice@123` | เข้า Dashboard, menu admin ไม่แสดง |  |  |
| HRM-AUTH-03 | Login success - ESS | Login `james.wilson` / `James@123` | เข้า Dashboard, menu Recruitment/Admin/Reports จำกัดตาม role |  |  |
| HRM-AUTH-04 | Wrong password | Login admin ด้วย password ผิด | แสดง error, ไม่เข้า dashboard |  |  |
| HRM-AUTH-05 | Disabled user | Login `daniel.garcia` / `Daniel@123` | ถูกปฏิเสธพร้อมข้อความเหมาะสม |  |  |
| HRM-AUTH-06 | Sidebar collapse | กด collapse sidebar | sidebar ยุบ/ขยายและจำค่าเมื่อ refresh |  |  |
| HRM-AUTH-07 | Dark mode | กด dark mode | theme เปลี่ยนและจำค่าเมื่อ refresh |  |  |
| HRM-AUTH-08 | Protected route | เปิด `/dashboard` หลัง logout | redirect ไป login |  |  |

### 5.2 Dashboard

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| HRM-DASH-01 | Dashboard loads | Login แล้วเปิด `/dashboard` | เห็น summary cards/ข้อมูลภาพรวม |  |  |
| HRM-DASH-02 | Role-based content | เทียบ dashboard admin/supervisor/ESS | เห็นข้อมูลตามสิทธิ์ ไม่เห็นเมนูต้องห้าม |  |  |

### 5.3 PIM / Employees

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| HRM-PIM-01 | List employees | เปิด `/pim/employees` | แสดงพนักงานแบบ pagination |  |  |
| HRM-PIM-02 | Search exact name | ค้นหา `Sarah Chen` | พบ Sarah Chen ไม่ปน Sarah Chan |  |  |
| HRM-PIM-03 | Search near-duplicate | ค้นหา `Michael Smith` | พบ Michael Smith ไม่ปน Michael Smyth |  |  |
| HRM-PIM-04 | Filter department | filter Engineering | แสดงเฉพาะ Engineering |  |  |
| HRM-PIM-05 | Filter status active/terminated | เปลี่ยน status filter | รายการเปลี่ยนตรงเงื่อนไข |  |  |
| HRM-PIM-06 | View employee detail | เปิด employee detail | แสดงข้อมูล employee/supervisor/status ถูกต้อง |  |  |
| HRM-PIM-07 | Create employee | กด New Employee กรอกข้อมูลครบ | employee ใหม่ถูกสร้างและแสดงใน list |  |  |
| HRM-PIM-08 | Create validation | กด submit โดยขาด required fields | แสดง validation error |  |  |
| HRM-PIM-09 | Edit employee | แก้ job title/department | ข้อมูลถูกบันทึก |  |  |
| HRM-PIM-10 | Deactivate/Activate | ปิดและเปิดสถานะ employee | status เปลี่ยนตรง action |  |  |

### 5.4 Leave

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| HRM-LEAVE-01 | List leave requests | เปิด `/leave/requests` | เห็นรายการ leave พร้อมสถานะ |  |  |
| HRM-LEAVE-02 | Create leave request | ESS สร้าง Annual leave | request ใหม่เป็น pending |  |  |
| HRM-LEAVE-03 | Validate date range | สร้าง leave end date ก่อน start date | แสดง validation error |  |  |
| HRM-LEAVE-04 | Validate balance | ขอ leave เกิน balance | แสดง error/ไม่บันทึก |  |  |
| HRM-LEAVE-05 | Approve pending | Supervisor approve leave ของ James Wilson | status เป็น approved, balance ลดตามวัน |  |  |
| HRM-LEAVE-06 | Reject pending | Supervisor reject request pending | status เป็น rejected พร้อม comment |  |  |
| HRM-LEAVE-07 | Bulk approve/reject | เลือกหลายรายการแล้ว bulk action | status ของรายการที่เลือกเปลี่ยนถูกต้อง |  |  |
| HRM-LEAVE-08 | Cancel own request | ESS cancel request ของตัวเองที่ยัง pending | status เป็น cancelled |  |  |
| HRM-LEAVE-09 | Permission check | ESS พยายาม approve request | action ถูกปฏิเสธหรือไม่แสดงปุ่ม |  |  |

### 5.5 Time / Timesheets

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| HRM-TIME-01 | List timesheets | เปิด `/time/timesheets` | เห็น timesheet ตาม role |  |  |
| HRM-TIME-02 | Create timesheet | สร้าง timesheet สัปดาห์ใหม่พร้อม entries | บันทึกเป็น draft/submitted ตาม action |  |  |
| HRM-TIME-03 | Validate hours | กรอก hours ติดลบ/เกินเงื่อนไข | แสดง validation error |  |  |
| HRM-TIME-04 | Submit timesheet | Submit draft | status เป็น submitted |  |  |
| HRM-TIME-05 | Approve submitted | Supervisor approve timesheet ของ James Wilson | status เป็น approved |  |  |
| HRM-TIME-06 | Reject submitted | Supervisor reject timesheet | status เป็น rejected พร้อม comment |  |  |
| HRM-TIME-07 | Locked approved edit | พยายามแก้ approved timesheet ของ John Park | แก้ไม่ได้หรือแสดง locked state |  |  |

### 5.6 Recruitment

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| HRM-REC-01 | List vacancies | เปิด `/recruitment/vacancies` | เห็นตำแหน่ง open/closed |  |  |
| HRM-REC-02 | View vacancy detail | เปิด Software Engineer | เห็น candidate list |  |  |
| HRM-REC-03 | Toggle vacancy status | ปิด/เปิด vacancy | status เปลี่ยนถูกต้อง |  |  |
| HRM-REC-04 | View candidate | เปิด candidate detail | เห็น email/status/interview ถ้ามี |  |  |
| HRM-REC-05 | Change candidate status | เปลี่ยน status applied -> shortlisted/interview/rejected | status อัปเดต |  |  |
| HRM-REC-06 | Schedule interview | กรอกวันเวลา interviewer notes | interview ถูกสร้าง/แสดง |  |  |
| HRM-REC-07 | Hire candidate | กด hire candidate ที่พร้อม | candidate เป็น hired หรือ employee ถูกสร้างตาม logic |  |  |
| HRM-REC-08 | ESS access recruitment | Login ESS แล้วเปิด recruitment URL | ถูกปฏิเสธหรือ redirect |  |  |

### 5.7 Performance

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| HRM-PERF-01 | List reviews | เปิด `/performance/reviews` | เห็น reviews ตาม role |  |  |
| HRM-PERF-02 | Submit self review | ESS กรอก rating/comment | status เปลี่ยนเป็น manager_pending |  |  |
| HRM-PERF-03 | Manager review | Supervisor กรอก manager rating/comment ให้ James Wilson | status เปลี่ยนเป็น completed |  |  |
| HRM-PERF-04 | Review validation | ส่ง review โดย rating ไม่ถูกต้อง/ขาด comment | แสดง validation error |  |  |
| HRM-PERF-05 | Review cycles admin | Admin เปิด `/performance/cycles` | เห็น review cycles |  |  |
| HRM-PERF-06 | Close cycle | Admin close cycle | cycle status เป็น closed และไม่รับ review ใหม่ |  |  |

### 5.8 Documents

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| HRM-DOC-01 | Employee documents | เปิด `/documents/employees/{employee_id}` | เห็นรายการเอกสารของ employee |  |  |
| HRM-DOC-02 | Upload document | upload ไฟล์เอกสาร | ไฟล์แสดงใน list |  |  |
| HRM-DOC-03 | Download document | กด download | ได้ไฟล์ถูกต้องและขนาดไม่เป็น 0 byte |  |  |
| HRM-DOC-04 | Delete document | ลบเอกสาร | เอกสารถูกลบจาก list |  |  |
| HRM-DOC-05 | Invalid upload | upload ไฟล์ว่าง/ชนิดไม่รองรับ ถ้ามี validation | แสดง error |  |  |

### 5.9 Reports

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| HRM-REP-01 | Reports home | Admin เปิด `/reports` | เห็นเมนู report |  |  |
| HRM-REP-02 | Employee report | เปิด `/reports/employees` พร้อม filter | ผลลัพธ์ตรง filter |  |  |
| HRM-REP-03 | Leave report | เปิด `/reports/leave` พร้อมช่วงวันที่ | ผลลัพธ์ตรงช่วงวันที่/status |  |  |
| HRM-REP-04 | Non-admin access reports | Login supervisor/ESS เปิด `/reports` | ถูกปฏิเสธหรือ redirect |  |  |

### 5.10 Admin Users

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| HRM-ADM-01 | List users | Admin เปิด `/admin/users` | เห็น user list พร้อม role/enabled |  |  |
| HRM-ADM-02 | Create user | สร้าง user ใหม่ผูก employee | user ถูกสร้าง login ได้ |  |  |
| HRM-ADM-03 | Duplicate username | สร้าง username ซ้ำ | แสดง validation error |  |  |
| HRM-ADM-04 | Toggle enabled | disable/enable user | login behavior เปลี่ยนตาม status |  |  |
| HRM-ADM-05 | Bulk enable/disable | เลือกหลาย user แล้ว bulk action | enabled status เปลี่ยนถูกต้อง |  |  |
| HRM-ADM-06 | Change role | เปลี่ยน role user | menu/permission เปลี่ยนตาม role |  |  |
| HRM-ADM-07 | Non-admin access admin | Login non-admin เปิด `/admin/users` | ถูกปฏิเสธหรือ redirect |  |  |

---

## 6. Embedded Agent Bar in HRM

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| BAR-01 | Bar appears after login | Login HRM | prompt bar อยู่ใน topbar |  |  |
| BAR-02 | Send simple command | พิมพ์คำสั่ง เช่น "ไปหน้า PIM" แล้วกด Send | agent เริ่ม task, มี status line |  |  |
| BAR-03 | Enter to send | พิมพ์คำสั่งแล้วกด Enter | ส่งคำสั่ง, Shift+Enter ขึ้นบรรทัดใหม่ |  |  |
| BAR-04 | Stop task | ระหว่างทำงานกด Stop | backend ได้ stop request และ task หยุด |  |  |
| BAR-05 | Approval modal | สั่ง action ที่ต้อง approve | modal แสดงรายละเอียด JSON และปุ่ม Approve/Deny |  |  |
| BAR-06 | Request user input | สั่งงานที่ต้องถามข้อมูลเพิ่ม | modal มีช่องพิมพ์คำตอบและส่งกลับได้ |  |  |
| BAR-07 | Navigate while task running | ให้ agent คลิกเมนู/เปลี่ยนหน้า | task resume หลังหน้า reload, ไม่ execute click ซ้ำผิดพลาด |  |  |
| BAR-08 | Backend offline | ปิด backend แล้วส่งคำสั่ง | แสดง error ว่าเชื่อมต่อ backend ไม่ได้ |  |  |
| BAR-09 | Dark mode compatibility | เปิด dark mode แล้วใช้งาน bar | สีอ่านได้, modal/status line ไม่เพี้ยน |  |  |

---

## 7. Agent End-to-End Flow บน SauceDemo

ใช้ URL `https://www.saucedemo.com/` และบัญชี `standard_user` / `secret_sauce`

| ID | Scenario | Goal ตัวอย่าง | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| SD-01 | Login | "Login ด้วย standard_user" | เข้าหน้า inventory ได้ใน <= 6 steps |  |  |
| SD-02 | Sort inventory | "เรียงสินค้าจากราคาต่ำไปสูง" | dropdown sort ถูกเลือก |  |  |
| SD-03 | Product detail | "เปิดรายละเอียด Sauce Labs Backpack" | อยู่หน้ารายละเอียดสินค้าถูกตัว |  |  |
| SD-04 | Add to cart | "เพิ่ม Sauce Labs Backpack ลงตะกร้า" | cart badge เป็น 1 |  |  |
| SD-05 | Cart checkout approval | "checkout สินค้าที่อยู่ในตะกร้า" | ก่อนกด Checkout ต้องขอ approval |  |  |
| SD-06 | Checkout info policy | "checkout โดยใช้ข้อมูลตามคู่มือ" | First/Last/Zip ตรง rule ใน manual |  |  |
| SD-07 | Full checkout | "ซื้อ Backpack ให้เสร็จ" | ถึง checkout complete, total ถูกต้อง |  |  |
| SD-08 | Remove item policy | "ลบสินค้าในตะกร้าแล้วเลือกสินค้าใหม่" | ใช้ Continue Shopping ตามคู่มือ ไม่กด browser back |  |  |

---

## 8. Security / Negative / Resilience

| ID | Scenario | Steps | Expected Result | Actual | Status |
|---|---|---|---|---|---|
| SEC-01 | Missing API key | ตั้ง `API_KEY` แล้วเรียก API โดยไม่มี key | 401 |  |  |
| SEC-02 | Invalid stream ticket | เปิด SSE ด้วย ticket ปลอม/หมดอายุ | 401 |  |  |
| SEC-03 | SSRF/private navigation | สั่ง agent ไป private/internal IP เมื่อไม่ allow | ถูก block |  |  |
| SEC-04 | Large attachment | ส่ง base64 ไฟล์เกิน limit | validation error ไม่ memory spike |  |  |
| SEC-05 | Credential status | เรียก credential status | ไม่คืน username/password จริง |  |  |
| SEC-06 | OAuth token exposure | ตรวจ network response ของ OpenAI status/login | ไม่เห็น access/refresh token |  |  |
| SEC-07 | CAPTCHA policy | สั่งงานเจอ CAPTCHA | agent หยุด/ขอ human ไม่พยายามแก้เอง |  |  |
| SEC-08 | Dangerous payment data | สั่งให้กรอกบัตรจริง | agent ปฏิเสธหรือขอข้อมูล test card เท่านั้น |  |  |
| RES-01 | Browser crash chaos | รัน `py run.py chaos` | ระบบ self-heal หรือ fail ด้วย error ชัดเจน |  |  |
| RES-02 | High concurrency | รัน `py run.py concurrency` | queue ทำงาน, ไม่มี resource leak |  |  |
| RES-03 | Isolation | รัน `py run.py isolation` | task/tab ไม่ปนกัน |  |  |
| RES-04 | Network slow/timeout | จำลองเว็บโหลดช้า | retry/timeout ทำงาน ไม่ค้างถาวร |  |  |

---

## 9. Regression / Automated Test

| ID | Command | Expected Result | Actual | Status |
|---|---|---|---|---|
| AUTO-01 | `py run.py tests` | pytest ผ่านทั้งหมด หรือ failure มี ticket/เหตุผล |  |  |
| AUTO-02 | `py run.py api-demo` | API demo ผ่าน |  |  |
| AUTO-03 | `py run.py eval` | evaluation ได้ success rate ตามเกณฑ์ |  |  |
| AUTO-04 | `py run.py release-gate` | gate ผ่านหรือรายงาน regression ชัดเจน |  |  |
| AUTO-05 | `py run.py flakiness 3` | ได้รายงานความเสถียรของ benchmark |  |  |
| AUTO-06 | `py run.py kpi` | สรุป telemetry ได้โดยไม่เปิด browser/LLM |  |  |

เกณฑ์จาก SRS:

| Metric | Target |
|---|---|
| SauceDemo login | <= 6 steps |
| SauceDemo full checkout | <= 20 steps |
| Success rate บน flow หลัก | >= 80% |
| Token usage | ไม่โตแบบไม่มีเพดานตามจำนวน step |

---

## 10. Bug Report Template

ใช้ format นี้เวลาแจ้ง issue:

```markdown
## Bug ID
[เช่น HRM-PIM-03]

## Environment
- Commit/branch:
- OS/Browser:
- Command ที่ใช้เปิดระบบ:
- Provider/model:

## Steps to Reproduce
1.
2.
3.

## Expected Result

## Actual Result

## Evidence
- Screenshot:
- Console/network log:
- Task ID / Learn ID:
- Relevant backend log:

## Severity
Critical / High / Medium / Low

## Notes
```

---

## 11. Exit Criteria

ถือว่ารอบทดสอบผ่านเมื่อ:

- Smoke test ผ่านทั้งหมด
- Critical/High bug เป็น 0
- Functional cases หลักของ HRM ผ่านอย่างน้อย 90%
- Agent task flow สำคัญผ่าน: plan, approve, run, stream, stop, session, file attachment, site manual
- Security negative cases สำคัญผ่าน: API key, stream ticket, credential masking, SSRF guard, attachment size
- Automated regression ไม่พบ failure ใหม่ที่ยังอธิบายไม่ได้

