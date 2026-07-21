"""
run.py — จุดรันเดียวของโปรเจกต์ (รวม server / test / demo ไว้ในไฟล์นี้ไฟล์เดียว)

วิธีใช้:
    python run.py            # เปิดเมนูให้เลือก
    python run.py server     # รัน API server (uvicorn --reload)
    python run.py test       # รัน pytest ทั้งหมด
    python run.py perception # รัน demo perception.py (login saucedemo.com)
    python run.py ingest [path]  # ingest คู่มือเข้า ChromaDB (default: manual_test.txt)
    python run.py query      # ถาม query แล้ว retrieve() ค้นคู่มือที่ ingest ไว้
    python run.py agent      # รัน Orchestrator.run_task() จริงบน saucedemo.com
                              # เปิดหน้าต่าง browser จริงให้เห็น + print log ทีละ
                              # step ลง terminal คู่กัน (ต้องมี API key จริงของ
                              # provider ที่ตั้งไว้ใน .env — LLM_PROVIDER=anthropic/
                              # gemini/groq — หรือ override ท้าย argument ก็ได้ เช่น
                              # `python run.py agent "goal" gemini`) — โชว์แผนก่อน
                              # แล้วรอกด y/n ยืนยันก่อนค่อยเริ่ม loop จริง
                              # (confirm_plan=True) + ถ้าเจอ action เสี่ยง (submit/
                              # delete/purchase/pay หรือ goto โดเมนที่บล็อกไว้)
                              # จะหยุดถามยืนยันอีกรอบ
    python run.py permission  # ทดสอบ permission layer (classify_action + execute)
                              # ตรงๆ ไม่ต้องเปิด browser/ยิง LLM API จริง — โชว์ผล
                              # classify_action() ของ action หลายแบบ (safe/blocked/
                              # needs_confirmation) แล้วลอง execute() จริงเคสที่
                              # ต้องขอยืนยัน (จะเจอ prompt y/n ให้ลองตอบเอง) — action
                              # ประเภท submit/delete/purchase/pay ไม่ได้อยู่ใน schema
                              # ที่ LLM เรียกได้จริง เลยยังไม่มีทางเจอ path นี้ผ่าน
                              # agent loop ปกติ ต้องทดสอบตรงๆ แบบนี้แทน

    W7[A] — Memory demo (long-term memory + context compaction), รันจริงบน
    saucedemo.com ด้วย Orchestrator.run_task() (ไม่ mock) ต้องมี API key จริงใน .env:
    python run.py memory-a    # Test Case A: จำ pattern ความผิดพลาด — รัน task เดิม 2
                              # รอบ (กด "Remove" ในตะกร้า ซึ่งโดน auto-reject จำลอง
                              # ว่า action นี้ "พัง" เสมอ) รอบ 2 ควรได้เห็น long-term
                              # memory ของรอบ 1 ถูกดึงมาใช้ (recall() print ให้ดูตรงๆ
                              # ก่อนเริ่มรอบ 2)
    python run.py memory-b    # Test Case B: จดจำข้อมูลสำคัญข้ามรอบ — task 1 หาราคา
                              # สินค้าแล้วรายงานใน finish_task message (ถูกบันทึกเข้า
                              # long-term memory อัตโนมัติ) task 2 กรอกฟอร์ม checkout
                              # โดยบอกแค่ "ใช้ค่าจาก task ก่อนหน้า" ให้ agent ดึงมาเอง
    python run.py memory-c    # Test Case C: token/context compaction (Gemini เท่านั้น
                              # — ดู _GEMINI_COMPACT_AFTER_STEPS ใน orchestrator.py) —
                              # รัน task ยาวหลาย step บังคับ provider=gemini ดู log
                              # [tokens] ต่อ step ว่าไม่โตไม่หยุดตามจำนวน step + log
                              # [gemini-compact] ตอนบีบอัดทำงานจริง

    W7[B] — RAG-based permission demo, รันจริงบน saucedemo.com (ไม่ mock) ต้องมี
    API key จริงใน .env และต้อง ingest manual_test.txt ไว้แล้ว (มี RULE-04 ที่บอกว่า
    การกด Checkout ต้องขออนุมัติก่อนเสมอ — python run.py ingest):
    python run.py permission-rag  # Test Case D: ให้ agent login -> add to cart ->
                              # ไปหน้าตะกร้า -> กด Checkout (type="click" ธรรมดา ไม่ใช่
                              # submit/purchase ที่ hardcode ไว้ และ label "Checkout" ก็
                              # ไม่ตรง RISKY_LABEL_KEYWORDS เลย) — ต้องเห็น log
                              # [HUMAN-IN-THE-LOOP] ขึ้นก่อนกด Checkout ทั้งที่ hardcode
                              # rules เดิมจะปล่อยผ่านเฉยๆ พิสูจน์ว่าคู่มือ (RAG) เป็นคน
                              # สั่งขออนุมัติ ไม่ใช่ hardcoded rule

    W8 — บูรณาการรอบแรก (Perception + คู่มือ RAG + ความจำ = ครบ 3 สมอง), รันจริงบน
    saucedemo.com (ไม่ mock) ต้องมี API key จริงใน .env และต้อง ingest manual_test.txt
    ไว้แล้ว (มี RULE-03 กำหนดค่า First Name/Last Name/Zip Code ที่ต้องใช้ตอน checkout):
    python run.py integration  # Goal บังคับให้ agent ต้องอ่านหน้าเว็บทุก step
                              # (perception) + ดึงค่า First Name/Last Name/Zip Code
                              # จากคู่มือ RULE-03 แทนการเดาเอง (RAG) — รัน task เดิม
                              # 2 รอบเพื่อดูว่า long-term memory (recall()) ทำงาน
                              # ร่วมกับอีก 2 สมองได้จริงไหม (steps รอบ 2 ควรน้อยกว่า/
                              # เท่าเดิม ไม่ใช่มากกว่า) เช็คอัตโนมัติด้วยว่า history
                              # ของแต่ละรอบมีค่าจากคู่มือครบทั้ง 3 ค่าจริง (ไม่ใช่แค่
                              # agent "อ้างว่า" ทำตามคู่มือใน finish_task message เฉยๆ)

    W9[A] — Vision fallback + handle error states (popup), รันจริงบน saucedemo.com
    (ไม่ mock) ต้องมี API key จริงใน .env (provider=gemini เท่านั้น — vision fallback
    scope แค่ Gemini ตอนนี้):
    python run.py vision-fallback  # ฉีด overlay ปลอม (จำลอง cookie-consent banner)
                              # บังปุ่ม "Add to cart" ของสินค้าชิ้นแรกจริงๆ ก่อนให้ agent
                              # เริ่มทำงาน — คลิกจริงจะ fail (Playwright ตรวจจับว่า
                              # element โดน overlay บังจริง ไม่ใช่ mock) แล้ว vision
                              # fallback ต้องทำงาน: ถ่าย screenshot จริงส่งให้ Gemini
                              # อธิบาย ป้อนกลับเข้า step ถัดไป ดู log [vision-fallback]
                              # ว่าโมเดลอธิบายเห็น cookie banner จริงไหม แล้ว agent หา
                              # ทางไปต่อได้เอง (เช่น ปิด banner ก่อน) ไหม

    W10[A] — API endpoints (FastAPI) + Browser Pool (persistent): `python run.py server`
    (action "1") ตอนนี้เสิร์ฟ endpoint ใหม่ด้วย (ดู backend/app/api/routes.py) นอกเหนือ
    จาก /health, /config/check เดิม:
        POST /tasks              body: {"url", "goal", "max_steps"?, "provider"?,
                                  "headless"?, "auto_approve"?} คืน 202 + {"task_id",
                                  "status":"running"} ทันที ไม่รอ task จบ (รันเป็น
                                  background ผ่าน core/api/task_manager.py — เพราะ
                                  Orchestrator.run_task() ใช้เวลาเป็นนาที) — auto_approve
                                  default false = action ที่ต้องขออนุมัติ (permission
                                  layer) จะถูก "ปฏิเสธ" อัตโนมัติเสมอ (fail closed เพราะ
                                  ยังไม่มี human-in-the-loop จริงผ่าน REST — ส่ง true มา
                                  เองถ้ายอมรับความเสี่ยงนี้)
        GET  /tasks/{task_id}    poll สถานะ: "running"/"done"/"error" + result/error
        GET  /tasks              list task ทั้งหมด (ใหม่สุดก่อน)
        GET  /pool/status        {"size","available","in_use"} ของ BrowserPool
    python run.py api-demo    # (alias: w10) เปิด uvicorn subprocess จริง ยิง HTTP
                              # request จริงเข้า endpoint ข้างบนทั้งหมด (ไม่ mock) รวม
                              # POST /tasks จริงบน saucedemo.com ผ่าน provider=gemini
                              # ต้องมี GEMINI_API_KEY จริงใน .env ดู log [pool] เทียบ
                              # available ก่อน/ระหว่าง/หลัง task เพื่อพิสูจน์ว่า browser
                              # ถูกยืมแล้วคืน pool จริง ไม่ได้เปิด/ปิดใหม่ทุกรอบเหมือน
                              # W1-W9 — ท้ายสุดมี log [report] สรุปลำดับ action ของ task
                              # แบบอ่านง่าย (แค่ step/action/OK-FAIL) แยกจาก [result] ที่
                              # เป็น raw dict เต็มด้านบน (ไว้ debug ละเอียด)

    W11 — Integration hardening: เปิด uvicorn subprocess จริงของตัวเอง (port แยกจาก
    settings.api_port กันชนกับ server ที่อาจเปิดอยู่แล้ว) + ยิง HTTP จริง (httpx) เข้า
    endpoint ของ W10[A] ไม่ mock สักจุด ต้องมี API key จริงใน .env (ทุกอันบังคับ
    auto_approve=True + confirm_plan=False กัน task ค้างรอ human ตอบระหว่างที่ test
    กำลังพยายามวัดจังหวะ concurrency/chaos พอดี):
    python run.py chaos       # (alias: w11a) W11[A] Chaos Test — ยิง task จริงแล้วฆ่า
                              # process chrome.exe ของ BrowserPool จริงกลางคัน (หา PID
                              # ผ่าน psutil เดินต้นไม้ descendant ของ uvicorn subprocess
                              # เท่านั้น กันพลาดไปฆ่า Chrome จริงของเครื่อง/ของ server อื่น
                              # ที่เปิดอยู่) ดูว่า server ยัง /health ปกติไหม (ไม่ล่มทั้งตัว
                              # ตาม browser) + task เดิมจบด้วย status="error" อย่างสุภาพไหม
                              # + ยิง task ใหม่หลังจากนั้นเช็คว่า pool "self-heal" จริงไหม
                              # (คำตอบที่คาดไว้ตอนเขียน: ไม่ self-heal — BrowserPool ยังไม่มี
                              # health check ตอน acquire() เลย เป็นข้อจำกัดจริง ไม่ใช่การเดา)
    python run.py concurrency # (alias: w11b) W11[B] High-Concurrency Test — ยิง 5 task
                              # พร้อมกันจริงเข้า pool ขนาด 2 (browser_pool_size default)
                              # เช็คว่า in_use ไม่มีวันเกิน pool size, เห็น task ส่วนเกิน
                              # รอคิวจริง (ไม่ crash/deadlock) แล้วสุดท้ายจบครบทั้ง 5
    python run.py isolation   # (alias: w11c) W11[C] Isolation Test — รัน 2 task พร้อมกัน
                              # จริงบน pool เดียวกัน ด้วย credential ที่ตั้งใจให้ผลต่างกัน
                              # ชัดเจน (task A login ถูกต้อง, task B login ผิดรหัสตั้งใจ)
                              # เช็คผลจริงว่า BrowserContext ของแต่ละ task แยกกันจริงไหม
                              # (ไม่แชร์ cookie/session ข้าม task แม้ใช้ browser ตัวเดียวกัน
                              # จาก pool) — a ต้อง login สำเร็จไม่มี error ของ b ปนมา, b ต้อง
                              # เห็น error message จริงของตัวเอง ไม่ได้แอบ "สำเร็จ" เพราะ
                              # ดันไปสวม session ที่ a login ไว้ก่อน

    real-browser — เชื่อม agent เข้ากับ Chrome จริงที่ user เปิดใช้งานอยู่ (มี cookie/
    login ค้างอยู่จริง เช่น mail) แทนที่จะเปิด Chromium ว่างๆ เองเหมือนคำสั่งอื่นทั้งหมด
    ด้านบน (ดู backend/app/core/user_browser.py):
    python run.py real-browser  # (alias: user-browser) ก่อนรันต้อง:
                              # 1) ปิด Chrome ทุกหน้าต่าง/process ให้หมดก่อนจริงๆ (เช็ค
                              #    Task Manager ด้วยถ้าจำเป็น — ถ้ามี Chrome instance อื่น
                              #    เปิดค้างอยู่บน profile เดียวกัน Chrome จะเงียบๆ ไม่เปิด
                              #    debug port ให้เลย ส่ง flag ไปที่ instance เดิมแทน เป็น
                              #    สาเหตุ fail ที่พบบ่อยที่สุด)
                              # 2) เปิด Chrome ใหม่ด้วย flag "chrome.exe"
                              #    --remote-debugging-port=9222 (ห้ามใส่ --user-data-dir
                              #    เพื่อใช้ profile จริงเดิมที่ login ไว้อยู่แล้ว)
                              # 3) เช็คว่า debug port ทำงานจริง: เปิด
                              #    http://localhost:9222/json/version ต้องได้ JSON กลับมา
                              # ไม่ใส่ argument จะถาม goal + target URL ทาง terminal (ต่าง
                              # จาก python run.py agent ที่ hardcode saucedemo.com เพราะ
                              # โหมดนี้มีไว้ใช้กับเว็บจริงที่ user login เท่านั้น) หรือใส่
                              # ตรงๆ ก็ได้: python run.py real-browser "goal" "https://..."
                              # [provider] — agent จะต่อเข้า browser จริงผ่าน CDP (ใช้
                              # BrowserContext เดิมที่มี cookie จริง ไม่ใช่ context ว่าง
                              # เปล่าใหม่), จำกัด goto/navigation ให้อยู่แค่โดเมนของ
                              # target URL เท่านั้น (default-deny โดเมนอื่นทั้งหมดแม้ session
                              # จะ login ค้างไว้ก็ตาม — กัน agent หลุดไปแตะ mail/บัญชีอื่น
                              # โดยไม่ตั้งใจ) และถ้ามี tab ที่ user เปิดค้างไว้ตรงกับโดเมน
                              # เป้าหมายอยู่แล้ว จะถามยืนยันก่อนเข้าไปใช้เสมอ (ตั้งค่า
                              # เริ่มต้นผ่าน .env: USER_BROWSER_TAB_REUSE_POLICY =
                              # ask/always_new_tab/always_reuse) — ไม่มีทางปิด Chrome ของ
                              # user เองเด็ดขาด ปิดแค่ tab ที่ agent เปิดเองเท่านั้นตอนจบ
                              # task (ดู core/user_browser.py::resolve_target_page +
                              # orchestrator.py::run_task ส่วน connect_to_user_browser)

    W12[B] — Evaluation แนว WebVoyager (success rate / จำนวน step / token ต่อ task):
    python run.py eval        # (alias: evaluation, w12) รัน BENCHMARK_TASKS (ดู
                              # core/evaluation.py — goal เดิมเป๊ะจาก demo อื่นๆ ในไฟล์นี้
                              # ครอบคลุม 3 ระดับความยาว: สั้น/กลาง/ยาว) ทีละตัวตามลำดับผ่าน
                              # Orchestrator.run_task() ตรงๆ บน saucedemo.com จริง
                              # (headless, auto_approve, confirm_plan=False, provider จาก
                              # .env — ไม่มีคนเฝ้าหน้าจอตอบ approve/confirm) แล้วพิมพ์
                              # รายงานสรุป: ผลรายภารกิจ (สำเร็จ/ไม่, step, token) + สรุปรวม
                              # (success rate, avg steps, avg tokens) — ไม่ใช่ pytest
                              # (ต้องใช้ browser+LLM key+เวลาเป็นนาที เหมือน demo อื่นๆ ที่
                              # ต้องยิงจริง ไม่ mock)
"""

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

# แก้ปัญหา UnicodeEncodeError เวลา print ข้อความไทยบน Windows console (cp1252)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def run_server():
    print("=== เริ่ม API server (uvicorn) — ctrl+C เพื่อหยุด ===", flush=True)
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--reload"],
        check=False,
    )


def run_tests():
    print("=== รัน pytest ===", flush=True)
    subprocess.run([sys.executable, "-m", "pytest"], check=False)


def run_perception_demo():
    print("=== รัน perception.py demo (saucedemo.com) ===", flush=True)
    from backend.app.core.perception import demo

    asyncio.run(demo())


def run_ingest():
    print("=== Ingest คู่มือเข้า ChromaDB ===", flush=True)
    from backend.app.rag.ingestion import ingest_manual

    default_manual = Path("backend/tests/manual_test.txt")
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_manual
    if not path.exists():
        print(f"ไม่พบไฟล์: {path}")
        sys.exit(1)

    ingest_manual(path)


def run_query():
    print("=== ค้นคู่มือด้วย retrieve() (ctrl+C เพื่อออก) ===", flush=True)
    from backend.app.rag.retriever import retrieve

    try:
        while True:
            q = input("\nQuery: ").strip()
            if not q:
                continue
            results = retrieve(q)
            if not results:
                print(" (ไม่พบผลลัพธ์)")
                continue
            for i, chunk in enumerate(results):
                print(f" [{i}] {chunk}")
    except (KeyboardInterrupt, EOFError):
        print("\nออกจากโหมด query")




_DEFAULT_AGENT_GOAL = "Log in, add first product, change item to second product , and proceed to checkout"

# ป้องกัน infinite spawn: process ลูกที่ถูกเปิดในหน้าต่าง console ใหม่จะมี env
# ตัวนี้ติดมาด้วย เลยรู้ตัวว่าเป็นลูกแล้ว ไม่ต้องเปิดหน้าต่างใหม่ซ้อนอีกที
_AGENT_WORKER_ENV = "AI_AGENT_WORKER"


def run_agent():
    # agent loop verbose print ทุก step (goto/plan/action/result) ยาวมากเวลาโมเดลวน
    # หลาย step กว่าจะจบ goal เดียว รกหน้าจอ terminal หลักที่อาจกำลังใช้งานอย่างอื่น
    # อยู่ด้วย — เปิด console หน้าต่างใหม่แยกไว้โชว์ log ของ agent รอบนี้โดยเฉพาะแทน
    # (บน Windows เท่านั้น เพราะ CREATE_NEW_CONSOLE เป็น flag เฉพาะ Windows)
    if sys.platform == "win32" and os.environ.get(_AGENT_WORKER_ENV) != "1":
        # ส่ง goal/provider ต่อไปให้ process ลูกถ้าผู้ใช้ใส่มาทาง CLI อยู่แล้ว
        # (python run.py agent "goal" gemini) ถ้าไม่ใส่มา (เช่นเลือกจากเมนู) ก็ปล่อย
        # ให้ process ลูกถาม goal เอาเองในหน้าต่างใหม่ตามปกติ
        child_argv = [sys.executable, __file__, "agent"] + sys.argv[2:]
        child_env = {**os.environ, _AGENT_WORKER_ENV: "1"}
        # ห่อด้วย "cmd /k" กัน console ปิดตัวเองทันทีที่ python process จบ (auto-close
        # เดิมทำให้อ่าน log ไม่ทันถ้า agent จบเร็ว/error เร็ว) — /k แปลว่ารัน command
        # แล้ว "ค้าง" shell ไว้ ต้องปิดหน้าต่างเอง (หรือพิมพ์ exit) หลังอ่าน log เสร็จ
        cmd_str = subprocess.list2cmdline(child_argv)
        subprocess.Popen(
            ["cmd", "/k", cmd_str],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            env=child_env,
        )
        print("=== เปิด log ของ Agent Loop ในหน้าต่างใหม่แล้ว ===")
        print("(agent จบแล้ว หน้าต่างจะไม่ปิดเอง — ปิดเองหรือพิมพ์ exit ได้เลย)")
        return

    _run_agent_inline()


def _run_agent_inline():
    print("=== รัน Agent Loop (Orchestrator.run_task) บน saucedemo.com ===", flush=True)
    from backend.app.config import settings
    from backend.app.core.orchestrator import Orchestrator

    # python run.py agent "goal" groq   <- ยัง override provider ผ่าน CLI arg ได้ถ้าต้องการ
    # แต่ปกติไม่ต้องใส่อะไรเลย ดึง LLM_PROVIDER จาก .env อัตโนมัติ (settings.llm_provider)
    if len(sys.argv) > 2:
        goal = sys.argv[2]
    else:
        goal = input(f"Goal [{_DEFAULT_AGENT_GOAL}]: ").strip() or _DEFAULT_AGENT_GOAL

    provider = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Goal: {goal}", flush=True)
    print(f"Provider: {provider or settings.llm_provider}", flush=True)

    async def _run():
        result = await Orchestrator().run_task(
            "https://www.saucedemo.com/", goal,
            headless=False, verbose=True, provider=provider, confirm_plan=True,
        )
        print("\n=== ผลลัพธ์ ===")
        print(f"plan   : {result['plan']}")
        print(f"success: {result['success']}")
        print(f"steps  : {result['steps']}")
        print(f"message: {result['message']}")
        tokens = result["tokens"]
        total = tokens["input"] + tokens["output"] + tokens["cache_read"] + tokens["cache_creation"]
        print(
            f"tokens : input={tokens['input']} output={tokens['output']}"
            f" cache_read={tokens['cache_read']} cache_write={tokens['cache_creation']} total={total}"
        )
        print("history:")
        for h in result["history"]:
            print(" ", h)

    asyncio.run(_run())


_DEFAULT_REAL_BROWSER_URL = "https://mail.google.com/"
_DEFAULT_REAL_BROWSER_GOAL = "หา email ล่าสุดใน inbox แล้วสรุปหัวข้อ/ผู้ส่งให้ฟัง"


def run_agent_real_browser():
    # เหมือน run_agent() เป๊ะ — เปิด console หน้าต่างใหม่แยกโชว์ log กันรกหน้าจอหลัก
    # (แค่ argv[1] ที่ส่งต่อให้ process ลูกเป็น "real-browser" แทน "agent")
    if sys.platform == "win32" and os.environ.get(_AGENT_WORKER_ENV) != "1":
        child_argv = [sys.executable, __file__, "real-browser"] + sys.argv[2:]
        child_env = {**os.environ, _AGENT_WORKER_ENV: "1"}
        cmd_str = subprocess.list2cmdline(child_argv)
        subprocess.Popen(
            ["cmd", "/k", cmd_str],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            env=child_env,
        )
        print("=== เปิด log ของ Agent Loop (real browser) ในหน้าต่างใหม่แล้ว ===")
        print("(agent จบแล้ว หน้าต่างจะไม่ปิดเอง — ปิดเองหรือพิมพ์ exit ได้เลย)")
        return

    _run_agent_real_browser_inline()


def _run_agent_real_browser_inline():
    print("=== รัน Agent Loop บน Chrome จริงของ user (CDP connect) ===", flush=True)
    print("*** ก่อนรันต้องปิด Chrome ทุกหน้าต่าง/process ให้หมดก่อน แล้วเปิดใหม่ด้วย ***")
    print('*** "chrome.exe" --remote-debugging-port=9222  (ห้ามใส่ --user-data-dir  ***')
    print("*** เพื่อใช้ profile จริงเดิมที่ login mail ไว้อยู่แล้ว) — เช็คว่าเปิดสำเร็จ  ***")
    print("*** ได้ที่ http://localhost:9222/json/version (ต้องได้ JSON กลับมา)        ***\n")
    from backend.app.config import settings
    from backend.app.core.orchestrator import Orchestrator

    if len(sys.argv) > 2:
        goal = sys.argv[2]
    else:
        goal = input(f"Goal [{_DEFAULT_REAL_BROWSER_GOAL}]: ").strip() or _DEFAULT_REAL_BROWSER_GOAL

    if len(sys.argv) > 3:
        url = sys.argv[3]
    else:
        url = input(f"Target URL [{_DEFAULT_REAL_BROWSER_URL}]: ").strip() or _DEFAULT_REAL_BROWSER_URL

    provider = sys.argv[4] if len(sys.argv) > 4 else None

    print(f"Goal: {goal}", flush=True)
    print(f"URL: {url}", flush=True)
    print(f"Provider: {provider or settings.llm_provider}", flush=True)
    print(
        f"Domain ที่จะถูกจำกัด (allowed_domains): auto-derive จาก URL ด้านบน — agent "
        "จะไปโดเมนอื่นไม่ได้เลยแม้ session จะ login ค้างอยู่ก็ตาม",
        flush=True,
    )

    async def _run():
        result = await Orchestrator().run_task(
            url, goal,
            headless=False, verbose=True, provider=provider, confirm_plan=True,
            connect_to_user_browser=True,
        )
        print("\n=== ผลลัพธ์ ===")
        print(f"plan   : {result['plan']}")
        print(f"success: {result['success']}")
        print(f"steps  : {result['steps']}")
        print(f"message: {result['message']}")
        print("history:")
        for h in result["history"]:
            print(" ", h)

    asyncio.run(_run())


class _FakePage:
    """เพจปลอม (ไม่ใช่ Playwright จริง) ไว้ demo permission layer อย่างเดียว — จำลอง
    click()/fill() ให้ "สำเร็จ" เสมอ (ไม่ throw) พิสูจน์ว่า flow permission check ->
    dispatch จริงทำงานถูกทั้งระบบ โดยไม่ต้องเปิด browser จริง"""

    async def click(self, selector, timeout=5000):
        pass

    async def fill(self, selector, text, timeout=5000):
        pass


_PERMISSION_DEMO_CMDS = [
    ("SAFE — click ปกติ", {"type": "click", "index": 0}),
    ("BLOCKED — goto โดเมนที่บล็อกไว้", {"type": "goto", "url": "https://malicious.com/login"}),
    ("NEEDS_CONFIRMATION — submit (index 3)", {"type": "submit", "index": 3}),
    ("NEEDS_CONFIRMATION — purchase (index 7)", {"type": "purchase", "index": 7}),
]


def run_permission_demo():
    print("=== ทดสอบ Permission Layer (classify_action + execute) ===", flush=True)
    print("ใช้ fake page (ไม่เปิด browser จริง ไม่ยิง LLM API จริง) — พิสูจน์ flow", flush=True)
    print("permission check -> dispatch จริงทำงานถูกทั้งระบบ\n", flush=True)
    from backend.app.core.actions import execute
    from backend.app.permission.rules import classify_action

    print("--- 1) classify_action(cmd) อย่างเดียว ---")
    for label, cmd in _PERMISSION_DEMO_CMDS:
        risk = classify_action(cmd)
        print(f"  [{label}]")
        print(f"    cmd={cmd} -> {risk.value}")

    print("\n--- 2) execute(page, cmd) จริง (fake page) ---")
    print("    NEEDS_CONFIRMATION จะขึ้น prompt y/n จริง ให้ลองตอบเอง — กด y แล้วต้อง")
    print("    เห็น [OK] จริง (submit/purchase เป็น alias ของ click ที่ต้องขอยืนยันก่อน)\n")

    async def _run():
        page = _FakePage()
        for label, cmd in _PERMISSION_DEMO_CMDS:
            print(f"[{label}]")
            result = await execute(page, cmd)
            print(f"  -> {result}\n")

    asyncio.run(_run())


# --- W7[A] Memory demos: รัน Orchestrator.run_task() จริงบน saucedemo.com (ไม่ mock) ---
# ต้องมี API key จริงใน .env — เปิด browser ให้เห็น (headless=False) + verbose=True
# เหมือน run_agent() แต่ไม่ spawn console ใหม่ (demo พวกนี้มีหลาย task/หลาย print
# แทรกระหว่างกลาง เปิด console แยกจะทำ re-entry logic ซับซ้อนเกินความจำเป็น)

_TEST_CASE_A_GOAL = (
    "Log in as standard_user/secret_sauce, add the first product to the cart, "
    "go to the cart page, and remove that item from the cart"
)


async def _auto_reject_confirmation(cmd: dict) -> bool:
    """จำลอง action ที่ "พัง"/โดนบล็อกเสมอ — ปฏิเสธทุก action ที่ permission layer
    ถามยืนยัน (NEEDS_CONFIRMATION) แบบ deterministic ไม่ต้องพึ่งพฤติกรรม LLM ตัดสินใจ
    เอง (ปุ่ม "Remove" ใน goal ด้านบนโดน RISKY_LABEL_KEYWORDS จับอยู่แล้วตั้งแต่ W5)"""
    print(f"  [permission] auto-reject action ที่ต้องขอยืนยัน (demo Test Case A): {cmd}", flush=True)
    return False


def run_test_case_a():
    print("=== Test Case A: จำ Pattern ความผิดพลาด (long-term memory) ===", flush=True)
    print("Task รอบที่ 1: ให้ agent เอาสินค้าออกจากตะกร้า (ปุ่ม 'Remove') แต่ ask_user_func", flush=True)
    print("จำลอง auto-reject ทุกครั้ง (เหมือนปุ่มนี้ 'พัง'/โดนบล็อกเสมอ) -> บันทึกเข้า", flush=True)
    print("long-term memory ตอนจบ task 1\n", flush=True)
    from backend.app.core import long_term_memory
    from backend.app.core.orchestrator import Orchestrator

    async def _run():
        result1 = await Orchestrator().run_task(
            "https://www.saucedemo.com/", _TEST_CASE_A_GOAL,
            headless=False, verbose=True, ask_user_func=_auto_reject_confirmation,
        )
        print("\n=== Task รอบที่ 1 จบ ===")
        print(f"success: {result1['success']}  message: {result1['message']}\n")

        print("=== recall() ก่อนเริ่ม Task รอบที่ 2 (ควรเห็นร่องรอยของรอบที่ 1) ===")
        recalled = long_term_memory.recall(query=_TEST_CASE_A_GOAL, k=3)
        if recalled:
            for r in recalled:
                print(f"  - {r}")
        else:
            print("  (recall() ว่างเปล่า — เช็คว่า record_task() ของรอบ 1 เขียนสำเร็จไหม)")

        print("\n=== Task รอบที่ 2: สั่งงานเดิมซ้ำ (ask_user_func เดิม auto-reject เหมือนกัน) ===\n")
        result2 = await Orchestrator().run_task(
            "https://www.saucedemo.com/", _TEST_CASE_A_GOAL,
            headless=False, verbose=True, ask_user_func=_auto_reject_confirmation,
        )
        print("\n=== Task รอบที่ 2 จบ ===")
        print(f"success: {result2['success']}  message: {result2['message']}")
        print(
            "\n*** เทียบ log 2 รอบด้านบน: รอบที่ 2 ควรลองกด 'Remove' น้อยครั้งกว่า/เปลี่ยนวิธี"
            "\nเร็วกว่ารอบแรก ถ้า long-term memory ช่วยจริง — พฤติกรรมโมเดลเป็น stochastic"
            "\nไม่การันตี 100% (ดูหมายเหตุเดียวกันกับ guard อื่นๆ ในโปรเจกต์นี้) ***"
        )

    asyncio.run(_run())


_TEST_CASE_B_TASK1_GOAL = (
    "Log in as standard_user/secret_sauce, then click directly on the 'Sauce Labs Backpack' "
    "product title or image to open its Product Detail page (do not scroll around the "
    "inventory list looking for the price — the detail page shows it clearly), read the "
    "exact price shown there, and report it in your final message (e.g. $29.99)"
)
# 2026-07-14: เดิม goal เขียนว่า "go to checkout" เฉยๆ — agent สับสนระหว่างหน้า Cart
# (มีแค่ปุ่ม Checkout) กับหน้า checkout information (มีฟอร์ม First/Last/Zip จริง) เพราะ
# ทั้งคู่เกี่ยวกับ "checkout" ในความหมายกว้างๆ — เขียนใหม่ให้ระบุลำดับหน้า/ตำแหน่งฟอร์ม
# ชัดเจนขึ้น แยก 2 หน้าออกจากกันตรงๆ ในข้อความเลย
_TEST_CASE_B_TASK2_GOAL = (
    "Log in as standard_user/secret_sauce, add 'Sauce Labs Backpack' to the cart. "
    "Click the shopping cart icon to open the Cart page, then click the 'Checkout' button "
    "on that page to go to the checkout information page (a different page from the Cart "
    "page — it has First Name, Last Name, and Zip Code input fields). Once you see those "
    "3 input fields in the indexed elements, you are already on the right page — do not "
    "navigate there again. Fill First Name with 'Test', fill Last Name with 'User', and "
    "fill Zip Code with the price value you learned from the previous task (numbers only, "
    "no dollar sign), then click Continue"
)

# ต้องมีคำว่า "$" ตามด้วยตัวเลขอย่างน้อย 1 หลัก ถึงจะถือว่า task รอบ 1 หาราคาเจอจริง —
# กัน finish_task(success=true) ที่ claim เฉยๆ โดยไม่มีหลักฐานราคาจริงในข้อความ
_PRICE_PATTERN = re.compile(r"\$\s?\d")


def run_test_case_b():
    print("=== Test Case B: จดจำข้อมูลสำคัญข้ามรอบ (long-term memory) ===", flush=True)
    print("Task รอบที่ 1: ให้ agent หาราคาสินค้าแล้วรายงานใน finish_task message", flush=True)
    print("(ถูกบันทึกเข้า long-term memory อัตโนมัติทุก task — ดู record_task())\n", flush=True)
    from backend.app.core.orchestrator import Orchestrator

    async def _run():
        result1 = await Orchestrator().run_task(
            "https://www.saucedemo.com/", _TEST_CASE_B_TASK1_GOAL, headless=False, verbose=True,
        )
        print("\n=== Task รอบที่ 1 จบ (ค่าที่ควรถูกจำไว้) ===")
        print(f"success: {result1['success']}  message: {result1['message']}\n")

        # 2026-07-14: user ขอ — ถ้ารอบ 1 ทำภารกิจไม่สำเร็จ หรือสำเร็จแต่ไม่มีราคาจริงอยู่
        # ในข้อความเลย (LLM claim success=true ลอยๆ โดยไม่มีหลักฐาน) ห้ามรันรอบ 2 ต่อ —
        # รอบ 2 พึ่งข้อมูลจากรอบ 1 ผ่าน long-term memory โดยตรง ถ้ารอบ 1 ไม่มีราคาให้จำ
        # รอบ 2 ก็ไม่มีทางผ่านได้อยู่แล้ว รันต่อไปมีแต่เสีย API quota เปล่าๆ
        price_found = bool(_PRICE_PATTERN.search(result1["message"] or ""))
        if not result1["success"] or not price_found:
            print("*** Task รอบที่ 1 ไม่ผ่านเกณฑ์ — หยุดตรงนี้ ไม่รันรอบที่ 2 ต่อ ***")
            print(f"    success={result1['success']}  price_found_in_message={price_found}")
            print("    แก้ไข Task รอบที่ 1 ให้ผ่านก่อน (เช่น ปรับ _TEST_CASE_B_TASK1_GOAL")
            print("    หรือ SYSTEM_PROMPT ใน llm.py ให้ agent หาราคาเจอแน่นอนขึ้น) แล้วค่อยลองใหม่")
            return

        print("=== Task รอบที่ 2: บอกแค่ \"ใช้ข้อมูลจาก task ก่อนหน้า\" ไม่บอกราคาตรงๆ ===\n")
        result2 = await Orchestrator().run_task(
            "https://www.saucedemo.com/", _TEST_CASE_B_TASK2_GOAL, headless=False, verbose=True,
        )
        print("\n=== Task รอบที่ 2 จบ ===")
        print(f"success: {result2['success']}  message: {result2['message']}")
        print(
            "\n*** เช็คจาก log [step ...] ด้านบนว่า action fill ที่ช่อง Zip Code ใช้ค่าราคา"
            "\nจาก task รอบที่ 1 จริงไหม (ไม่ใช่ค่าที่เดาขึ้นมาเอง) ***"
        )

    asyncio.run(_run())


_TEST_CASE_C_GOAL = (
    "Log in as standard_user/secret_sauce, sort products by name Z to A, then sort back "
    "to name A to Z, add the first three products to the cart one at a time, go to the "
    "cart page, remove one item, then proceed to checkout, fill First Name with 'Test', "
    "Last Name with 'User', Zip Code with '10110', click Continue, then click Finish"
)


def run_test_case_c():
    from backend.app.core.orchestrator import Orchestrator, _GEMINI_COMPACT_AFTER_STEPS

    print("=== Test Case C: Token/Context compaction (Gemini เท่านั้น) ===", flush=True)
    print("บังคับ provider=gemini (compaction scope แค่ตัวนี้ — ดู orchestrator.py)", flush=True)
    print(f"เกิน {_GEMINI_COMPACT_AFTER_STEPS} step แล้วดู log [tokens] ต่อ step ว่า", flush=True)
    print("input token ไม่โตไม่หยุดตามจำนวน step อีกต่อไป + log [gemini-compact]\n", flush=True)

    async def _run():
        result = await Orchestrator().run_task(
            "https://www.saucedemo.com/", _TEST_CASE_C_GOAL,
            headless=False, verbose=True, provider="gemini", max_steps=25,
        )
        print("\n=== ผลลัพธ์ ===")
        print(f"success: {result['success']}  steps: {result['steps']}")
        print(f"message: {result['message']}")
        tokens = result["tokens"]
        print(f"tokens : input={tokens['input']} output={tokens['output']}")

    asyncio.run(_run())


# --- W7[B]: RAG-based permission demo (คู่มือกำหนด action ที่ต้องขออนุมัติ) ---
# รัน Orchestrator.run_task() จริงบน saucedemo.com (ไม่ mock) เหมือน Test Case A/B/C
# ของ W7[A] ด้านบน — ต่างกันตรงที่เป้าหมายของเทสนี้คือพิสูจน์ว่า classify_action()
# ยก risk เป็น NEEDS_CONFIRMATION ได้จาก "เนื้อหาคู่มือ" ล้วนๆ ไม่ใช่จาก hardcoded
# type (submit/delete/purchase/pay) หรือ RISKY_LABEL_KEYWORDS เลย — ปุ่ม "Checkout"
# ของ saucedemo เป็นแค่ type="click" ธรรมดา label "Checkout" (ไม่ตรงคำเสี่ยงไหนใน
# RISKY_LABEL_KEYWORDS) ถ้าไม่มี RULE-04 ในคู่มือ (ดู manual_test.txt) action นี้จะ
# เป็น SAFE เฉยๆ ไม่ถาม human เลย
_TEST_CASE_D_GOAL = (
    "Log in as standard_user/secret_sauce, add the first product to the cart, "
    "click the shopping cart icon to open the Cart page, then click the 'Checkout' button"
)


async def _log_and_approve_confirmation(cmd: dict) -> bool:
    """log ให้เห็นชัดๆ ว่า permission layer มาถามจริง (พิสูจน์ classify_action() ยก
    risk จากคู่มือ) แล้วอนุมัติผ่านไปเฉยๆ (ไม่ต้อง block เทสด้วย input() จริง)"""
    print(
        f"  [permission] (W7[B] RAG-based) ask_user_func ถูกเรียกสำหรับ: {cmd} "
        "— ถ้าไม่มี RULE-04 ในคู่มือ action นี้จะเป็น SAFE เฉยๆ ไม่มาถึงจุดนี้เลย",
        flush=True,
    )
    return True


def run_test_case_d():
    print("=== Test Case D: RAG-based permission (คู่มือกำหนด action ที่ต้องขออนุมัติ) ===", flush=True)
    print("ต้อง ingest manual_test.txt ไว้ก่อน (มี RULE-04: Checkout ต้องขออนุมัติ) —", flush=True)
    print("ถ้ายังไม่เคย ingest ให้รัน `python run.py ingest` ก่อนเทสนี้\n", flush=True)
    from backend.app.core.orchestrator import Orchestrator

    async def _run():
        result = await Orchestrator().run_task(
            "https://www.saucedemo.com/", _TEST_CASE_D_GOAL,
            headless=False, verbose=True, ask_user_func=_log_and_approve_confirmation,
        )
        print("\n=== ผลลัพธ์ ===")
        print(f"success: {result['success']}  steps: {result['steps']}")
        print(f"message: {result['message']}")
        print(
            "\n*** เช็คว่า log ด้านบนมีบรรทัด [permission] (W7[B] RAG-based) ก่อนบรรทัด"
            "\n[step ...] ที่กด Checkout ไหม — ถ้ามี แปลว่าคู่มือ (ไม่ใช่ hardcoded rule)"
            "\nเป็นคนสั่งขออนุมัติจริง ***"
        )

    asyncio.run(_run())


# --- W8: บูรณาการรอบแรก (Perception + คู่มือ RAG + ความจำ = ครบ 3 สมอง) ---
# ต่างจาก Test Case A-D ที่แต่ละอันแยกทดสอบ "สมอง" เดียว — เดโมนี้ตั้งใจให้ 1 goal
# เดียวต้องพึ่งทั้ง 3 อย่างพร้อมกัน: (1) perception เพราะต้องเดินหลายหน้า (login ->
# cart -> checkout info) อ่าน indexed elements ใหม่ทุก step (2) RAG manual เพราะ
# goal ไม่บอกค่า First Name/Last Name/Zip Code ตรงๆ บังคับให้ต้องดึงจาก RULE-03 ใน
# คู่มือแทน (3) long-term memory เพราะรัน goal เดิมซ้ำ 2 รอบเหมือน Test Case A —
# รอบ 2 ควรเห็น record_task()/recall() ของรอบ 1 ทำงานร่วมด้วย
_W8_INTEGRATION_GOAL = (
    "Log in as standard_user/secret_sauce, add the first product to the cart, "
    "click the shopping cart icon to open the Cart page, then click the 'Checkout' "
    "button to go to the checkout information page (a page with First Name, Last "
    "Name, and Zip Code input fields). Fill in First Name, Last Name, and Zip/Postal "
    "Code exactly according to the store's official policy manual — do not invent "
    "your own values, check the reference manual for the exact values required — "
    "then click Continue"
)

# ค่าที่ควรมาจาก RULE-03 ในคู่มือเท่านั้น (ดู manual_test.txt) — ใช้เช็คแบบตรงไปตรงมา
# ว่า agent ดึงค่าจากคู่มือมาใช้จริง ไม่ใช่แค่ "อ้างว่า" ทำตามคู่มือใน finish_task
# message เฉยๆ โดยไม่มีหลักฐานจริงใน history
_MANUAL_EXPECTED_VALUES = ["siamyut", "phasida", "12110"]


def _history_mentions_all_manual_values(history: list[dict]) -> bool:
    combined = " ".join(str(h.get("cmd", "")) for h in history).lower()
    return all(v in combined for v in _MANUAL_EXPECTED_VALUES)


def run_w8_integration():
    print("=== W8: บูรณาการรอบแรก — Perception + คู่มือ (RAG) + ความจำ (memory) ===", flush=True)
    print("ต้อง ingest manual_test.txt ไว้ก่อน (มี RULE-03: ค่า First/Last Name/Zip", flush=True)
    print("Code ที่ต้องใช้ตอน checkout) — ถ้ายังไม่เคย ingest ให้รัน `python run.py", flush=True)
    print("ingest` ก่อนเทสนี้\n", flush=True)
    from backend.app.core import long_term_memory
    from backend.app.core.orchestrator import Orchestrator

    async def _run():
        result1 = await Orchestrator().run_task(
            "https://www.saucedemo.com/", _W8_INTEGRATION_GOAL, headless=False, verbose=True,
        )
        print("\n=== รอบที่ 1 จบ ===")
        print(f"success: {result1['success']}  steps: {result1['steps']}")
        print(f"message: {result1['message']}")
        manual_used_1 = _history_mentions_all_manual_values(result1["history"])
        print(f"*** ใช้ค่าจากคู่มือครบทั้ง 3 ค่า (Siamyut/Phasida/12110): {manual_used_1} ***\n")

        print("=== recall() ก่อนเริ่มรอบที่ 2 (ควรเห็นร่องรอยของรอบที่ 1) ===")
        recalled = long_term_memory.recall(query=_W8_INTEGRATION_GOAL, k=3)
        if recalled:
            for r in recalled:
                print(f"  - {r}")
        else:
            print("  (recall() ว่างเปล่า — เช็คว่า record_task() ของรอบ 1 เขียนสำเร็จไหม)")

        print("\n=== รอบที่ 2: สั่งงานเดิมซ้ำ ===\n")
        result2 = await Orchestrator().run_task(
            "https://www.saucedemo.com/", _W8_INTEGRATION_GOAL, headless=False, verbose=True,
        )
        print("\n=== รอบที่ 2 จบ ===")
        print(f"success: {result2['success']}  steps: {result2['steps']}")
        print(f"message: {result2['message']}")
        manual_used_2 = _history_mentions_all_manual_values(result2["history"])
        print(f"*** ใช้ค่าจากคู่มือครบทั้ง 3 ค่า (Siamyut/Phasida/12110): {manual_used_2} ***")
        print(
            "\n*** ครบ 3 สมอง: perception (log [step ...]/[OK] ทุก step ด้านบนอ่าน element"
            "\nจริงทุกรอบ), RAG manual (เช็คค่า Siamyut/Phasida/12110 ด้านบนทั้ง 2 รอบ),"
            "\nmemory (เทียบจำนวน steps รอบ 2 กับรอบ 1 + recall() ที่ดึงคืนมาได้จริงก่อน"
            "\nเริ่มรอบ 2) ***"
        )

    asyncio.run(_run())


# --- W9[A]: Vision fallback + handle error states (popup) ---
# ต่างจาก demo อื่นด้านบนที่ขับผ่าน Orchestrator.run_task() ทั้ง goal — ตัวนี้ขับ
# perception/actions/llm ตรงๆ ทีละขั้น (เหมือน run_permission_demo() ด้านบนที่ขับ
# execute() ตรงๆ ไม่ผ่าน Orchestrator) เพราะต้องฉีด overlay ปลอมเข้า DOM "ระหว่างกลาง"
# หลัง login เสร็จแต่ก่อนจะลองคลิก ซึ่ง Orchestrator.run_task() ไม่มี hook ให้แทรกจังหวะ
# นี้ (ไม่อยากเพิ่ม plumbing ใหม่ใน orchestrator.py แค่เพื่อ demo อย่างเดียว)


async def _inject_fake_cookie_banner(page):
    """ฉีด overlay ปลอมที่หน้าตาเหมือน cookie-consent banner จริง (fixed position คลุม
    ครึ่งล่างของจอ, z-index สูง) ครอบส่วนที่มีปุ่ม 'Add to cart' ของสินค้าชิ้นแรก —
    จำลอง popup ที่เว็บจริงชอบมี แต่ saucedemo.com เองไม่มี (ไว้ทดสอบ W9[A] เท่านั้น)"""
    await page.evaluate("""
        () => {
            const banner = document.createElement('div');
            banner.id = 'fake-cookie-banner';
            banner.style.position = 'fixed';
            banner.style.bottom = '0';
            banner.style.left = '0';
            banner.style.width = '100vw';
            banner.style.height = '100vh';
            banner.style.background = 'rgba(20,20,20,0.85)';
            banner.style.zIndex = '9999';
            banner.style.color = 'white';
            banner.style.display = 'flex';
            banner.style.alignItems = 'center';
            banner.style.justifyContent = 'center';
            banner.style.fontSize = '28px';
            banner.innerText = 'We use cookies. Please accept our cookie policy to continue.';
            document.body.appendChild(banner);
        }
    """)


def run_vision_fallback_demo():
    print("=== W9[A]: Vision fallback + handle error states (popup) ===", flush=True)
    print("ฉีด overlay ปลอม (จำลอง cookie-consent banner) บังปุ่ม 'Add to cart' ของ", flush=True)
    print("สินค้าชิ้นแรกจริงๆ หลัง login เสร็จ แล้วลองคลิกตรงๆ (ผ่าน actions.execute()", flush=True)
    print("ไม่ใช่ mock) ดูว่า perception มองเห็นว่าโดนบัง + คลิกจริง fail + vision", flush=True)
    print("fallback (Gemini) อธิบายสิ่งที่เห็นถูกไหม\n", flush=True)
    from playwright.async_api import async_playwright
    from backend.app.core.perception import get_snapshot
    from backend.app.core.actions import execute
    from backend.app.core import llm
    from backend.app.config import settings

    async def _run():
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            await page.goto("https://www.saucedemo.com/")
            await page.fill("#user-name", "standard_user")
            await page.fill("#password", "secret_sauce")
            await page.click("#login-button")
            await page.wait_for_load_state("networkidle")

            await _inject_fake_cookie_banner(page)
            print("[setup] ฉีด fake cookie banner เข้าหน้าเว็บแล้ว\n", flush=True)

            elements, _ = await get_snapshot(page)
            target = next((e for e in elements if "backpack" in e["label"].lower()), elements[0])
            print(f"[perceive] target element: {target}", flush=True)
            print(f"           *** ต้องเห็น '[ถูกบังอยู่]' ต่อท้าย label ถ้า overlay detection ทำงาน ***\n", flush=True)

            result = await execute(page, {"type": "click", "index": target["index"]})
            print(f"[act] click({target['index']}) -> {result}\n", flush=True)

            if not result.success:
                print("[vision-fallback] action fail จริง กำลังถ่าย screenshot + เรียก Gemini...", flush=True)
                screenshot = await page.screenshot(type="png")
                client = llm.build_gemini_client(settings.gemini_api_key)
                description = await llm.describe_screenshot(
                    client, settings.gemini_model, screenshot, "click", target["index"]
                )
                print(f"[vision-fallback] Gemini อธิบาย: {description}\n", flush=True)
                print("*** เช็คว่าคำอธิบายด้านบนพูดถึง cookie banner/popup ที่ฉีดไว้จริงไหม ***")
            else:
                print("*** click สำเร็จ (ไม่คาดคิด — overlay อาจไม่ได้บัง element จริง) ***")

            await browser.close()

    asyncio.run(_run())


# --- W10[A]: API endpoints (FastAPI) + Browser Pool (persistent) ---
def _print_readable_task_report(status: dict) -> None:
    """รายงานสรุปแบบอ่านง่าย — โชว์แค่ "ทำอะไรไปบ้าง + สำเร็จไหม" ทีละบรรทัด ไม่ต้องไล่
    อ่าน raw dict เต็ม (cmd/tokens/success ครบทุก key ต่อ step) เหมือน [result] ด้านบน —
    ใช้คู่กัน: [result] ไว้ debug ละเอียด ส่วนนี้ไว้กวาดตาเร็วๆ ว่า agent ทำ action อะไร
    ไปบ้างตามลำดับ"""
    result = status.get("result")
    print("[report] สรุปผลแบบอ่านง่าย", flush=True)
    if not result:
        print(f"  {status['status']}: {status.get('error') or 'ไม่มีผลลัพธ์'}", flush=True)
        return

    outcome = "สำเร็จ" if result["success"] else "ไม่สำเร็จ"
    print(f"  ผลลัพธ์: {outcome} ({result['steps']} step)", flush=True)
    for h in result["history"]:
        cmd = h["cmd"]
        action = cmd.get("type", "?")
        if "index" in cmd:
            action += f"({cmd['index']})"
        mark = "OK" if h["success"] else "FAIL"
        print(f"    step {h['step']}: [{mark}] {action}", flush=True)
    print(flush=True)


def run_w10_api_demo():
    """เปิด API server จริง (uvicorn subprocess เหมือน run_server() แต่ปิด --reload กัน
    subprocess ลูกซ้อน) แล้วยิง HTTP request จริงเข้าไป (httpx ไม่ mock) พิสูจน์ 2 อย่าง:

    1) BrowserPool เปิด browser ไว้ล่วงหน้าจริงตอน startup (ดู GET /pool/status ก่อนยิง
       task ใดๆ เลย — available ต้องเท่ากับ size ตั้งแต่แรก ไม่ต้องรอ task แรกมาสร้าง)
    2) POST /tasks คืน task_id ทันที (ไม่ block รอ task จบ ดู core/api/task_manager.py)
       แล้ว GET /tasks/{id} เห็นสถานะเปลี่ยนจาก running -> done จริง พร้อม result ของ
       Orchestrator.run_task() ตัวจริงที่รันบน saucedemo.com ผ่าน browser ที่ยืมมาจาก
       pool (ดู /pool/status ระหว่างรัน available ต้องลดลง แล้วกลับมาเท่าเดิมหลังจบ —
       พิสูจน์ว่า browser ถูกคืน pool ไม่ได้ถูกปิดทิ้งเหมือน W1-W9)
    """
    import time

    import httpx

    from backend.app.config import settings

    port = settings.api_port
    base_url = f"http://127.0.0.1:{port}"

    print("=== W10[A]: API endpoints + Browser Pool (persistent) ===", flush=True)
    print(f"[setup] เปิด uvicorn subprocess จริงที่ {base_url} ...", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", str(port)],
    )
    try:
        with httpx.Client(base_url=base_url, timeout=10) as http:
            for _ in range(60):
                try:
                    if http.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.5)
            else:
                print("[setup] server ไม่ขึ้นภายในเวลาที่กำหนด", flush=True)
                return

            pool_before = http.get("/pool/status").json()
            print(f"[pool] status ก่อนยิง task ใดๆ เลย: {pool_before}", flush=True)
            print("       *** available ต้องเท่ากับ size แล้วตั้งแต่ startup (เปิดไว้ล่วงหน้า) ***\n", flush=True)

            print("[submit] POST /tasks (goal: login + Add to cart สินค้าชิ้นแรกบน saucedemo.com)", flush=True)
            resp = http.post("/tasks", json={
                "url": "https://www.saucedemo.com/",
                "goal": "login ด้วย standard_user/secret_sauce แล้วกด Add to cart ของสินค้าชิ้นแรก",
                "max_steps": 15,
                "provider": "gemini",
            })
            resp.raise_for_status()
            task_id = resp.json()["task_id"]
            print(f"[submit] -> {resp.status_code}, task_id={task_id}, status={resp.json()['status']}\n", flush=True)

            pool_during = http.get("/pool/status").json()
            print(f"[pool] status ระหว่าง task กำลังรัน: {pool_during}", flush=True)
            print("       *** available ต้องลดลง 1 (ถูกยืมไปแล้ว) ***\n", flush=True)

            print("[poll] รอ task จบ (GET /tasks/{task_id} ซ้ำๆ)...", flush=True)
            status = None
            for _ in range(120):
                status = http.get(f"/tasks/{task_id}").json()
                if status["status"] != "running":
                    break
                time.sleep(2)
            else:
                print("[poll] task ไม่จบภายในเวลาที่กำหนด", flush=True)
                return

            print(f"[result] status={status['status']}", flush=True)
            print(f"[result] {status['result'] or status['error']}\n", flush=True)

            _print_readable_task_report(status)

            pool_after = http.get("/pool/status").json()
            print(f"[pool] status หลัง task จบ: {pool_after}", flush=True)
            print(
                "       *** available ต้องกลับมาเท่ากับ size (browser คืน pool แล้ว "
                "ไม่ถูกปิดทิ้งเหมือน W1-W9) ***",
                flush=True,
            )
    finally:
        print("\n[teardown] ปิด uvicorn subprocess...", flush=True)
        proc.terminate()
        proc.wait(timeout=10)


# --- W11: Integration hardening (chaos / concurrency / isolation) ---
# ทั้ง 3 test ข้างล่างเปิด uvicorn subprocess จริงของตัวเอง (คนละ port กัน และแยกจาก
# settings.api_port default กันชนกับ server ที่ user อาจเปิดใช้งานอยู่แล้วจริงๆ) แล้วยิง
# HTTP จริงเข้า endpoint ของ W10[A] เหมือน run_w10_api_demo() ด้านบน — แยก helper
# _start_test_api_server()/_wait_task_done() ออกมาใช้ร่วมกันแทนก็อปโค้ด start/poll/
# teardown ซ้ำ 3 รอบ (run_w10_api_demo() เดิมปล่อยไว้แบบเดิมไม่แตะ กันกระทบของเก่า)


def _start_test_api_server(port: int):
    """เปิด uvicorn subprocess จริง (เหมือน run_w10_api_demo()) รอจน /health ตอบ 200 แล้ว
    คืน (proc, http_client) กลับมาให้ผู้เรียกใช้ต่อเอง — ผู้เรียกต้องเรียก proc.terminate()
    + proc.wait() เองตอนจบ (ไม่ใช้ context manager เพราะต้องการคุม try/finally เองที่
    เรียก เผื่ออยาก assert/print เพิ่มเติมระหว่างทาง)"""
    import time

    import httpx

    print(f"[setup] เปิด uvicorn subprocess จริงที่ port {port} ...", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", str(port)],
    )
    http = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15)
    for _ in range(60):
        try:
            if http.get("/health").status_code == 200:
                return proc, http
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    proc.terminate()
    proc.wait(timeout=10)
    raise RuntimeError(f"server ที่ port {port} ไม่ขึ้นภายในเวลาที่กำหนด")


def _wait_task_done(http, task_id: str, *, timeout_s: float = 120.0, poll_s: float = 2.0) -> dict:
    """poll GET /tasks/{id} จนกว่า status จะไม่ใช่ 'running' แล้ว (เหมือน pattern ใน
    run_w10_api_demo() เดิม) คืน body สุดท้ายที่ได้กลับมาเสมอ แม้จะ timeout (ให้ผู้เรียก
    เห็นสถานะล่าสุดที่ค้างอยู่แทนที่จะ raise เฉยๆ — test พวกนี้อยากรู้ "มันค้างตรงไหน"
    ไม่ใช่แค่ "มันไม่จบ")"""
    import time

    deadline = time.monotonic() + timeout_s
    status = http.get(f"/tasks/{task_id}").json()
    while time.monotonic() < deadline and status["status"] == "running":
        time.sleep(poll_s)
        status = http.get(f"/tasks/{task_id}").json()
    return status


def _find_chromium_pids(server_pid: int) -> list[int]:
    """หา PID ของ chrome.exe/chrome ที่เป็น "หัวสุด" ของแต่ละ instance ที่ BrowserPool
    เปิดไว้ — เดินต้นไม้ descendant ของ server_pid (uvicorn subprocess ที่ตัวเองเปิดขึ้น
    เท่านั้น) แบบ recursive ผ่าน psutil แล้วกรองเฉพาะชื่อ process ที่มีคำว่า "chrome"
    เป็นสำคัญ — จำกัดผลไว้แค่ descendant ของ subprocess ตัวนี้เท่านั้น (ไม่ใช่เดินทั้งเครื่อง)
    กันพลาดไปฆ่า Chrome จริงของเครื่อง หรือ browser ของ server W10/W11 อีกตัวที่อาจเปิด
    ค้างอยู่พร้อมกัน — คืนเฉพาะ process ที่ parent ไม่ใช่ chrome ด้วยกันเอง (กันฆ่า renderer/
    gpu process ลูกซ้ำ ทั้งที่จะตายเองอยู่แล้วทันทีที่ process หัวถูกฆ่า)"""
    import psutil

    try:
        root = psutil.Process(server_pid)
    except psutil.NoSuchProcess:
        return []
    descendants = root.children(recursive=True)
    chrome_procs = []
    for p in descendants:
        try:
            if "chrome" in (p.name() or "").lower():
                chrome_procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    chrome_pid_set = {p.pid for p in chrome_procs}
    top_level = []
    for p in chrome_procs:
        try:
            parent = p.parent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if parent is None or parent.pid not in chrome_pid_set:
            top_level.append(p.pid)
    return top_level


def run_chaos_test():
    """W11[A]: Chaos Test — ฆ่า process ของ browser จริง (chrome.exe) กลางคันขณะ task
    กำลังรันอยู่จริง (ไม่ mock) แล้วเช็ค 2 อย่าง:

    1) server ยังทำงานต่อได้ปกติไหม (ไม่ล่มทั้งตัวตาม browser ที่ตาย) และ task ที่โดน
       ฆ่ากลางคันจบด้วย status="error" อย่างสุภาพไหม (ไม่ hang ค้างตลอดไป) — คาดว่า "ผ่าน"
       เพราะ task_manager.py::_run() ครอบ try/except ทุก Exception ไว้อยู่แล้ว
    2) หลังจากนั้นยิง task ใหม่อีกตัว เช็คว่า BrowserPool "self-heal" (เปลี่ยน browser ที่
       ตายให้เป็นตัวใหม่) จริงไหม — ฆ่า chrome.exe *ทุกตัว* ในทุก browser ของ pool (ไม่ใช่
       แค่ตัวที่ task แรกใช้) กันผลเพี้ยนจาก FIFO ของ asyncio.Queue ที่อาจทำให้ task ใหม่
       บังเอิญได้ browser ตัวที่ไม่โดนฆ่าไปใช้แทน (ดูเหมือน self-heal ทั้งที่จริงๆ แค่ยังมี
       ตัวสำรองเหลือใน pool เฉยๆ)
    """
    import time

    print("=== W11[A]: Chaos Test — ฆ่า browser process กลางคัน (verify self-healing) ===", flush=True)
    port = 8011  # แยกจาก settings.api_port (8000) กันชนกับ server จริงที่ user อาจเปิดอยู่
    proc, http = _start_test_api_server(port)
    try:
        pool_before = http.get("/pool/status").json()
        print(f"[pool] ก่อนเริ่ม: {pool_before}", flush=True)

        print("[submit] POST /tasks (goal ยาวพอให้มีเวลาฆ่า browser ทันตอนกำลังทำงานจริง)", flush=True)
        resp = http.post("/tasks", json={
            "url": "https://www.saucedemo.com/",
            "goal": "Log in as standard_user/secret_sauce, add the first three products "
                    "to the cart one at a time, then go to the cart page.",
            "max_steps": 15, "auto_approve": True, "confirm_plan": False, "headless": True,
        })
        resp.raise_for_status()
        task_id = resp.json()["task_id"]
        print(f"[submit] task_id={task_id}", flush=True)

        print("[wait] รอให้ task ทำอย่างน้อย 1 step สำเร็จก่อน (ถึงจะนับว่าฆ่า 'กลางคัน' จริง)", flush=True)
        history = []
        for _ in range(30):
            status = http.get(f"/tasks/{task_id}").json()
            history = (status.get("result") or {}).get("history") or []
            if len(history) >= 1 or status["status"] != "running":
                break
            time.sleep(1)
        print(f"[wait] เห็น {len(history)} step ก่อนฆ่า browser (status={status['status']})", flush=True)

        chrome_pids = _find_chromium_pids(proc.pid)
        print(f"[chaos] เจอ chromium process ลูกของ server ตัวนี้: {chrome_pids}", flush=True)
        if not chrome_pids:
            print("[chaos] *** ไม่เจอ chromium process เลย — ข้าม test นี้ "
                  "(เช็คว่า psutil/Playwright ติดตั้งครบไหม) ***", flush=True)
            return
        killed = 0
        import psutil
        for pid in chrome_pids:
            try:
                psutil.Process(pid).kill()
                killed += 1
            except psutil.NoSuchProcess:
                pass
        print(f"[chaos] ฆ่า chromium process ไปแล้ว {killed}/{len(chrome_pids)} ตัว "
              "(จำลอง browser ทั้ง pool crash พร้อมกัน)", flush=True)

        print("\n[verify 1] รอ task เดิมจบ (ควรจบด้วย status='error' อย่างสุภาพ ไม่ hang)...", flush=True)
        final = _wait_task_done(http, task_id, timeout_s=60)
        print(f"[verify 1] task เดิมหลังโดนฆ่า browser: status={final['status']}, "
              f"error={final.get('error')}", flush=True)

        server_alive = http.get("/health").status_code == 200
        print(f"[verify 1] server ยังตอบ /health ปกติไหม (ไม่ล่มทั้งตัวตาม browser): "
              f"{server_alive}", flush=True)

        print("\n[verify 2] ยิง task ใหม่อีกตัวหลัง chaos เช็คว่า pool self-heal จริงไหม...", flush=True)
        resp2 = http.post("/tasks", json={
            "url": "https://www.saucedemo.com/",
            "goal": "Log in as standard_user/secret_sauce.",
            "max_steps": 8, "auto_approve": True, "confirm_plan": False, "headless": True,
        })
        task_id2 = resp2.json()["task_id"]
        final2 = _wait_task_done(http, task_id2, timeout_s=60)
        success2 = bool((final2.get("result") or {}).get("success"))
        print(f"[verify 2] task ใหม่หลัง chaos: status={final2['status']}, success={success2}, "
              f"error={final2.get('error')}", flush=True)

        print("\n=== สรุป ===", flush=True)
        print(f"  server รอด (ไม่ล่มทั้งตัว): {server_alive}", flush=True)
        print(f"  task ที่โดนฆ่ากลางคันจบแบบสุภาพ (ไม่ hang): {final['status'] != 'running'}", flush=True)
        if success2:
            print("  *** ไม่คาดคิด: task หลัง chaos สำเร็จ — เช็คว่า pool ยังมี browser ตัวอื่น"
                  "\n      ที่ไม่โดนฆ่าเหลืออยู่ไหม (ถ้าฆ่าครบทุกตัวแล้วแปลว่า self-heal จริง) ***",
                  flush=True)
        else:
            print("  self-heal: ไม่ทำงาน — BrowserPool ปัจจุบันไม่มี health check ตอน acquire() "
                  "เลย ถ้า browser ที่ปล่อยกลับเข้า pool ตายไปแล้ว จะถูกส่งให้ task ถัดไปใช้ซ้ำ"
                  "\n  ไปเรื่อยๆ จนกว่า server จะ restart เท่านั้น (ยืนยันได้จริงจาก test นี้ "
                  "ไม่ใช่การเดา — ถ้าจะแก้ ต้องเพิ่ม health check เช่น browser.is_connected() "
                  "ก่อนคืน browser ให้ task ถัดไปใน browser_pool.py::acquire())", flush=True)
    finally:
        print("\n[teardown] ปิด uvicorn subprocess...", flush=True)
        proc.terminate()
        proc.wait(timeout=10)


def run_high_concurrency_test():
    """W11[B]: High-Concurrency Test — ยิง 5 task พร้อมกันจริงเข้า pool ที่มีแค่ 2 browser
    (settings.browser_pool_size default) พิสูจน์ queueing จริง: in_use ต้องไม่มีวันเกิน
    pool size, ต้องเห็น task ส่วนเกินรอคิวอยู่จริง (ไม่ crash/deadlock) แล้วสุดท้ายทั้ง 5
    ต้องจบให้ครบ — หมายเหตุ: status="running" ของ API/UI ปัจจุบันไม่แยกระหว่าง "รอคิวอยู่"
    กับ "กำลังทำงานจริง" (ดู roadmap.txt W11 findings) เลยใช้ pool.in_use เทียบ pool.size
    แทนการอ่าน status ตรงๆ เป็นหลักฐานว่ามี task รอคิวจริง"""
    import time

    print("=== W11[B]: High-Concurrency Test — 5 task พร้อมกันบน pool ขนาด 2 ===", flush=True)
    port = 8012  # แยกจาก settings.api_port (8000) และจาก W11[A] (8011) กันชนกัน
    proc, http = _start_test_api_server(port)
    try:
        pool_before = http.get("/pool/status").json()
        print(f"[pool] ก่อนยิง: {pool_before}", flush=True)

        print("[submit] ยิง 5 task พร้อมกัน (auto_approve, confirm_plan=False, goal เบาๆ)...", flush=True)
        task_ids = []
        for i in range(5):
            resp = http.post("/tasks", json={
                "url": "https://www.saucedemo.com/",
                "goal": f"Log in as standard_user/secret_sauce. (concurrency probe #{i})",
                "max_steps": 6, "auto_approve": True, "confirm_plan": False, "headless": True,
            })
            resp.raise_for_status()
            task_ids.append(resp.json()["task_id"])
            print(f"  submitted #{i}: {task_ids[-1]}", flush=True)

        print("\n[poll] ติดตามทุก 2 วิ จนกว่าทั้ง 5 จะจบ...", flush=True)
        max_in_use_seen = 0
        saw_queueing = False
        for _ in range(120):
            pool = http.get("/pool/status").json()
            max_in_use_seen = max(max_in_use_seen, pool["in_use"])
            statuses = [http.get(f"/tasks/{tid}").json()["status"] for tid in task_ids]
            running = statuses.count("running")
            if pool["in_use"] >= pool["size"] and running > pool["size"]:
                saw_queueing = True
            print(f"  pool={pool} statuses={statuses}", flush=True)
            if all(s != "running" for s in statuses):
                break
            time.sleep(2)
        else:
            print("*** ไม่จบครบภายในเวลาที่กำหนด — อาจ deadlock จริง ***", flush=True)

        print(f"\n[verify] in_use สูงสุดที่เห็นตลอดการทดสอบ: {max_in_use_seen}/{pool_before['size']} "
              "(ต้องไม่เกิน pool size เด็ดขาด ไม่งั้น pool มี bug จริง)", flush=True)
        print(f"[verify] เคยเห็น task รอคิวจริง (running มากกว่า pool size พร้อมกัน): "
              f"{saw_queueing}", flush=True)

        results = [http.get(f"/tasks/{tid}").json() for tid in task_ids]
        succeeded = sum(1 for r in results if (r.get("result") or {}).get("success"))
        print(f"\n[result] สำเร็จ {succeeded}/5 — statuses: {[r['status'] for r in results]}", flush=True)
        for r in results:
            if r["status"] != "done" or not (r.get("result") or {}).get("success"):
                print(f"  *** ตัวที่ไม่สำเร็จ: {r['task_id']} status={r['status']} "
                      f"error={r.get('error')}", flush=True)
    finally:
        print("\n[teardown] ปิด uvicorn subprocess...", flush=True)
        proc.terminate()
        proc.wait(timeout=10)


_W11C_TASK_A_GOAL = (
    "Log in with username 'standard_user' and password 'secret_sauce' (these credentials "
    "are correct). After logging in, report the exact page heading/title you see."
)
_W11C_TASK_B_GOAL = (
    "Log in with username 'standard_user' and password 'this_is_wrong_on_purpose' (this "
    "password is deliberately incorrect, do not try any other password). Report the exact "
    "error message text shown on the page after the failed login attempt."
)


def run_isolation_test():
    """W11[C]: Isolation Test — รัน 2 task พร้อมกันจริง (ไม่ทีละตัว) บน pool เดียวกัน ด้วย
    credential ที่ตั้งใจให้ผลต่างกันชัดเจน (task A login ถูกต้องจริง, task B login ผิดรหัส
    ตั้งใจ) พิสูจน์ว่า BrowserContext ของแต่ละ task แยกกันจริง ไม่แชร์ cookie/session ข้าม
    task แม้จะใช้ browser process เดียวกันจาก pool ก็ตาม (ดู orchestrator.py ~บรรทัด 492:
    browser.new_context() ใหม่ทุกครั้งที่ยืม browser จาก pool — เทสนี้ยืนยันด้วยผลลัพธ์จริง
    แทนอ่านโค้ดเฉยๆ):
      - task A ต้องเห็นหน้า Products (login สำเร็จจริง) ไม่มีร่องรอย error ของ B ปนมา
      - task B ต้องเห็น error message จริงของ saucedemo (login ล้มเหลวจริงตามที่ตั้งใจ) ไม่ได้
        กลายเป็น "login สำเร็จ" เพราะดันไปสวม session ที่ A login ไว้ก่อน (= context รั่ว)
    """
    print("=== W11[C]: Isolation Test — 2 task พร้อมกัน คนละ session ===", flush=True)
    port = 8013  # แยกจาก settings.api_port (8000), W11[A] (8011), W11[B] (8012)
    proc, http = _start_test_api_server(port)
    try:
        pool_before = http.get("/pool/status").json()
        print(f"[pool] ก่อนยิง: {pool_before}", flush=True)

        print("[submit] Task A (credential ถูก) + Task B (credential ผิดตั้งใจ) พร้อมกัน", flush=True)
        resp_a = http.post("/tasks", json={
            "url": "https://www.saucedemo.com/", "goal": _W11C_TASK_A_GOAL,
            "max_steps": 6, "auto_approve": True, "confirm_plan": False, "headless": True,
        })
        resp_b = http.post("/tasks", json={
            "url": "https://www.saucedemo.com/", "goal": _W11C_TASK_B_GOAL,
            "max_steps": 6, "auto_approve": True, "confirm_plan": False, "headless": True,
        })
        task_a, task_b = resp_a.json()["task_id"], resp_b.json()["task_id"]
        print(f"  task A (ควร login สำเร็จ): {task_a}", flush=True)
        print(f"  task B (ควร login ล้มเหลวด้วย error จริง): {task_b}", flush=True)

        pool_during = http.get("/pool/status").json()
        print(f"[pool] ระหว่างรันพร้อมกัน: {pool_during} (ควรเห็น in_use=2 ถ้าจับจังหวะทัน)", flush=True)

        final_a = _wait_task_done(http, task_a, timeout_s=90)
        final_b = _wait_task_done(http, task_b, timeout_s=90)

        state_a = (final_a.get("result") or {}).get("final_page_state", "") or ""
        state_b = (final_b.get("result") or {}).get("final_page_state", "") or ""
        msg_a = (final_a.get("result") or {}).get("message", "") or ""
        msg_b = (final_b.get("result") or {}).get("message", "") or ""
        print(f"\n[task A] status={final_a['status']} message={msg_a!r}", flush=True)
        print(f"[task A] final_page_state={state_a!r}", flush=True)
        print(f"\n[task B] status={final_b['status']} message={msg_b!r}", flush=True)
        print(f"[task B] final_page_state={state_b!r}", flush=True)

        blob_a = (state_a + " " + msg_a).lower()
        blob_b = (state_b + " " + msg_b).lower()
        a_logged_in = "inventory" in blob_a or "products" in blob_a
        a_has_error_leak = "epic sadface" in blob_a
        # ข้อสังเกต: b_shows_real_error เป็นแค่ข้อมูลประกอบ (บาง run โมเดลไม่ทันได้เห็น/
        # รายงานข้อความ error ตรงๆ ก่อนโดน loop-detection guard ตัดจบ เพราะ perception.py
        # เก็บแค่ indexed *interactive* elements — banner ข้อความ error ธรรมดาไม่ใช่
        # element ที่คลิกได้ เลยไม่ติด index ให้เห็นเสมอไป) ไม่ใช่ตัวชี้วัด isolation ตรงๆ —
        # ตัวชี้วัด isolation จริงคือ b_wrongly_logged_in ล้วนๆ (ถ้า context รั่วจริง B จะ
        # จบด้วยหน้า inventory/products เหมือน A ทั้งที่ตั้งใจกรอกรหัสผิด)
        b_shows_real_error = "epic sadface" in blob_b or "do not match" in blob_b
        b_wrongly_logged_in = "inventory" in blob_b or "products" in blob_b

        isolation_ok = a_logged_in and not a_has_error_leak and not b_wrongly_logged_in

        print("\n=== สรุปผล isolation ===", flush=True)
        print(f"  Task A login สำเร็จจริง ไม่มี error ของ B ปนมา: "
              f"{a_logged_in and not a_has_error_leak}", flush=True)
        print(f"  Task B ไม่ได้แอบ login สำเร็จเพราะ session รั่วจาก A "
              f"(ยังคาราคาซังที่หน้า login เหมือนที่ควรจะเป็น): "
              f"{not b_wrongly_logged_in}", flush=True)
        print(f"  (ข้อมูลประกอบ ไม่ใช่ตัวชี้วัด isolation) Task B รายงานข้อความ error "
              f"ตรงๆ ได้ทันไหม: {b_shows_real_error}", flush=True)
        if a_has_error_leak or b_wrongly_logged_in:
            print("  *** พบสัญญาณ session รั่วข้าม task จริง — ต้องสืบต่อว่า BrowserContext "
                  "แยกกันจริงไหม (ดู orchestrator.py::run_task เส้นทาง owns_browser=False) ***",
                  flush=True)
        elif isolation_ok:
            print("  isolation ทำงานถูกต้อง: แต่ละ task ได้ BrowserContext ของตัวเองจริง "
                  "ไม่แชร์ session กันแม้ใช้ browser process เดียวกันจาก pool", flush=True)
        else:
            print("  *** ผลไม่ชัดเจน (เช่น task A เองไม่ได้ login สำเร็จด้วยเหตุผลอื่น) — "
                  "ไล่ดู [task A]/[task B] ด้านบนเพื่อสืบสาเหตุ ***", flush=True)
    finally:
        print("\n[teardown] ปิด uvicorn subprocess...", flush=True)
        proc.terminate()
        proc.wait(timeout=10)


def run_evaluation_harness():
    """W12[B]: Evaluation แนว WebVoyager — รัน BENCHMARK_TASKS (goal เดิมจาก demo อื่นๆ ใน
    ไฟล์นี้ ดู core/evaluation.py) ทีละตัวตามลำดับผ่าน Orchestrator.run_task() ตรงๆ บน
    saucedemo.com จริง (headless, auto_approve, provider จาก .env — ไม่มีคนเฝ้าหน้าจอ)
    แล้วพิมพ์รายงานสรุป success rate / avg steps / avg tokens ต่อ task"""
    print("=== W12[B]: Evaluation แนว WebVoyager (success rate / step / token ต่อ task) ===", flush=True)
    from backend.app.config import settings
    from backend.app.core.evaluation import BENCHMARK_TASKS, run_evaluation

    print(f"Provider: {settings.llm_provider}", flush=True)
    print(f"รัน {len(BENCHMARK_TASKS)} task บน saucedemo.com (headless, auto_approve)...\n", flush=True)

    async def _run():
        report = await run_evaluation()

        print("=== ผลลัพธ์รายภารกิจ ===", flush=True)
        for r in report.results:
            outcome = "สำเร็จ" if r.success else "ไม่สำเร็จ"
            print(f"  [{r.name}] {outcome} — {r.steps} step, {r.total_tokens} token", flush=True)
            print(f"      {'error: ' + r.error if r.error else 'message: ' + r.message}", flush=True)

        n = len(report.results)
        n_success = sum(1 for r in report.results if r.success)
        print("\n=== สรุปรวม ===", flush=True)
        print(f"  Success rate : {report.success_rate:.0%} ({n_success}/{n})", flush=True)
        print(f"  Avg steps    : {report.avg_steps:.1f}", flush=True)
        print(f"  Avg tokens   : {report.avg_tokens:.0f}", flush=True)

    asyncio.run(_run())


ACTIONS = {
    "1": ("รัน API server", run_server),
    "2": ("รัน tests (pytest)", run_tests),
    "3": ("รัน perception demo (saucedemo login)", run_perception_demo),
    "4": ("Ingest คู่มือเข้า ChromaDB", run_ingest),
    "5": ("ค้นคู่มือด้วย retrieve()", run_query),
    "6": ("รัน Agent Loop (Orchestrator.run_task)", run_agent),
    "7": ("ทดสอบ Permission Layer (classify_action + execute)", run_permission_demo),
    "8": ("W7[A] Test Case A: จำ pattern ความผิดพลาด (long-term memory)", run_test_case_a),
    "9": ("W7[A] Test Case B: จดจำข้อมูลสำคัญข้ามรอบ (long-term memory)", run_test_case_b),
    "10": ("W7[A] Test Case C: token/context compaction (Gemini เท่านั้น)", run_test_case_c),
    "11": ("W7[B] Test Case D: RAG-based permission (คู่มือกำหนด action ที่ต้องขออนุมัติ)", run_test_case_d),
    "12": ("W8: บูรณาการรอบแรก (Perception + คู่มือ RAG + ความจำ = ครบ 3 สมอง)", run_w8_integration),
    "13": ("W9[A]: Vision fallback + handle error states (popup)", run_vision_fallback_demo),
    "14": ("W10[A]: API endpoints (FastAPI) + Browser Pool (persistent)", run_w10_api_demo),
    "15": ("W11[A]: Chaos Test - kill browser process กลางคัน (verify self-healing)", run_chaos_test),
    "16": ("W11[B]: High-Concurrency Test - 5 task พร้อมกันบน pool ขนาด 2 (verify queueing)", run_high_concurrency_test),
    "17": ("W11[C]: Isolation Test - multi-tab ไม่แทรกแซงกัน (verify context isolation)", run_isolation_test),
    "18": ("รัน Agent Loop บน Chrome จริงของ user (CDP connect, มี login mail ค้างอยู่)", run_agent_real_browser),
    "19": ("W12[B]: Evaluation แนว WebVoyager (success rate / step / token ต่อ task)", run_evaluation_harness),
}

ALIASES = {
    "server": "1",
    "test": "2",
    "tests": "2",
    "perception": "3",
    "demo": "3",
    "ingest": "4",
    "query": "5",
    "agent": "6",
    "permission": "7",
    "perm": "7",
    "memory-a": "8",
    "memory-b": "9",
    "memory-c": "10",
    "permission-rag": "11",
    "integration": "12",
    "w8": "12",
    "vision-fallback": "13",
    "w9": "13",
    "api-demo": "14",
    "w10": "14",
    "chaos": "15",
    "w11a": "15",
    "concurrency": "16",
    "w11b": "16",
    "isolation": "17",
    "w11c": "17",
    "real-browser": "18",
    "user-browser": "18",
    "eval": "19",
    "evaluation": "19",
    "w12": "19",
}


def show_menu():
    print("=== AI Browser Agent — เลือกสิ่งที่จะรัน ===")
    for key, (label, _) in ACTIONS.items():
        print(f"  {key}) {label}")
    print("  q) ออก")
    return input("เลือก: ").strip().lower()


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().lower()
        choice = ALIASES.get(arg, arg)
    else:
        choice = show_menu()

    if choice in ("q", "quit", "exit"):
        return

    action = ACTIONS.get(choice)
    if action is None:
        print(f"ไม่รู้จักตัวเลือก: {choice!r}")
        sys.exit(1)

    _, func = action
    func()


if __name__ == "__main__":
    main()
