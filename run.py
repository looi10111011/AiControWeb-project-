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
