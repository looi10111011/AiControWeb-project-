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
"""

import asyncio
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




_DEFAULT_AGENT_GOAL = "Log in as standard_user/secret_sauce and add Sauce Labs Backpack to cart and chechout by siamyut phaseeda 12110"


def run_agent():
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


ACTIONS = {
    "1": ("รัน API server", run_server),
    "2": ("รัน tests (pytest)", run_tests),
    "3": ("รัน perception demo (saucedemo login)", run_perception_demo),
    "4": ("Ingest คู่มือเข้า ChromaDB", run_ingest),
    "5": ("ค้นคู่มือด้วย retrieve()", run_query),
    "6": ("รัน Agent Loop (Orchestrator.run_task)", run_agent),
    "7": ("ทดสอบ Permission Layer (classify_action + execute)", run_permission_demo),
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
