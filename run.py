"""
run.py — จุดรันเดียวของโปรเจกต์ (รวม server / test / demo ไว้ในไฟล์นี้ไฟล์เดียว)

วิธีใช้:
    python run.py            # เปิดเมนูให้เลือก
    python run.py server     # รัน API server (uvicorn --reload)
    python run.py test       # รัน pytest ทั้งหมด
    python run.py perception # รัน demo perception.py (login saucedemo.com)
"""

import asyncio
import subprocess
import sys

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


ACTIONS = {
    "1": ("รัน API server", run_server),
    "2": ("รัน tests (pytest)", run_tests),
    "3": ("รัน perception demo (saucedemo login)", run_perception_demo),
}

ALIASES = {
    "server": "1",
    "test": "2",
    "tests": "2",
    "perception": "3",
    "demo": "3",
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
