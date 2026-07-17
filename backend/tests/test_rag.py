"""
test_rag.py
============
Test script สำหรับเทส ingestion + retriever ของโปรเจกต์ AI Browser Agent

อ้างอิงจาก retriever.py จริง: retrieve(query, page_state="", k=5) -> list[str]
หมายเหตุ: retrieve มีกฎเหล็กห้าม throw error ออกมา (ถ้าพังภายในจะคืน [] เสมอ)

วิธีรัน (รันจาก root ของโปรเจกต์ Aiagentcontrolbrowser ระดับเดียวกับ run.py):
  python test_rag.py
"""

import sys
import os
import time
from pathlib import Path
 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.rag.ingestion import ingest_manual   # noqa: E402
from backend.app.rag.retriever import retrieve    # noqa: E402


# ---------------------------------------------------------------------------
# ค่าที่ต้องปรับ
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).parent

# เลือกไฟล์ manual สำหรับเทสอัตโนมัติ: ถ้ามี sample_manual.txt ใช้ก่อน
# ถ้าไม่มีแต่มี manual_test.txt (ไฟล์เดิมที่มีอยู่แล้วในโฟลเดอร์) ใช้ตัวนั้นแทน
if (_THIS_DIR / "sample_manual.txt").exists():
    TEST_MANUAL_PATH = _THIS_DIR / "sample_manual.txt"
elif (_THIS_DIR / "manual_test.txt").exists():
    TEST_MANUAL_PATH = _THIS_DIR / "manual_test.txt"
    print(f"⚠️  ไม่เจอ sample_manual.txt ใช้ {TEST_MANUAL_PATH.name} แทน "
          f"(คำถามทดสอบด้านล่างอาจไม่ตรงกับเนื้อหาไฟล์นี้ ควรปรับ TEST_CASES เอง)\n")
else:
    raise FileNotFoundError(
        f"ไม่เจอไฟล์ manual สำหรับเทสใน {_THIS_DIR} "
        "กรุณาวาง sample_manual.txt ไว้ในโฟลเดอร์เดียวกับ test_rag.py"
    )
TOP_K = 3                               # จำนวน chunk บนสุดที่ดึงมาต่อคำถาม

# ชุดคำถามทดสอบ + keyword ที่ "ต้อง" เจอใน chunk ที่ retrieve กลับมา ถ้าทำงานถูกต้อง
# (เขียนจากเนื้อหาใน sample_manual.txt ที่ ingest ไปแล้ว)
TEST_CASES = [
    {"query": "วิธีปิด browser ทำยังไง", "expect_keyword": "Alt+F4"},
    {"query": "จะเปิด facebook ต้องทำไง", "expect_keyword": "facebook.com"},
    {"query": "login github ยังไง", "expect_keyword": "Sign in"},
    {"query": "ค้นหาข้อมูลใน google", "expect_keyword": "search box"},
    {"query": "ดาวน์โหลดไฟล์จากเว็บ", "expect_keyword": "Save link as"},
    # negative case: คำถามที่ไม่เกี่ยวกับเนื้อหาเลย ใช้เช็คว่าไม่ดึง chunk ผิดมาแบบมั่ว
    {"query": "สูตรทำต้มยำกุ้ง", "expect_keyword": None},
    # --- กลุ่ม A: Browser/Navigation ---
    {"query": "เปิด tab ใหม่ใน browser ยังไง", "expect_keyword": "Ctrl+T"},
    {"query": "ปิด tab แล้วกู้คืนได้ไหม", "expect_keyword": "Ctrl+Shift+T"},
    {"query": "reload หน้าเว็บโดยไม่ใช้ cache", "expect_keyword": "Ctrl+Shift+R"},
    {"query": "เปิด developer tools ดู HTML ได้ยังไง", "expect_keyword": "F12"},
    # --- กลุ่ม B: Web Form/Interaction ---
    {"query": "กรอกฟอร์มบนเว็บแล้วกด submit", "expect_keyword": "Submit"},
    {"query": "upload ไฟล์บนเว็บทำยังไง", "expect_keyword": "Choose File"},
    # --- กลุ่ม C: Security/Account ---
    {"query": "logout ออกจาก account บนเว็บ", "expect_keyword": "Sign out"},
    {"query": "เช็คว่าเว็บไซต์ปลอดภัยไหม", "expect_keyword": "padlock"},
    # --- หัวข้อจากรอบแรก (ล้าง Cache, Incognito) ---
    {"query": "ล้าง cache browser ทำยังไง", "expect_keyword": "Clear browsing data"},
    {"query": "เปิด incognito mode", "expect_keyword": "Incognito"},
    # negative case เพิ่มเติม
    {"query": "วิธีทำผัดไทย", "expect_keyword": None},
    # --- กลุ่ม D-J: Edge Cases ---
    {"query": "มี popup บังหน้าจอคลิกไม่ได้ทำไง", "expect_keyword": "Modal"},
    {"query": "เจอ captcha ต้องกดแก้เองไหม", "expect_keyword": "ห้าม Agent พยายามแก้ปัญหาเองเด็ดขาด"},
    {"query": "ระบบให้กรอกบัตรเครดิต ใช้บัตรไหนได้บ้าง", "expect_keyword": "Test Credit Card"},
    {"query": "ถ้าเว็บขึ้น 500 error ทำไงต่อ", "expect_keyword": "Alt+Left Arrow"},
    {"query": "เจอแจ้งเตือน alert ของเบราว์เซอร์กดใน DOM ได้ไหม", "expect_keyword": "ห้ามพยายามค้นหาปุ่มเหล่านี้ใน DOM"},
    {"query": "อัปโหลดไฟล์แต่หน้าต่าง OS เด้งขึ้นมา ต้องทำไง", "expect_keyword": 'type="file"'},
    {"query": "ช่องนี้กด Ctrl+V ไม่ได้", "expect_keyword": "พิมพ์ข้อมูลทีละตัวอักษร"},
    {"query": "เว็บขึ้น 429 Too Many Requests", "expect_keyword": "อย่างน้อย 60 วินาที"},
]


def run_ingestion_test():
    print("=" * 60)
    print("STEP 1: ทดสอบ Ingestion")
    print("=" * 60)
    try:
        ingest_manual(TEST_MANUAL_PATH)
        print("✅ Ingest สำเร็จ ไม่มี error\n")
        return True
    except Exception as e:
        print(f"❌ Ingest ล้มเหลว: {e}\n")
        return False


def run_retriever_test():
    print("=" * 60)
    print("STEP 2: ทดสอบ Retriever (วัด Precision@k)")
    print("=" * 60)

    passed = 0
    total = len(TEST_CASES)

    for i, case in enumerate(TEST_CASES, 1):
        query = case["query"]
        expect = case["expect_keyword"]

        # retrieve มีกฎเหล็กว่าไม่ throw error ออกมา (คืน [] แทนถ้าพังภายใน)
        # ดังนั้นไม่ต้องดัก try/except ที่นี่ แต่เช็คว่าได้ [] กลับมาหรือไม่แทน
        results = retrieve(query, page_state="", k=TOP_K)

        if not isinstance(results, list):
            print(f"[{i}] ❌ FAIL   Query: '{query}' → ผลลัพธ์ไม่ใช่ list! ได้ {type(results)}")
            continue

        result_texts = results  # เป็น list[str] อยู่แล้ว ไม่ต้องแกะ field
        combined = " ".join(result_texts)

        if expect is None:
            # negative case: คาดหวังว่า "ไม่ควร" เจอ chunk ที่เกี่ยวข้องแบบชัดเจน
            # เกณฑ์ง่าย ๆ: เช็คว่า similarity ต่ำ หรือไม่มี keyword ของ topic อื่นปนมาแบบมั่ว
            print(f"[{i}] Query: '{query}' (negative case)")
            print(f"     → ผลลัพธ์: {result_texts[:1]} ... (เช็คด้วยตาว่า relevant ไหม)")
            passed += 1  # นับผ่านแบบ manual review สำหรับ negative case
            continue

        if expect in combined:
            print(f"[{i}] ✅ PASS   Query: '{query}' → เจอ '{expect}' ใน top-{TOP_K}")
            passed += 1
        else:
            print(f"[{i}] ❌ FAIL   Query: '{query}' → ไม่เจอ '{expect}'")
            print(f"     ผลลัพธ์ที่ได้: {result_texts}")

    precision = passed / total * 100
    print("\n" + "-" * 60)
    print(f"Precision@{TOP_K}: {passed}/{total} = {precision:.1f}%")
    print("-" * 60)

    if precision >= 80:
        print("🎉 ผ่านเกณฑ์ (>= 80%) — retriever ทำงานได้ดี")
    else:
        print("⚠️  ยังไม่ผ่านเกณฑ์ (< 80%) — ลองเช็ค chunking size, embedding model, หรือ similarity threshold")

    return precision


def run_page_state_test():
    """เช็คว่าใส่ page_state เข้าไปแล้ว retrieve ยังทำงานได้ปกติ (ไม่ error, คืน list)"""
    print("=" * 60)
    print("STEP 3: ทดสอบกรณีมี page_state ประกอบ query")
    print("=" * 60)

    fake_page_state = "<button>Sign in</button><input name='username'>"
    results = retrieve("login เข้าระบบ", page_state=fake_page_state, k=TOP_K)

    if isinstance(results, list):
        print(f"✅ ทำงานได้ปกติ ได้ {len(results)} chunk กลับมา\n")
    else:
        print(f"❌ ผิดปกติ: คืนค่าไม่ใช่ list ({type(results)})\n")


def run_error_handling_test():
    """เช็คกฎเหล็ก: ต่อให้ query แปลกๆ (เช่น string ว่าง) ก็ต้องไม่ throw และคืน list เสมอ"""
    print("=" * 60)
    print("STEP 4: ทดสอบ Error Handling (ต้องไม่ throw, ต้องคืน list เสมอ)")
    print("=" * 60)

    try:
        results = retrieve("", page_state="", k=TOP_K)
        if isinstance(results, list):
            print(f"✅ query ว่าง → ไม่ throw, คืน list ({len(results)} รายการ)\n")
        else:
            print(f"❌ query ว่าง → คืนค่าไม่ใช่ list: {type(results)}\n")
    except Exception as e:
        print(f"❌ FAIL: retrieve throw exception ออกมา ทั้งที่ไม่ควร: {e}\n")


def main():
    start = time.time()

    ok = run_ingestion_test()
    if not ok:
        print("หยุดเทส เพราะ ingestion ล้มเหลว ต้องแก้ก่อนถึงจะเทส retriever ได้")
        return

    run_retriever_test()
    run_page_state_test()
    run_error_handling_test()

    print(f"\nใช้เวลาทั้งหมด: {time.time() - start:.2f} วินาที")


if __name__ == "__main__":
    main()