import asyncio
from backend.app.core.actions import execute
async def test():
    # 1. ทดสอบเข้าเว็บโดนแบน
    cmd_goto = {"type": "goto", "url": "https://malicious.com"}
    res1 = await execute(None, cmd_goto)
    print("Test 1 (goto malicious):", res1)  
    # คาดหวังผลลัพธ์: [FAIL] goto -> Action ถูกบล็อกโดยระบบรักษาความปลอดภัย (Blocklist)
    # 2. ทดสอบแอคชันที่ต้องขอยืนยัน
    cmd_submit = {"type": "submit"}
    res2 = await execute(None, cmd_submit)
    print("Test 2 (submit):", res2)
    # คาดหวังผลลัพธ์: [FAIL] submit -> Action นี้ต้องได้รับการยืนยันจากมนุษย์ก่อน (Needs Confirmation)
if __name__ == "__main__":
    asyncio.run(test())