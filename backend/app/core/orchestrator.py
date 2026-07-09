"""Agent Loop: Perceive -> Plan -> Act -> Verify.

W1: skeleton only. W4: ทำ loop จริงกับเว็บง่าย 1 หน้า. W5: เพิ่ม verify/retry.
"""

from backend.app.core.actions import BrowserSession
from backend.app.core.memory import ShortTermMemory
from backend.app.core.perception import PerceptionEngine


class Orchestrator:
    def __init__(self):
        self.browser_session = BrowserSession()
        self.perception = PerceptionEngine()
        self.memory = ShortTermMemory()

    async def run_task(self, url: str, goal: str):
        raise NotImplementedError("W4: implement Perceive -> Plan -> Act -> Verify loop")
