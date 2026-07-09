"""Perception: หน้าเว็บ -> indexed elements snapshot.

W1: skeleton only. W2: ทำ snapshot จริง (หัวใจของระบบ, ประหยัด token).
"""

from dataclasses import dataclass

from playwright.async_api import Page


@dataclass
class ElementSnapshot:
    index: int
    tag: str
    text: str
    selector: str


class PerceptionEngine:
    async def snapshot(self, page: Page) -> list[ElementSnapshot]:
        raise NotImplementedError("W2: build indexed element snapshot")
