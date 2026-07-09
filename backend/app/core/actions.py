"""Browser control via Playwright.

W1: skeleton only — launches a browser and confirms Playwright works end-to-end.
W3: implement click/type/scroll/dropdown/tab-switch actions here.
"""

from playwright.async_api import async_playwright, Browser, Page

from backend.app.config import settings


class BrowserSession:
    def __init__(self):
        self._playwright = None
        self.browser: Browser | None = None
        self.page: Page | None = None

    async def start(self) -> Page:
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(
            headless=settings.browser_headless
        )
        self.page = await self.browser.new_page()
        return self.page

    async def stop(self):
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()
