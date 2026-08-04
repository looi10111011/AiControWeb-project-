import pytest
from playwright.async_api import async_playwright

from backend.app.core.dom_locator import compute_locator_descriptor, resolve_locator

# เทสต์กลุ่มนี้เปิด chromium จริง (ไม่ mock) เหมือน test_perception.py — เพราะ
# compute_locator_descriptor()/resolve_locator() พึ่ง page.evaluate()/Playwright locator
# API จริงบน DOM จริง mock ยากกว่าเปิด browser เปล่าตรงๆ

_HTML = """
<html><body>
  <button data-testid="save-btn">Save</button>
  <label for="email-input">Email</label>
  <input id="email-input" type="text" />
  <input type="text" placeholder="Search users" />
  <button aria-label="Close dialog"><svg></svg></button>
</body></html>
"""


@pytest.mark.asyncio
async def test_compute_locator_descriptor_data_testid_button():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML)

        descriptor = await compute_locator_descriptor(page, "button[data-testid='save-btn']")

        await browser.close()

    assert descriptor["tag"] == "button"
    assert descriptor["implicit_role"] == "button"
    assert descriptor["accessible_name"] == "Save"
    assert descriptor["data_testid"] == "save-btn"


@pytest.mark.asyncio
async def test_compute_locator_descriptor_labeled_input():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML)

        descriptor = await compute_locator_descriptor(page, "#email-input")

        await browser.close()

    assert descriptor["implicit_role"] == "textbox"
    assert descriptor["accessible_name"] == "Email"  # ดึงจาก <label for="email-input">


@pytest.mark.asyncio
async def test_compute_locator_descriptor_placeholder_only_input():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML)

        descriptor = await compute_locator_descriptor(page, "input[placeholder='Search users']")

        await browser.close()

    assert descriptor["accessible_name"] == "Search users"  # ไม่มี label ผูกไว้ ใช้ placeholder แทน


@pytest.mark.asyncio
async def test_compute_locator_descriptor_icon_only_aria_label_button():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML)

        descriptor = await compute_locator_descriptor(page, "button[aria-label='Close dialog']")

        await browser.close()

    assert descriptor["accessible_name"] == "Close dialog"


@pytest.mark.asyncio
async def test_compute_locator_descriptor_returns_empty_dict_on_missing_element():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML)

        descriptor = await compute_locator_descriptor(page, "#does-not-exist")

        await browser.close()

    assert descriptor == {}


@pytest.mark.asyncio
async def test_resolve_locator_fallback_chain_resolves_each_fixture():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML)

        selectors = [
            "button[data-testid='save-btn']",
            "#email-input",
            "input[placeholder='Search users']",
            "button[aria-label='Close dialog']",
        ]
        for selector in selectors:
            descriptor = await compute_locator_descriptor(page, selector)
            resolved = await resolve_locator(page, descriptor)
            assert resolved is not None, f"expected {selector} to resolve"
            assert await resolved.count() == 1

        await browser.close()


@pytest.mark.asyncio
async def test_resolve_locator_returns_none_when_nothing_matches():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(_HTML)

        bogus = {
            "explicit_role": "",
            "implicit_role": "button",
            "accessible_name": "Nonexistent Button XYZ",
            "data_testid": "",
            "css_fallback": ".nope",
        }
        resolved = await resolve_locator(page, bogus)

        await browser.close()

    assert resolved is None
