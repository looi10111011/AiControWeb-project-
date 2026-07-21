import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core.browser_pool import BrowserPool


def _patch_playwright(n_launch_calls_expected: int | None = None):
    """mock chain เดียวกับ orchestrator เทสต์: async_playwright() -> .start() ->
    playwright -> .chromium.launch() -> browser (คืน mock browser ใหม่ทุกครั้งที่ถูก
    launch() เพื่อให้แยกแยะได้ว่า pool เปิด instance ต่างกันจริงตามจำนวน size)"""
    launched: list[AsyncMock] = []

    async def _launch(**kwargs):
        b = AsyncMock()
        b.close = AsyncMock()
        launched.append(b)
        return b

    mock_playwright_instance = AsyncMock()
    mock_playwright_instance.chromium.launch = AsyncMock(side_effect=_launch)
    mock_playwright_instance.stop = AsyncMock()

    mock_p_helper = MagicMock()
    mock_p_helper.start = AsyncMock(return_value=mock_playwright_instance)
    mock_async_playwright = MagicMock(return_value=mock_p_helper)

    return mock_async_playwright, mock_playwright_instance, launched


@pytest.mark.asyncio
async def test_start_launches_size_browsers_and_fills_available():
    mock_async_playwright, mock_playwright_ctx, launched = _patch_playwright()
    with patch("backend.app.core.browser_pool.async_playwright", mock_async_playwright):
        pool = BrowserPool(size=3)
        await pool.start()

    assert mock_playwright_ctx.chromium.launch.await_count == 3
    assert len(launched) == 3
    assert pool.size == 3
    assert pool.available == 3


@pytest.mark.asyncio
async def test_start_is_idempotent():
    mock_async_playwright, mock_playwright_ctx, launched = _patch_playwright()
    with patch("backend.app.core.browser_pool.async_playwright", mock_async_playwright):
        pool = BrowserPool(size=2)
        await pool.start()
        await pool.start()  # เรียกซ้ำ ต้อง no-op ไม่ launch เพิ่ม

    assert mock_playwright_ctx.chromium.launch.await_count == 2
    assert pool.available == 2


@pytest.mark.asyncio
async def test_acquire_hands_out_browser_and_returns_it_on_exit():
    mock_async_playwright, mock_playwright_ctx, launched = _patch_playwright()
    with patch("backend.app.core.browser_pool.async_playwright", mock_async_playwright):
        pool = BrowserPool(size=1)
        await pool.start()

        assert pool.available == 1
        async with pool.acquire() as browser:
            assert browser is launched[0]
            assert pool.available == 0  # ยืมไปแล้ว ไม่เหลือให้คนอื่น
        assert pool.available == 1  # คืนกลับหลังออกจาก block


@pytest.mark.asyncio
async def test_acquire_returns_browser_even_if_task_raises():
    mock_async_playwright, mock_playwright_ctx, launched = _patch_playwright()
    with patch("backend.app.core.browser_pool.async_playwright", mock_async_playwright):
        pool = BrowserPool(size=1)
        await pool.start()

        with pytest.raises(RuntimeError):
            async with pool.acquire():
                raise RuntimeError("boom")

        assert pool.available == 1  # ไม่หลุดหายไปจาก pool แม้ task ข้างในพัง


@pytest.mark.asyncio
async def test_acquire_blocks_until_browser_is_released():
    mock_async_playwright, mock_playwright_ctx, launched = _patch_playwright()
    with patch("backend.app.core.browser_pool.async_playwright", mock_async_playwright):
        pool = BrowserPool(size=1)
        await pool.start()

        release_event = asyncio.Event()
        second_acquired = asyncio.Event()

        async def _hold_then_release():
            async with pool.acquire():
                await release_event.wait()

        async def _second_acquirer():
            async with pool.acquire():
                second_acquired.set()

        first = asyncio.create_task(_hold_then_release())
        second = asyncio.create_task(_second_acquirer())
        await asyncio.sleep(0.05)
        assert not second_acquired.is_set()  # ตัวที่สองต้องยังรออยู่ เพราะ pool มีแค่ 1

        release_event.set()
        await asyncio.wait_for(second_acquired.wait(), timeout=1)
        await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_acquire_before_start_raises():
    pool = BrowserPool(size=1)
    with pytest.raises(RuntimeError):
        async with pool.acquire():
            pass


@pytest.mark.asyncio
async def test_shutdown_closes_all_browsers_and_stops_playwright():
    mock_async_playwright, mock_playwright_ctx, launched = _patch_playwright()
    with patch("backend.app.core.browser_pool.async_playwright", mock_async_playwright):
        pool = BrowserPool(size=2)
        await pool.start()
        await pool.shutdown()

    for b in launched:
        b.close.assert_awaited_once()
    mock_playwright_ctx.stop.assert_awaited_once()
    assert pool.available == 0


@pytest.mark.asyncio
async def test_shutdown_before_start_is_noop():
    pool = BrowserPool(size=1)
    await pool.shutdown()  # ไม่ throw แม้ยังไม่เคย start()


# acquire_one()/release_one() — ตัวยืม/คืนแบบดิบๆ ไม่ผ่าน context manager ไว้ให้ resource
# ที่ต้องมีชีวิตอยู่ข้าม request เดียว (session_registry.py::SessionRegistry) ยืมได้
# ยาวๆ ไม่ auto-return


@pytest.mark.asyncio
async def test_acquire_one_and_release_one_round_trip():
    mock_async_playwright, mock_playwright_ctx, launched = _patch_playwright()
    with patch("backend.app.core.browser_pool.async_playwright", mock_async_playwright):
        pool = BrowserPool(size=1)
        await pool.start()

        assert pool.available == 1
        browser = await pool.acquire_one()
        assert browser is launched[0]
        assert pool.available == 0  # ยืมไปแล้ว ไม่ auto-return จนกว่าจะ release_one() เอง

        await pool.release_one(browser)
        assert pool.available == 1


@pytest.mark.asyncio
async def test_acquire_one_before_start_raises():
    pool = BrowserPool(size=1)
    with pytest.raises(RuntimeError):
        await pool.acquire_one()
