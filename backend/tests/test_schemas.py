"""Security (follow-up to SEC audit): attached_file_content_base64 ไม่เคยมี size limit
เลย — เสี่ยง memory-exhaustion/zip-bomb ผ่าน .xlsx/.docx ที่ไม่มี guard เรื่องขนาดหลัง
decompress เลย (ดู schemas.py::_MAX_ATTACHED_FILE_BASE64_CHARS สำหรับเหตุผลเต็ม)"""

import pytest
from pydantic import ValidationError

from backend.app.api.schemas import (
    _MAX_ATTACHED_FILE_BASE64_CHARS,
    CreateTaskRequest,
    ExecutePlanRequest,
    GeneratePlanRequest,
)


def test_create_task_request_rejects_oversized_attached_file():
    oversized = "a" * (_MAX_ATTACHED_FILE_BASE64_CHARS + 1)
    with pytest.raises(ValidationError):
        CreateTaskRequest(url="https://example.com", goal="x", attached_file_content_base64=oversized)


def test_create_task_request_accepts_file_within_limit():
    within_limit = "a" * _MAX_ATTACHED_FILE_BASE64_CHARS
    req = CreateTaskRequest(url="https://example.com", goal="x", attached_file_content_base64=within_limit)
    assert req.attached_file_content_base64 == within_limit


def test_create_task_request_still_accepts_no_attachment():
    """sanity: ไม่แนบไฟล์เลย (None, ปกติที่สุด) ต้องไม่ถูกกระทบจาก max_length เลย"""
    req = CreateTaskRequest(url="https://example.com", goal="x")
    assert req.attached_file_content_base64 is None


def test_generate_plan_request_rejects_oversized_attached_file():
    oversized = "a" * (_MAX_ATTACHED_FILE_BASE64_CHARS + 1)
    with pytest.raises(ValidationError):
        GeneratePlanRequest(url="https://example.com", goal="x", attached_file_content_base64=oversized)


def test_execute_plan_request_rejects_oversized_attached_file():
    oversized = "a" * (_MAX_ATTACHED_FILE_BASE64_CHARS + 1)
    with pytest.raises(ValidationError):
        ExecutePlanRequest(url="https://example.com", goal="x", attached_file_content_base64=oversized)
