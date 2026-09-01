"""W_token_trim scoreboard — assemble the exact per-step LLM input (system prompt + tool
JSON + one built user turn) for a synthetic mid-task step and print its size.

Not a pytest test (no assertions, no fixtures) — a plain script in the `run.py` "prove a
number" spirit. Run before and after each token-trim phase; the raw char delta (and the
char//4 token proxy) is the headline number the plan reports.

    .venv\\Scripts\\python.exe backend/tests/measure_prompt_size.py

Cross-check the proxy against real `data/step_trace.jsonl` `tokens.input` on a live
`python run.py agent` run, and `python run.py kpi` on accumulated `source="api"` rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.core import llm  # noqa: E402


# --- synthetic mid-task step (~step 8, OrangeHRM-style table task) ------------------------

_PAGE_TEXT = "\n".join(
    [
        "[0] link 'Admin' (navigation)",
        "[1] link 'PIM' (navigation)",
        "[2] input(text) 'Username'",
        "[3] div 'User Role: -- Select --'",
        "[4] div 'Status: -- Select --'",
        "[5] button 'Search'",
        "[6] button 'Reset'",
        "[7] button 'Add'",
        *[f"[{8 + i}] row 'user{i} / Employee {i} / ESS / Enabled'  button 'Edit'  button 'Delete'"
          for i in range(40)],
    ]
)

_SITE_MANUAL = (
    "[PRE_LEARNED_MANUAL]\n"
    "Page: Admin > User Management > Users (/web/index.php/admin/viewSystemUsers)\n"
    "- The filter form has: Username (text), User Role (custom dropdown), Status (custom "
    "dropdown), Employee Name (autocomplete). Press the Search button to apply.\n"
    "- Each result row has a pencil (Edit) and a trash (Delete) icon button.\n"
    "- 'Select All' checkbox lives in the table header once results are shown; a 'Delete "
    "Selected' button appears after ticking it.\n"
) * 2

_ACTION_HISTORY = "\n".join(
    f"- step {s}: {{'type': 'click', 'index': 3}} -> [OK] clicked 'User Role: -- Select --'"
    if s % 2 else
    f"- step {s}: {{'type': 'read_page_data', 'query': 'roles in table'}} -> [OK] "
    f"read_page_data -> {json.dumps([{'user': f'user{i}', 'role': 'ESS'} for i in range(40)])}"
    for s in range(1, 8)
)

_MEMORY_CONTEXT = (
    "- {'type': 'select', 'index': 3, 'label': 'ESS'} -> [FAIL] element 3 is not a <select>"
)

_LONG_TERM = (
    "- previously on this site: the User Role dropdown is a custom combobox; click to open, "
    "then click the option labelled exactly 'ESS'."
)

_PLAN_TEXT = (
    "1. Go to the Admin page\n"
    "2. Set the User Role filter to ESS\n"
    "3. Press Search\n"
    "4. Verify every row shows Role = ESS\n"
    "5. Tick Select All and press Delete Selected\n"
    "6. Confirm no ESS rows remain"
)


def _sizes(label: str) -> None:
    tool_json = json.dumps(llm._OPENAI_TOOLS, ensure_ascii=False)
    full_manual, ref_manual = llm.site_manual_blocks(_SITE_MANUAL, "orangehrmlive.com")

    for section_label, sections in (("full (sections=None)", None),
                                    ("{plan, table}", frozenset({"plan", "table"}))):
        system = llm.build_system_prompt(sections)
        # a mid-task step re-references the site manual by id (M3); step 1 sends it in full
        for manual_label, manual_block in (("mid-task step (manual by id)", ref_manual),
                                           ("first step (manual in full)", full_manual)):
            user_turn = llm._build_user_turn_text(
                goal="ไปที่หน้า Admin แล้วลบผู้ใช้ที่ Role=ESS ออกให้หมด",
                page_text=_PAGE_TEXT,
                manual_context="",
                memory_context=_MEMORY_CONTEXT,
                long_term_context=_LONG_TERM,
                site_manual_context=manual_block,
                current_url="https://opensource-demo.orangehrmlive.com/web/index.php/admin/viewSystemUsers",
                action_history_context=_ACTION_HISTORY,
                plan_context=llm_focused_plan(),
            )
            total = len(system) + len(tool_json) + len(user_turn)
            print(f"  [{section_label}] [{manual_label}]")
            print(f"    system prompt : {len(system):>7,} chars  (~{len(system)//4:>6,} tok)")
            print(f"    tool JSON     : {len(tool_json):>7,} chars  (~{len(tool_json)//4:>6,} tok)")
            print(f"    user turn     : {len(user_turn):>7,} chars  (~{len(user_turn)//4:>6,} tok)")
            print(f"    TOTAL / step  : {total:>7,} chars  (~{total//4:>6,} tok)")
            print()


def llm_focused_plan() -> str:
    """_focused_plan_context lives in orchestrator.py; import lazily so this script still
    runs if that import chain is heavier."""
    from backend.app.core.orchestrator import _focused_plan_context

    return _focused_plan_context(_PLAN_TEXT, 2)


if __name__ == "__main__":
    print("=== W_token_trim prompt-size scoreboard ===\n")
    _sizes("current")
    print("Compare this output before vs after each phase. char//4 is a rough token proxy;")
    print("the real number is data/step_trace.jsonl tokens.input on a live run.")
