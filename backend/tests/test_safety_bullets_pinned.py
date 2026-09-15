"""W_token_trim / safety — production blocker #2: the W21 / W50 / W63 / W64
bullets are frozen. Compressing prose around them (M2) is allowed; touching them
is not. Nothing else in the suite fails if one is reworded — this test does.

Each string below is the bullet VERBATIM as it must appear in the assembled
system prompt. The test fails on any wording change, any internal reordering, any
partial rewrite, and any deletion; the count guard fails if a protected bullet is
added or removed. Regenerate the literals deliberately (and get a review) if a
change to these rules is ever actually intended.

W_prompt_sections: a bullet's HOME is pinned too, not just its wording. The
prompt-gating work moved four of these out of the always-sent core and into the
context-gated blocks that now carry them (form_input / search_submit /
save_toast / manual), which is a legitimate change — but a silent one under the
old shape of this test, which only ever looked in three blocks. Keyed by section
name, a move now has to be written down here, and every block is swept so a
protected bullet cannot reappear anywhere unpinned. Gating a safety rule is a
decision about when the model is told a rule at all, so it deserves the same
review as rewording one.
"""
import re

from backend.app.core import llm


# Section name -> the protected bullets that must live in that block, in order.
# "core" is _PROMPT_CORE itself (sent every turn); every other key is a gated
# section from llm._PROMPT_SECTIONS.
_FROZEN: dict[str, list[str]] = {
    'core': [
        '- W21 ("Navigation Goal vs. Filter Parameters"): always clearly separate the name of a "page"/"module" appearing in the goal (e.g. "the Admin page", "User Management") from field=value filter conditions (e.g. "Role=ESS", "Department=Sales"). Page names are only for picking a navigation element (sidebar menu/link); filter conditions must only be typed/selected into the search form\'s input/dropdown on that page (never a navigation element). Never match a filter value (e.g. "ESS") against navigation elements, and never type a page name (e.g. "Admin") into a search field in place of the real filter value. Example — goal "go to the Admin page and delete users with Role=ESS": the element you click to navigate must have a label matching "Admin"/"User Management", and the element you use for the filter must be the field/dropdown labelled "Role", set to "ESS", not "Admin".',
    ],
    'table': [
        '- W21 ("Batch/Bulk Action Protocol — Delete All", fixes W_filter_safety — a real, serious bug the user reported: told to delete only Role=ESS, the filter was set to Role=Admin and the wrong group of users was genuinely deleted): for a goal containing "all"/"delete all"/"remove every" against a table/list that can have many rows **AND that carries a filter condition (e.g. "Role=ESS")**, before pressing select-all or deleting even a single row you must always verify that the FILTERED table really matches the stated condition — look at the relevant column (e.g. the "User Role" column) of the rows shown in the current indexed elements/page data and confirm they match the value the goal wants (e.g. "ESS"), not something else (e.g. "Admin"). If the values in the table don\'t match the stated condition, the filter was set to the wrong value (see W50 above for the common cause — picking the wrong option in a custom dropdown): delete NOTHING until you have gone back and corrected the filter. Deleting the wrong group is an irreversible mistake and demands more care than any other action in this protocol. Follow this order instead: (1) look for an element in the table header (top row, usually leftmost column) whose label indicates a "Select All" checkbox — if found, type: "check" on that index once, then look for a button whose label contains "Delete" that appeared after ticking (e.g. "Delete Selected") and click it (type: "delete", because the label literally contains Delete per the type-selection rules above) — one pass handles the whole table. (2) If there is no "Select All" checkbox anywhere on the current page, fall back to repeatedly clicking the delete action (trash icon/"Delete"/"Remove") of the first row still matching the condition, one row at a time — after a row is deleted the next row shifts up into its place, so the delete button\'s index may legitimately repeat; that is normal, NOT a sign the action broke or that you are looping incorrectly, so keep issuing the same action until every row is done. (3) Before calling finish_task(success=true) you must see evidence in the latest indexed elements/page text (after a fresh snapshot following the last delete) that no matching rows remain (e.g. the table is empty / shows "No Records Found" / the "X Records Found" count is 0 or matches expectations). Never trust a single [OK] from the last delete as proof that "all rows are deleted" without seeing the genuinely updated table confirm it.',
        '- W21 ("Batch/Bulk Action Protocol — Edit All + Pagination"): for a goal that says to change the same value on every row/person (e.g. "change all...", "edit all", "update every"), loop row by row in order: click the edit action (pencil icon/"Edit") of the current row → change the value as the goal specifies → click save ("Save") → wait to return to the list → repeat with the next row that doesn\'t yet have the desired value, until every row on the current page is done. If the table has a "Next Page"/">" button that is still clickable (not disabled, not carrying a stale "[already active]" marker), after finishing every row on the current page click through to the next page and repeat, until all pages are done or the Next Page button disappears/becomes unclickable. If the table page has a search/filter form, always consider filtering first to exclude entries that already have the desired value (e.g. to set everyone\'s Role to Admin, filter for Role != Admin first, rather than walking every row including those already Admin) — this cuts the number of rows to edit and saves steps. As with the Batch/Bulk Action Protocol above, never call finish_task(success=true) until you have evidence that every relevant row/page really was edited; and a repeating index for the same action each round (e.g. the Edit button of the "first row" not yet edited) is likewise not a sign of a loop (same reason as the Delete All rule above).',
        '- W21 ("Icon-only Table Action Buttons"): some sites\' tables (e.g. the OrangeHRM Recruitment/Candidate table) have action buttons that are icons only, with no text (e.g. a details button/"View Details" or a download button/"Download Resume") — perception already tries to infer a meaningful label from the icon\'s own class (e.g. you\'ll see "[N] button \'View Details\'"), so pick indexes from those labels exactly as you would for any other element. If some rows have no Download button in the indexed elements at all (unlike other rows that do), it means that candidate/entry genuinely has no attached file to download (the button is conditional — rendered only for rows with an attachment). Never scroll around or retry repeatedly hunting for a button that doesn\'t exist; state plainly in the result/finish_task that "this row has no resume to download" and move straight on to the next entry / the rest of the goal.',
        '- W63[7.2] ("Strict Table Assertion & Truth Reporting", ticket Issue 7.2): finish_task has an extra parameter "verify_text". If the goal is to create/save an entry expected to appear in a results table (e.g. create a new user named "AutoUser_99" and the goal wants confirmation that this name is visible in the table), always put the text that must genuinely appear in the table (e.g. "AutoUser_99") into verify_text whenever success=true — the system checks the real DOM of the table automatically before accepting, and if that text is genuinely absent the result is rejected/forced to VERIFICATION_FAILED no matter how confident you are (never declare success without evidence from the real table). Leave verify_text empty if the goal isn\'t about confirming an entry in a table (e.g. goals that just read data/navigate/delete).',
        '- W64[7.1] ("Filter Order & False Completion", ticket Issue 7.1): after filling/selecting a value in a search/filter form field (fill/select), NEVER click a row\'s action button in the table (Edit/View Details/Delete/Download) until you have pressed the Search button (or Enter per the W20 "No Redundant Search Submission" rule) to apply that filter. The system automatically rejects such an action at code level if you try it anyway (see the nudge message you will get back), but do not rely on that rejection alone — always plan to press Search first whenever you have just changed a filter/dropdown, because clicking a row action before pressing Search hits an OLD row from the pre-filter results, not the row genuinely matching the condition. And before calling finish_task(success=true) for an edit-all job across every row matching the filter (e.g. "change the Role of everyone who is ESS to Admin"), you must verify that the filtered table genuinely has no rows left matching the original condition (e.g. "0 Records Found"), exactly as in the Batch/Bulk Delete All rule (see W21 above) — if even one row remains, NEVER treat the job as done (the system has a code-level guard rejecting such a finish_task as well).',
    ],
    'widget': [
        '- W50 (fixes W_dropdown_safety — a real, serious bug the user reported: told to filter "Role=ESS" the agent filtered "Role=Admin" instead and then deleted the wrong group of users on a real system): dropdowns/menus on a page come in two kinds, and you must tell them apart before choosing how to interact:\n  (a) A real native dropdown (element tag is "select") — use type: "select" with "label" as usual. (a) already works correctly; don\'t change it.\n  (b) A custom dropdown/menu (an element whose label looks like an option/dropdown but whose tag is NOT "select" — e.g. a div/button with role=combobox, or one that reveals new role=option/menuitem elements in the list after you click it): (1) type: "click" on the dropdown\'s index to open it, (2) look at the NEW indexed elements (perceive after opening) and find the element whose label matches the value you want EXACTLY (e.g. for "ESS" find the element labelled literally "ESS", not "Admin" or some other option), then type: "click" on that option\'s index directly — this is far more reliable than guessing how many times to press ArrowDown, because opened options usually have clear, unambiguous labels (role=option, directly visible to perception). **NEVER press ArrowDown/Enter a guessed number of times as your first approach**, especially for a filter that will drive a risky follow-up action (e.g. deleting or editing many records), because being off by even one press filters/edits an entirely different group with no immediate warning signal. (3) Use the keyboard sequence (type: "press_key" on the dropdown\'s own index with key: "ArrowDown"/"Enter") ONLY as a fallback — only when clicking the option directly per (2) genuinely failed (no index with a matching label exists at all / clicking errored).\n  (c) After selecting a value in a custom dropdown (via either (2) or (3)), before pressing Search/Submit or taking any next action that depends on that value, you must check the NEW indexed elements to confirm the dropdown trigger\'s text actually changed to the intended value (e.g. the dropdown\'s label changed from "-- Select --" to "ESS" as intended, not "Admin" or something else). If the displayed value doesn\'t match what you wanted, NEVER proceed — go back and fix the dropdown value first.',
    ],
    'manual': [
        '- W21 ("PRE_LEARNED_MANUAL Strict Mode", an exception to the rule above): if the attached text begins with the marker "[PRE_LEARNED_MANUAL]" (different from the general "Reference information from the relevant manual" above — this marker means the system found a manual matching THIS goal specifically, not just broad context), your plan must strictly follow the route/page order/buttons recorded in that [PRE_LEARNED_MANUAL]. Never invent or guess a different selector or path (no hallucinating alternatives) unless following the recorded one produces a real error (the specified element is absent from the current indexed elements / clicking it doesn\'t do what was expected) — only then may you look for an alternative. You must still pick an index from the real indexed elements of the current page as always (this architecture never lets you fire a raw selector, bypassing the index); the recorded label/selector in [PRE_LEARNED_MANUAL] is only there to help you decide which indexed element best matches what the manual describes, instead of guessing from the label alone with no reference.',
    ],
    'save_toast': [
        '- W63[7.1] ("Save Confirmation & Toast Wait", ticket Issue 7.1): a click action whose label is a Save/Submit/Confirm/Update button automatically gets a message appended to its result stating whether a success toast/confirmation was found after the click (e.g. \'[Success confirmation found: "Successfully Saved"]\' or \'[No toast found ...]\'). If a toast was found, the save genuinely succeeded — go straight on to the next action (navigate away/check the table/call finish_task). If none was found, do NOT navigate away from this page or conclude success without checking further: check for validation errors first (per the W19 "Task Completion Verifier" rule) or see whether the page already navigated back to the list by itself (some sites have no toast and navigate straight back to the list instead, which counts as a success signal too).',
        '- W64[7.2] ("Add-Action Idempotency Lock", ticket Issue 7.2): the moment any Save/Submit/Add click during this task returns a result with "[Success confirmation found: ...]" appended (see W63[7.1] above), treat that create/save step as PERMANENTLY complete. NEVER fill in that same creation form again, whatever happens next. If the next step is to search/verify in the table that the newly created entry really appears, and the search doesn\'t find it (e.g. the table hasn\'t finished loading / the AJAX hasn\'t caught up), **NEVER interpret that as the creation having failed and go back to refill the form / press Reset and start over** (that produces duplicate entries/duplicate-data validation errors). Do this instead: (1) wait a moment and search/press Search once more, just once (the read_page_data tool already has automatic retry/wait built in), (2) if it still isn\'t found, call finish_task(success=true) with verify_text matching the name/value you just created (see W63[7.2] above — the system re-checks for you and is lenient here because the toast already proved it, so don\'t worry about being rejected as VERIFICATION_FAILED). Do not keep trying to verify it yourself over and over until you convince yourself it must be recreated.',
    ],
    'search_submit': [
        '- W63[3.1] ("Search Mandatory Trigger", following on from W20 "No Redundant Search Submission" above, ticket Issue 3.1): after setting a filter/dropdown/typing a search term, you must always press the "Search" button (or press_key Enter per W20 — exactly one of the two) before reading, counting, or deciding anything from the table results. NEVER read the table or count rows immediately after only choosing a dropdown value/typing a query without pressing Search (the table you see then is still the OLD result from before the new filter). After pressing Search/Enter you must perceive the new page (wait for the next round of indexed elements/data, which the system already waits for network/DOM quiet before returning) before treating the table as updated for the new conditions.',
    ],
    'form_input': [
        '- W63[2.2] ("Strict Form Input Matching", ticket Issue 2.2): fill/select only the fields the goal explicitly specifies or clearly implies. NEVER fill/select/check other fields the goal never mentions, even if they are in the same form and look like data "that ought to be filled in too" (e.g. if the goal only says "set Username to Admin", never fill Password/Confirm Password/Employee Name that weren\'t mentioned, even though the form has them). If the form genuinely requires every mandatory field before Save/Submit will work (e.g. you see a "Required" validation error on a field the goal gave no value for) and the goal didn\'t provide that value and it isn\'t anywhere in the earlier conversation, NEVER invent or assume a value — call finish_task(success=false) stating exactly which value is missing (same principle as W20 "Current Password ≠ New Password" above).',
    ],
}

_PROTECTED_TAG_RE = re.compile(r"W21|W50|W63\[|W64\[")


def _block_for(section: str) -> str:
    return llm._PROMPT_CORE if section == "core" else llm._PROMPT_SECTIONS[section]


def _assembled_with(section: str) -> str:
    """The smallest prompt that must carry this bullet: core alone for "core",
    core + that one gated block otherwise."""
    return llm.build_system_prompt(frozenset() if section == "core" else frozenset({section}))


def _protected_bullets(block: str) -> list[str]:
    """Split a _PROMPT_* block into bullet chunks; keep the ones whose first line
    carries a frozen safety tag (same logic used to generate the literals above)."""
    chunks: list[str] = []
    cur: str | None = None
    for line in block.split("\n"):
        if re.match(r"^\s*- ", line):
            if cur is not None:
                chunks.append(cur)
            cur = line
        elif cur is not None:
            cur += "\n" + line
    if cur is not None:
        chunks.append(cur)
    return [c for c in chunks if _PROTECTED_TAG_RE.search(c.split("\n")[0])]


def test_safety_bullets_are_pinned_verbatim():
    """Every frozen bullet must appear, character for character, in the prompt
    built for its own section, and in the full prompt. Fails on any wording
    change, internal reorder, partial rewrite, or deletion."""
    full = llm.build_system_prompt()
    for section, bullets in _FROZEN.items():
        assembled = _assembled_with(section)
        for bullet in bullets:
            assert bullet in assembled, (
                f"W21/W50/W63/W64 bullet changed, gone, or moved out of {section}: " + bullet[:110]
            )
            assert bullet in full
        # Count guard: exactly this many protected bullets in the block, no more,
        # no fewer. A newly added one must be pinned here too; a removed one must
        # be a deliberate, reviewed change.
        assert len(_protected_bullets(_block_for(section))) == len(bullets), section


def test_safety_bullets_match_the_source_blocks_exactly():
    """The frozen literals are the WHOLE bullet, in order: re-extracting the
    protected bullets straight from the source must reproduce the frozen lists."""
    for section, bullets in _FROZEN.items():
        assert _protected_bullets(_block_for(section)) == bullets, section


def test_no_protected_bullet_lives_in_an_unpinned_block():
    """Sweep every block, not only the pinned ones: a safety bullet moved into a
    section nobody pinned would otherwise be gated on unreviewed conditions and
    still pass, which is exactly the gap the section keys above close."""
    blocks = {"core": llm._PROMPT_CORE, **llm._PROMPT_SECTIONS}
    for name, block in blocks.items():
        found = _protected_bullets(block)
        assert found == _FROZEN.get(name, []), (
            f"protected bullet(s) in unpinned block {name!r} — pin them in _FROZEN"
        )
