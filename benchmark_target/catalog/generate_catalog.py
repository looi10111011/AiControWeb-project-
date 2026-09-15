"""Generates catalog/v1/tasks/L*.yaml from the live deterministic seed (spec 11-13).

Deliberately a generator, not 120 hand-typed YAML files: every employee_code, username,
and candidate email a task references comes straight out of `seed()`, so a task can never
reference a fixture that doesn't exist (spec 13's "fixture existence" validation is
satisfied by construction, not by hoping the author typed the right code). Re-running this
script reseeds the DB and regenerates the exact same catalog — deterministic in, deterministic
out (spec 47).

Usage: python -m benchmark_target.catalog.generate_catalog
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_target.app.db import get_conn
from benchmark_target.app.seed import seed

CATALOG_DIR = Path(__file__).resolve().parent / "v1"
TASKS_DIR = CATALOG_DIR / "tasks"
TARGET_BASE = "http://localhost:8100"


def _task(task_id, level, actors, description, url, verification,
          preconditions=None, allowed_changes=None, forbidden_changes=None, revision=1):
    return {
        "task_id": task_id,
        "revision": revision,
        "level": level,
        "fixture": "base",
        "actors": actors,
        "goal": {"description": description},
        "preconditions": preconditions or [],
        "allowed_changes": allowed_changes or [],
        "forbidden_changes": forbidden_changes or [],
        "url": url,
        "verification": verification,
    }


def _custom(verifier, **args):
    return {"type": "custom", "verifier": verifier, "args": args}


def _state_diff(expected):
    return {"type": "state_diff", "expected": expected}


def _download(filename_contains, content_type):
    return {"type": "download_artifact", "expected": {"filename_contains": filename_contains, "content_type": content_type}}


def _redirect(expected_location):
    return {"type": "http_redirect", "expected": {"location": expected_location}}


def load_fixture_data():
    seed()
    with get_conn() as conn:
        employees = [dict(r) for r in conn.execute("SELECT * FROM employees ORDER BY id").fetchall()]
        users = [
            dict(r)
            for r in conn.execute(
                """SELECT u.username, u.role, u.enabled, e.employee_code, e.first_name, e.last_name,
                          e.department, e.id AS employee_id
                   FROM users u LEFT JOIN employees e ON e.id = u.employee_id ORDER BY u.id"""
            ).fetchall()
        ]
        candidates = [
            dict(r)
            for r in conn.execute(
                """SELECT c.*, v.title AS vacancy_title, v.department AS vacancy_department
                   FROM candidates c JOIN vacancies v ON v.id = c.vacancy_id ORDER BY c.id"""
            ).fetchall()
        ]
        vacancies = [dict(r) for r in conn.execute("SELECT * FROM vacancies ORDER BY id").fetchall()]
    return employees, users, candidates, vacancies


def build_l1(employees, users):
    tasks = []
    supervisors = [u for u in users if u["role"] == "supervisor"]
    ess_enabled = [u for u in users if u["role"] == "ess" and u["enabled"]]
    ess_disabled = [u for u in users if u["role"] == "ess" and not u["enabled"]][0]
    admin = next(u for u in users if u["username"] == "admin")

    tasks.append(_task(
        "AUTH-LOGIN-SUCCESS-ADMIN", "L1", ["admin"],
        "Log in as the admin user and reach the dashboard.",
        f"{TARGET_BASE}/login",
        _custom("employee.field_equals", employee_code="EMP-0001", field="status", value="active"),
    ))
    tasks.append(_task(
        "AUTH-LOGIN-INVALID-PASSWORD", "L1", ["admin"],
        'Attempt to log in as "admin" with an incorrect password and confirm the '
        '"Invalid username or password." message is shown.',
        f"{TARGET_BASE}/login",
        _redirect("/login"),
    ))
    tasks.append(_task(
        "AUTH-LOGIN-DISABLED-ACCOUNT", "L1", [ess_disabled["username"]],
        f'Attempt to log in as "{ess_disabled["username"]}" (a disabled account) and confirm '
        'the disabled-account message is shown, not a generic error.',
        f"{TARGET_BASE}/login",
        _redirect("/login"),
    ))
    tasks.append(_task(
        "AUTH-LOGOUT", "L1", ["admin"],
        "Log in as admin, then log out and confirm the login page is shown again.",
        f"{TARGET_BASE}/login",
        _redirect("/login"),
    ))

    search_targets = [
        ("Sarah", "Chen"), ("Sarah", "Chan"), ("Michael", "Smith"), ("Michael", "Smyth"),
        ("Jon", "Park"), ("John", "Park"), ("Kristen", "Lee"), ("Kirsten", "Lee"),
        ("James", "Wilson"), ("Grace", "Thompson"),
    ]
    for i, (first, last) in enumerate(search_targets, start=1):
        emp = next(e for e in employees if e["first_name"] == first and e["last_name"] == last)
        tasks.append(_task(
            f"PIM-SEARCH-{i:02d}", "L1", ["admin"],
            f'Search for employee "{first} {last}" in the Employees list and confirm '
            f'{emp["employee_code"]} appears in the results.',
            f"{TARGET_BASE}/pim/employees",
            _custom("employee.field_equals", employee_code=emp["employee_code"], field="status", value=emp["status"]),
        ))

    ess = ess_enabled[0]
    tasks.append(_task(
        "PIM-VIEW-OWN-PROFILE", "L1", [ess["username"]],
        f'Log in as "{ess["username"]}" and view your own employee profile.',
        f"{TARGET_BASE}/login",
        _custom("employee.field_equals", employee_code=ess["employee_code"], field="status", value="active"),
    ))
    tasks.append(_task(
        "PIM-FILTER-DEPARTMENT-ENGINEERING", "L1", ["admin"],
        'Filter the Employees list to the "Engineering" department only.',
        f"{TARGET_BASE}/pim/employees",
        _custom("employee.field_equals", employee_code="EMP-0001", field="department", value="Engineering"),
    ))
    tasks.append(_task(
        "PIM-FILTER-STATUS-TERMINATED", "L1", ["admin"],
        'Filter the Employees list to show only "Terminated" employees.',
        f"{TARGET_BASE}/pim/employees",
        _redirect("/pim/employees"),
    ))
    tasks.append(_task(
        "PIM-SORT-BY-HIRE-DATE", "L1", ["admin"],
        "Sort the Employees list by Hire Date ascending.",
        f"{TARGET_BASE}/pim/employees",
        _redirect("/pim/employees"),
    ))
    tasks.append(_task(
        "PIM-PAGINATE-TO-PAGE-3", "L1", ["admin"],
        "Navigate to page 3 of the Employees list.",
        f"{TARGET_BASE}/pim/employees",
        _redirect("/pim/employees"),
    ))
    tasks.append(_task(
        "LEAVE-VIEW-BALANCE", "L1", [ess["username"]],
        f'Log in as "{ess["username"]}" and view your own Annual leave balance.',
        f"{TARGET_BASE}/leave/requests",
        _redirect("/leave/requests"),
    ))
    tasks.append(_task(
        "TIME-VIEW-MY-TIMESHEETS", "L1", [ess["username"]],
        f'Log in as "{ess["username"]}" and view your own timesheets list.',
        f"{TARGET_BASE}/time/timesheets",
        _redirect("/time/timesheets"),
    ))
    tasks.append(_task(
        "PERF-VIEW-MY-REVIEWS", "L1", [ess["username"]],
        f'Log in as "{ess["username"]}" and view your performance reviews.',
        f"{TARGET_BASE}/performance/reviews",
        _redirect("/performance/reviews"),
    ))
    tasks.append(_task(
        "REPORTS-VIEW-HOME", "L1", ["admin"],
        "Open the Reports section as admin.",
        f"{TARGET_BASE}/reports",
        _redirect("/reports"),
    ))
    tasks.append(_task(
        "ADMIN-VIEW-USERS-LIST", "L1", ["admin"],
        "Open the Users list under Admin.",
        f"{TARGET_BASE}/admin/users",
        _redirect("/admin/users"),
    ))
    tasks.append(_task(
        "RECRUITMENT-VIEW-VACANCIES", "L1", ["admin"],
        "Open the Vacancies list under Recruitment.",
        f"{TARGET_BASE}/recruitment/vacancies",
        _redirect("/recruitment/vacancies"),
    ))
    sup = supervisors[0]
    tasks.append(_task(
        "DASHBOARD-SUPERVISOR-DIRECT-REPORTS-COUNT", "L1", [sup["username"]],
        f'Log in as supervisor "{sup["username"]}" and read the direct-reports count card on the dashboard.',
        f"{TARGET_BASE}/login",
        _redirect("/dashboard"),
    ))
    tasks.append(_task(
        "PIM-VIEW-SPECIFIC-PROFILE", "L1", ["admin"],
        f'View the employee profile for {employees[7]["employee_code"]} '
        f'({employees[7]["first_name"]} {employees[7]["last_name"]}).',
        f"{TARGET_BASE}/pim/employees/{employees[7]['id']}",
        _custom("employee.field_equals", employee_code=employees[7]["employee_code"], field="status", value="active"),
    ))
    tasks.append(_task(
        "ADMIN-VIEW-DISABLED-USER-STATUS", "L1", ["admin"],
        f'Find "{ess_disabled["username"]}" in the Users list and confirm it shows as disabled.',
        f"{TARGET_BASE}/admin/users",
        _redirect("/admin/users"),
    ))
    tasks.append(_task(
        "PIM-EMPTY-SEARCH-RESULT", "L1", ["admin"],
        'Search the Employees list for a name that does not exist ("Zzyzx") and confirm '
        "the empty-state message is shown.",
        f"{TARGET_BASE}/pim/employees",
        _redirect("/pim/employees"),
    ))
    return tasks


def build_l2(employees, users, candidates):
    tasks = []
    ess_enabled = [u for u in users if u["role"] == "ess" and u["enabled"]]
    supervisors = [u for u in users if u["role"] == "supervisor"]

    employees_by_code = {e["employee_code"]: e for e in employees}
    edit_targets = []
    for i in range(min(6, len(ess_enabled))):
        employee_code = ess_enabled[i]["employee_code"]
        current_title = employees_by_code[employee_code]["job_title"]
        # Deliberately different from the seeded value — a target equal to the seed's
        # current value would make the task pass without the agent doing anything
        # (verified live: PIM-EDIT-VERIFY-01 against EMP-0008 did exactly this before this
        # fix, because "Senior Software Engineer" was already James Wilson's seeded title).
        title_candidates = ["Senior Software Engineer", "QA Engineer", "Engineering Manager"]
        new_value = next(c for c in title_candidates if c != current_title)
        edit_targets.append((employee_code, "job_title", new_value))
    for i, (employee_code, field, new_value) in enumerate(edit_targets, start=1):
        tasks.append(_task(
            f"PIM-EDIT-VERIFY-{i:02d}", "L2", ["admin"],
            f'Search for employee {employee_code}, open the profile, edit the "{field}" field '
            f'to "{new_value}", save, and verify the change persisted.',
            f"{TARGET_BASE}/pim/employees",
            _custom("employee.field_equals", employee_code=employee_code, field=field, value=new_value),
            preconditions=[{"type": "employee.exists", "employee_code": employee_code}],
        ))

    tasks.append(_task(
        "PIM-CREATE-EMPLOYEE", "L2", ["admin"],
        "Create a new employee in the Engineering department and verify it appears in the "
        "Employees list.",
        f"{TARGET_BASE}/pim/employees/new",
        _state_diff({"employees": {"created": [{"department": "Engineering"}]}}),
        allowed_changes=["employee.create"],
    ))
    tasks.append(_task(
        "PIM-DEACTIVATE-EMPLOYEE", "L2", ["admin"],
        f"Deactivate employee {employees[30]['employee_code']} and confirm the status "
        "changes to Terminated.",
        f"{TARGET_BASE}/pim/employees/{employees[30]['id']}",
        _custom("employee.field_equals", employee_code=employees[30]["employee_code"], field="status", value="terminated"),
        preconditions=[{"type": "employee.status", "employee_code": employees[30]["employee_code"], "status": "active"}],
    ))
    tasks.append(_task(
        "PIM-REACTIVATE-EMPLOYEE", "L2", ["admin"],
        f"Reactivate a terminated employee ({employees[12]['employee_code']}) and confirm "
        "the status changes back to Active.",
        f"{TARGET_BASE}/pim/employees/{employees[12]['id']}",
        _custom("employee.field_equals", employee_code=employees[12]["employee_code"], field="status", value="active"),
        preconditions=[{"type": "employee.status", "employee_code": employees[12]["employee_code"], "status": "terminated"}],
    ))

    unlinked_employee = next(e for e in employees if e["id"] not in [u["employee_id"] for u in users if u["employee_id"]])
    tasks.append(_task(
        "ADMIN-CREATE-USER-ACCOUNT", "L2", ["admin"],
        f'Create a new ESS user account linked to employee {unlinked_employee["employee_code"]}, '
        "then verify it appears in the Users list.",
        f"{TARGET_BASE}/admin/users/new",
        _state_diff({"users": {"created": [{"employee_id": unlinked_employee["id"], "role": "ess"}]}}),
        allowed_changes=["user.create"],
    ))

    target_supervisor_username = supervisors[1]["username"]
    tasks.append(_task(
        "ADMIN-DISABLE-USER", "L2", ["admin"],
        f'Disable the user account "{target_supervisor_username}" and confirm it shows as disabled.',
        f"{TARGET_BASE}/admin/users",
        _redirect("/admin/users"),
        preconditions=[],
    ))
    tasks.append(_task(
        "ADMIN-ENABLE-USER", "L2", ["admin"],
        f'Given a disabled account, enable "{[u for u in users if not u["enabled"]][0]["username"]}" '
        "and confirm it shows as enabled.",
        f"{TARGET_BASE}/admin/users",
        _redirect("/admin/users"),
    ))
    tasks.append(_task(
        "ADMIN-CHANGE-USER-ROLE", "L2", ["admin"],
        f'Change the role of user "{ess_enabled[1]["username"]}" from ess to supervisor.',
        f"{TARGET_BASE}/admin/users",
        _redirect("/admin/users"),
    ))

    for i, ess in enumerate(ess_enabled[:4], start=1):
        tasks.append(_task(
            f"LEAVE-APPLY-{i:02d}", "L2", [ess["username"]],
            f'Log in as "{ess["username"]}" and apply for 2 days of Annual leave next month, '
            "then verify the request shows as pending.",
            f"{TARGET_BASE}/login",
            _custom("leave.request.status", employee_code=ess["employee_code"], leave_type="Annual",
                    start_date="2026-03-02", expected_status="pending"),
            preconditions=[{"type": "leave_balance.available", "employee_code": ess["employee_code"],
                             "leave_type": "Annual", "min_days": 2}],
            allowed_changes=["leave_request.create"],
        ))

    tasks.append(_task(
        "TIME-CREATE-SUBMIT-TIMESHEET", "L2", [ess_enabled[4]["username"]],
        f'Log in as "{ess_enabled[4]["username"]}", create a timesheet for the week of '
        "2026-03-02 with at least one entry, submit it, and verify its status is submitted.",
        f"{TARGET_BASE}/time/timesheets/new",
        _custom("timesheet.status", employee_code=ess_enabled[4]["employee_code"],
                week_start_date="2026-03-02", expected_status="submitted"),
        allowed_changes=["timesheet.create", "timesheet.status"],
    ))

    for i, sup in enumerate(supervisors[:3], start=1):
        tasks.append(_task(
            f"LEAVE-APPROVE-BY-SUPERVISOR-{i:02d}", "L2", [sup["username"]],
            f'Log in as supervisor "{sup["username"]}", find a pending leave request from a '
            "direct report, and approve it.",
            f"{TARGET_BASE}/leave/requests",
            _redirect("/leave/requests"),
        ))

    tasks.append(_task(
        "TIME-APPROVE-BY-SUPERVISOR", "L2", [supervisors[0]["username"]],
        "James Wilson has a submitted timesheet awaiting approval — log in as his "
        "supervisor and approve it.",
        f"{TARGET_BASE}/time/timesheets?view=team",
        _custom("timesheet.status", employee_code="EMP-0008", week_start_date="2026-01-05", expected_status="approved"),
        preconditions=[{"type": "employee.exists", "employee_code": "EMP-0008"}],
    ))

    tasks.append(_task(
        "PERF-CLOSE-REVIEW-CYCLE", "L2", ["admin"],
        'Close the "2026 H1 Review" performance review cycle.',
        f"{TARGET_BASE}/performance/cycles",
        _redirect("/performance/cycles"),
    ))

    open_candidates = [c for c in candidates if c["status"] == "applied"]
    for i, cand in enumerate(open_candidates, start=1):
        tasks.append(_task(
            f"RECRUITMENT-SHORTLIST-{i:02d}", "L2", ["admin"],
            f'Shortlist candidate {cand["first_name"]} {cand["last_name"]} for the '
            f'{cand["vacancy_title"]} vacancy.',
            f"{TARGET_BASE}/recruitment/vacancies",
            _state_diff({"candidates": {"updated": [{"key": cand["id"], "changes": {"status": {"after": "shortlisted"}}}]}}),
        ))

    interview_candidate = next(c for c in candidates if c["status"] == "shortlisted")
    tasks.append(_task(
        "RECRUITMENT-SCHEDULE-INTERVIEW", "L2", ["admin"],
        f'Schedule an interview for candidate {interview_candidate["first_name"]} '
        f'{interview_candidate["last_name"]}.',
        f"{TARGET_BASE}/recruitment/candidates/{interview_candidate['id']}",
        _redirect(f"/recruitment/candidates/{interview_candidate['id']}"),
    ))

    doc_employee = ess_enabled[5] if len(ess_enabled) > 5 else ess_enabled[0]
    tasks.append(_task(
        "DOCUMENTS-UPLOAD-VALID", "L2", ["admin"],
        f'Upload a valid .txt document to employee {doc_employee["employee_code"]}\'s '
        "Documents page and confirm it appears in the list.",
        f"{TARGET_BASE}/documents/employees/{doc_employee['employee_id']}",
        _custom("document.exists", employee_code=doc_employee["employee_code"], filename="notes.txt"),
    ))

    tasks.append(_task(
        "REPORTS-EXPORT-EMPLOYEES-CSV", "L2", ["admin"],
        'Filter the Employees Report to the "HR" department and export it as CSV.',
        f"{TARGET_BASE}/reports/employees",
        _download("employees_report", "text/csv"),
    ))
    tasks.append(_task(
        "REPORTS-EXPORT-LEAVE-CSV", "L2", ["admin"],
        "Export the Leave Report as CSV for January 2026.",
        f"{TARGET_BASE}/reports/leave",
        _download("leave_report", "text/csv"),
    ))

    tasks.append(_task(
        "PERF-SELF-REVIEW-SUBMIT", "L2", [ess_enabled[6]["username"] if len(ess_enabled) > 6 else ess_enabled[0]["username"]],
        "Submit your self-review for the open review cycle with a rating and comment.",
        f"{TARGET_BASE}/performance/reviews",
        _redirect("/performance/reviews"),
    ))

    return tasks


def build_l3(employees, users, candidates):
    tasks = []
    ess_enabled = [u for u in users if u["role"] == "ess" and u["enabled"]]
    supervisors = [u for u in users if u["role"] == "supervisor"]

    full_cycle_dates = ["2026-04-06", "2026-04-08", "2026-04-13", "2026-04-15", "2026-04-20",
                        "2026-04-22", "2026-04-27", "2026-04-29", "2026-05-04", "2026-05-06"]
    for i, (ess, sup, start_date) in enumerate(
        zip(ess_enabled, [supervisors[i % len(supervisors)] for i in range(len(ess_enabled))], full_cycle_dates),
        start=1,
    ):
        tasks.append(_task(
            f"LEAVE-FULL-CYCLE-APPROVE-{i:02d}", "L3", [ess["username"], sup["username"]],
            f'Log in as "{ess["username"]}" and submit a 2-day Annual leave request. Log out, '
            f'log in as supervisor "{sup["username"]}", approve it. Log back in as '
            f'"{ess["username"]}" and verify the request shows as approved.',
            f"{TARGET_BASE}/login",
            _custom("leave.request.status", employee_code=ess["employee_code"], leave_type="Annual",
                    start_date=start_date, expected_status="approved"),
            preconditions=[{"type": "leave_balance.available", "employee_code": ess["employee_code"],
                             "leave_type": "Annual", "min_days": 2}],
            allowed_changes=["leave_request.create", "leave_request.status"],
        ))

    tasks.append(_task(
        "LEAVE-FULL-CYCLE-REJECT", "L3", [ess_enabled[0]["username"], supervisors[0]["username"]],
        f'Log in as "{ess_enabled[0]["username"]}", submit a leave request, then as supervisor '
        f'"{supervisors[0]["username"]}" reject it with a comment. Log back in as the employee '
        "and verify the rejected status and comment are visible.",
        f"{TARGET_BASE}/login",
        _custom("leave.request.status", employee_code=ess_enabled[0]["employee_code"], leave_type="Sick",
                start_date="2026-04-13", expected_status="rejected"),
        preconditions=[{"type": "leave_balance.available", "employee_code": ess_enabled[0]["employee_code"],
                         "leave_type": "Sick", "min_days": 1}],
    ))

    tasks.append(_task(
        "TIME-FULL-CYCLE-REJECT-RESUBMIT-APPROVE", "L3", [ess_enabled[1]["username"], supervisors[1]["username"]],
        f'Log in as "{ess_enabled[1]["username"]}", create and submit a timesheet. As supervisor '
        f'"{supervisors[1]["username"]}", reject it. Log back in as the employee, edit and '
        "resubmit it, then approve it as the supervisor. Verify the final status is approved.",
        f"{TARGET_BASE}/login",
        _custom("timesheet.status", employee_code=ess_enabled[1]["employee_code"],
                week_start_date="2026-03-09", expected_status="approved"),
        allowed_changes=["timesheet.create", "timesheet.status"],
    ))

    applied_candidate = next(c for c in candidates if c["status"] == "applied")
    tasks.append(_task(
        "RECRUITMENT-FULL-HIRE-WORKFLOW", "L3", ["admin"],
        f'Shortlist candidate {applied_candidate["first_name"]} {applied_candidate["last_name"]}, '
        "schedule an interview, then hire them. Verify a new employee record is created in the "
        f'{applied_candidate["vacancy_department"]} department.',
        f"{TARGET_BASE}/recruitment/candidates/{applied_candidate['id']}",
        _custom("candidate.hired", candidate_email=applied_candidate["email"]),
        allowed_changes=["candidate.status", "employee.create"],
        forbidden_changes=["employee.profile"],
    ))

    perf_ess = ess_enabled[2]
    perf_sup = supervisors[2]
    tasks.append(_task(
        "PERF-FULL-CYCLE-SELF-MANAGER-FINALIZE", "L3", [perf_ess["username"], perf_sup["username"], "admin"],
        f'Log in as "{perf_ess["username"]}" and submit a self-review. Log in as supervisor '
        f'"{perf_sup["username"]}" and submit the manager review (this finalizes it). Log in '
        "as admin and close the review cycle. Verify the review is finalized.",
        f"{TARGET_BASE}/login",
        _custom("performance.review.finalized", employee_code=perf_ess["employee_code"], cycle_name="2026 H1 Review"),
    ))

    new_hire_dept_employee = employees[35]
    tasks.append(_task(
        "CROSS-MODULE-CREATE-EMPLOYEE-AND-LOGIN", "L3", ["admin"],
        f"Create a new employee, then create a linked ESS user account for them, then verify "
        "the new user can log in and see their own profile.",
        f"{TARGET_BASE}/pim/employees/new",
        _state_diff({"employees": {"created": [{}]}, "users": {"created": [{"role": "ess"}]}}),
        allowed_changes=["employee.create", "user.create"],
    ))

    tasks.append(_task(
        "PIM-DEACTIVATE-SUPERVISOR-INTEGRITY-CHECK", "L3", ["admin"],
        f'Deactivate supervisor "{supervisors[3]["username"]}"\'s employee record '
        f'({supervisors[3]["employee_code"]}) and confirm their direct reports\' records are '
        "untouched (their supervisor link may now point at a terminated employee, which is "
        "expected — no orphaned or deleted rows should result).",
        f"{TARGET_BASE}/pim/employees/{supervisors[3]['employee_id']}",
        _custom("employee.field_equals", employee_code=supervisors[3]["employee_code"], field="status", value="terminated"),
        preconditions=[{"type": "employee.status", "employee_code": supervisors[3]["employee_code"], "status": "active"}],
    ))

    tasks.append(_task(
        "REPORTS-FILTERED-EXPORT-VERIFY-CONTENT", "L3", ["admin"],
        'Filter the Leave Report to "approved" status only and export it as CSV; confirm the '
        "exported file contains only approved leave requests.",
        f"{TARGET_BASE}/reports/leave",
        _download("leave_report", "text/csv"),
    ))

    for i, doc_employee in enumerate([ess_enabled[3], ess_enabled[7] if len(ess_enabled) > 7 else ess_enabled[3]], start=1):
        tasks.append(_task(
            f"DOCUMENTS-UPLOAD-REPLACE-DOWNLOAD-{i:02d}", "L3", ["admin"],
            f"Upload a document to employee {doc_employee['employee_code']}, delete it, "
            "re-upload a document with the same filename, then download it and confirm the "
            "content matches the second upload.",
            f"{TARGET_BASE}/documents/employees/{doc_employee['employee_id']}",
            _custom("document.exists", employee_code=doc_employee["employee_code"], filename="contract.txt"),
        ))

    for i, (ess, sup, week) in enumerate(
        [(ess_enabled[6], supervisors[6 % len(supervisors)], "2026-03-23"),
         (ess_enabled[8] if len(ess_enabled) > 8 else ess_enabled[6], supervisors[0], "2026-03-30")],
        start=2,
    ):
        tasks.append(_task(
            f"TIME-FULL-CYCLE-REJECT-RESUBMIT-APPROVE-{i:02d}", "L3", [ess["username"], sup["username"]],
            f'Log in as "{ess["username"]}", create and submit a timesheet. As supervisor '
            f'"{sup["username"]}", reject it. Log back in as the employee, edit and resubmit '
            "it, then approve it as the supervisor. Verify the final status is approved.",
            f"{TARGET_BASE}/login",
            _custom("timesheet.status", employee_code=ess["employee_code"], week_start_date=week, expected_status="approved"),
            allowed_changes=["timesheet.create", "timesheet.status"],
        ))

    interview_stage_candidate = next(c for c in candidates if c["status"] == "interview")
    tasks.append(_task(
        "RECRUITMENT-FULL-HIRE-WORKFLOW-02", "L3", ["admin"],
        f'Hire candidate {interview_stage_candidate["first_name"]} {interview_stage_candidate["last_name"]}, '
        "who is already at the interview stage. Verify a new employee record is created in "
        f'the {interview_stage_candidate["vacancy_department"]} department.',
        f"{TARGET_BASE}/recruitment/candidates/{interview_stage_candidate['id']}",
        _custom("candidate.hired", candidate_email=interview_stage_candidate["email"]),
        allowed_changes=["candidate.status", "employee.create"],
        forbidden_changes=["employee.profile"],
    ))

    for i, sup_idx in enumerate([4, 5], start=2):
        sup = supervisors[sup_idx % len(supervisors)]
        tasks.append(_task(
            f"PIM-DEACTIVATE-SUPERVISOR-INTEGRITY-CHECK-{i:02d}", "L3", ["admin"],
            f'Deactivate supervisor "{sup["username"]}"\'s employee record '
            f'({sup["employee_code"]}) and confirm their direct reports\' records are '
            "untouched (their supervisor link may now point at a terminated employee, which "
            "is expected — no orphaned or deleted rows should result).",
            f"{TARGET_BASE}/pim/employees/{sup['employee_id']}",
            _custom("employee.field_equals", employee_code=sup["employee_code"], field="status", value="terminated"),
            preconditions=[{"type": "employee.status", "employee_code": sup["employee_code"], "status": "active"}],
        ))

    tasks.append(_task(
        "REPORTS-FILTERED-EXPORT-VERIFY-CONTENT-02", "L3", ["admin"],
        'Filter the Employees Report to "Sales" department and "active" status, export it '
        "as CSV, and confirm every row matches both filters.",
        f"{TARGET_BASE}/reports/employees",
        _download("employees_report", "text/csv"),
    ))

    tasks.append(_task(
        "LEAVE-SUBMIT-THEN-CANCEL-BEFORE-APPROVAL", "L3", [ess_enabled[9]["username"] if len(ess_enabled) > 9 else ess_enabled[0]["username"]],
        f'Log in as "{ess_enabled[9]["username"] if len(ess_enabled) > 9 else ess_enabled[0]["username"]}", '
        "submit a leave request, then cancel it yourself before any supervisor acts on it. "
        "Confirm the final status is cancelled and the leave balance is fully restored.",
        f"{TARGET_BASE}/leave/requests/new",
        _custom("leave.request.status",
                employee_code=ess_enabled[9]["employee_code"] if len(ess_enabled) > 9 else ess_enabled[0]["employee_code"],
                leave_type="Annual", start_date="2026-05-25", expected_status="cancelled"),
        preconditions=[{"type": "leave_balance.available",
                         "employee_code": ess_enabled[9]["employee_code"] if len(ess_enabled) > 9 else ess_enabled[0]["employee_code"],
                         "leave_type": "Annual", "min_days": 2}],
    ))

    tasks.append(_task(
        "TIME-SUPERVISOR-APPROVES-MULTIPLE-IN-ONE-SESSION", "L3", [supervisors[0]["username"]],
        "James Wilson has a submitted timesheet. Log in as his supervisor, approve it, then "
        "confirm the Team view no longer lists it as pending approval.",
        f"{TARGET_BASE}/time/timesheets?view=team",
        _custom("timesheet.status", employee_code="EMP-0008", week_start_date="2026-01-05", expected_status="approved"),
    ))

    tasks.append(_task(
        "CROSS-MODULE-CREATE-SUPERVISOR-ACCOUNT-AND-VERIFY-SCOPE", "L3", ["admin"],
        "Create a new employee, create a linked user account with the supervisor role, then "
        "log in as that user and confirm they see a Team view (direct reports) but no other "
        "employee's data outside their scope.",
        f"{TARGET_BASE}/pim/employees/new",
        _state_diff({"employees": {"created": [{}]}, "users": {"created": [{"role": "supervisor"}]}}),
        allowed_changes=["employee.create", "user.create"],
    ))

    return tasks


def build_l4(employees, users):
    tasks = []
    ess_enabled = [u for u in users if u["role"] == "ess" and u["enabled"]]
    supervisors = [u for u in users if u["role"] == "supervisor"]

    tasks.append(_task(
        "RECOVERY-REFRESH-MID-EDIT-DISCARDS-UNSAVED", "L4", ["admin"],
        f"Open the edit form for {employees[5]['employee_code']}, change a field but do not "
        "save, then refresh the page. Confirm the change was NOT persisted.",
        f"{TARGET_BASE}/pim/employees/{employees[5]['id']}/edit",
        _custom("employee.field_equals", employee_code=employees[5]["employee_code"],
                field="department", value=employees[5]["department"]),
    ))
    tasks.append(_task(
        "RECOVERY-BACK-BUTTON-AFTER-SAVE-NO-DUPLICATE", "L4", ["admin"],
        f"Edit {employees[6]['employee_code']}, save successfully, then use the browser Back "
        "button and forward again. Confirm no duplicate record or duplicate save occurred.",
        f"{TARGET_BASE}/pim/employees/{employees[6]['id']}/edit",
        _redirect(f"/pim/employees/{employees[6]['id']}"),
    ))
    tasks.append(_task(
        "RECOVERY-DUPLICATE-LEAVE-SUBMISSION-BLOCKED", "L4", [ess_enabled[0]["username"]],
        f'Log in as "{ess_enabled[0]["username"]}", submit a leave request for '
        "2026-05-04 to 2026-05-05, then attempt to submit an overlapping request for the "
        "same dates. Confirm the second attempt is rejected with an overlap error and only "
        "one request exists.",
        f"{TARGET_BASE}/leave/requests/new",
        _state_diff({"leave_requests": {"created": [{}]}}),
        preconditions=[{"type": "leave_balance.available", "employee_code": ess_enabled[0]["employee_code"],
                         "leave_type": "Annual", "min_days": 4}],
    ))
    tasks.append(_task(
        "RECOVERY-INSUFFICIENT-BALANCE-BLOCKED", "L4", [ess_enabled[1]["username"]],
        f'Log in as "{ess_enabled[1]["username"]}" and attempt to submit a leave request '
        "far longer than your available balance. Confirm it is rejected and no leave "
        "request is created.",
        f"{TARGET_BASE}/leave/requests/new",
        _state_diff({"leave_requests": {"created": []}}),
    ))
    tasks.append(_task(
        "RECOVERY-CANCEL-PENDING-LEAVE", "L4", [ess_enabled[2]["username"]],
        f'Log in as "{ess_enabled[2]["username"]}", submit a leave request, then cancel it '
        "before it is approved. Confirm the status shows cancelled.",
        f"{TARGET_BASE}/leave/requests/new",
        _custom("leave.request.status", employee_code=ess_enabled[2]["employee_code"], leave_type="Annual",
                start_date="2026-05-11", expected_status="cancelled"),
        preconditions=[{"type": "leave_balance.available", "employee_code": ess_enabled[2]["employee_code"],
                         "leave_type": "Annual", "min_days": 2}],
    ))
    tasks.append(_task(
        "RECOVERY-DOUBLE-APPROVE-NOOP", "L4", [supervisors[0]["username"]],
        "James Wilson has a pending leave request — approve it as his supervisor, then "
        "attempt to approve the same request again. Confirm the second attempt is a no-op "
        "and does not error or change the decision.",
        f"{TARGET_BASE}/leave/requests",
        _custom("leave.request.status", employee_code="EMP-0008", leave_type="Annual",
                start_date="2026-02-02", expected_status="approved"),
    ))
    tasks.append(_task(
        "RECOVERY-FAULT-INJECTED-APPROVAL-CONFLICT-RETRY", "L4", [supervisors[0]["username"]],
        "James Wilson has a pending leave request. Approve it — the first attempt may fail "
        "with a conflict error; if so, reload and try again. Confirm it ends up approved.",
        f"{TARGET_BASE}/leave/requests",
        _custom("leave.request.status", employee_code="EMP-0008", leave_type="Annual",
                start_date="2026-02-02", expected_status="approved"),
    ))
    tasks.append(_task(
        "RECOVERY-UNAUTHORIZED-DIRECT-URL-BLOCKED", "L4", [ess_enabled[3]["username"]],
        f'Log in as "{ess_enabled[3]["username"]}" and attempt to navigate directly to the '
        "Admin Users page. Confirm access is denied and no data is exposed.",
        f"{TARGET_BASE}/admin/users",
        _redirect("/admin/users"),
    ))
    tasks.append(_task(
        "RECOVERY-UNAUTHORIZED-CROSS-EMPLOYEE-ACCESS-BLOCKED", "L4", [ess_enabled[4]["username"]],
        f'Log in as "{ess_enabled[4]["username"]}" and attempt to view another employee\'s '
        "profile by direct URL. Confirm access is denied.",
        f"{TARGET_BASE}/pim/employees/{employees[0]['id']}",
        _redirect(f"/pim/employees/{employees[0]['id']}"),
    ))
    tasks.append(_task(
        "RECOVERY-NO-RESULT-FILTER-THEN-RESET", "L4", ["admin"],
        'Filter the Employees list to a department/status combination with zero results, '
        "confirm the empty state, then reset the filters and find a specific employee "
        f"({employees[20]['employee_code']}).",
        f"{TARGET_BASE}/pim/employees",
        _custom("employee.field_equals", employee_code=employees[20]["employee_code"], field="status",
                value=employees[20]["status"]),
    ))
    tasks.append(_task(
        "RECOVERY-STALE-EDIT-FORM-CONFLICT", "L4", ["admin"],
        f"Open the edit form for {employees[8]['employee_code']}. While it's open, the "
        "record is changed elsewhere. Submit your (now stale) edit and confirm you get a "
        "conflict message rather than silently overwriting the other change.",
        f"{TARGET_BASE}/pim/employees/{employees[8]['id']}/edit",
        _redirect(f"/pim/employees/{employees[8]['id']}/edit"),
    ))
    tasks.append(_task(
        "RECOVERY-INVALID-DOCUMENT-TYPE-REJECTED", "L4", ["admin"],
        f"Attempt to upload a .exe file to employee {employees[9]['employee_code']}'s "
        "Documents page. Confirm it is rejected with a clear message and no document row "
        "is created.",
        f"{TARGET_BASE}/documents/employees/{employees[9]['id']}",
        _state_diff({"documents": {"created": []}}),
    ))
    tasks.append(_task(
        "RECOVERY-OVERSIZED-DOCUMENT-REJECTED", "L4", ["admin"],
        f"Attempt to upload a file larger than 2MB to employee {employees[10]['employee_code']}'s "
        "Documents page. Confirm it is rejected and no document row is created.",
        f"{TARGET_BASE}/documents/employees/{employees[10]['id']}",
        _state_diff({"documents": {"created": []}}),
    ))
    tasks.append(_task(
        "RECOVERY-DUPLICATE-FILENAME-REJECTED", "L4", ["admin"],
        f"Upload a document to employee {employees[11]['employee_code']}, then attempt to "
        "upload a different file with the same filename. Confirm the second upload is "
        "rejected as a duplicate.",
        f"{TARGET_BASE}/documents/employees/{employees[11]['id']}",
        _redirect(f"/documents/employees/{employees[11]['id']}"),
    ))
    tasks.append(_task(
        "RECOVERY-LOCKED-APPROVED-TIMESHEET-NO-EDIT", "L4", ["admin"],
        "John Park has an already-approved, locked timesheet. Confirm there is no way to "
        "edit its entries through the UI.",
        f"{TARGET_BASE}/time/timesheets?view=team",
        _custom("timesheet.status", employee_code="EMP-0021", week_start_date="2025-12-29", expected_status="approved"),
    ))
    tasks.append(_task(
        "RECOVERY-REJECTED-TIMESHEET-RESUBMIT", "L4", [ess_enabled[5]["username"]],
        f'Log in as "{ess_enabled[5]["username"]}", create and submit a timesheet, have it '
        "rejected, then edit and resubmit it successfully.",
        f"{TARGET_BASE}/time/timesheets/new",
        _custom("timesheet.status", employee_code=ess_enabled[5]["employee_code"],
                week_start_date="2026-03-16", expected_status="submitted"),
    ))
    tasks.append(_task(
        "RECOVERY-EMPTY-COMMENT-LEAVE-REQUEST-STILL-VALID", "L4", [ess_enabled[6]["username"]],
        f'Log in as "{ess_enabled[6]["username"]}" and submit a valid leave request with an '
        "empty comment field. Confirm it is still accepted.",
        f"{TARGET_BASE}/leave/requests/new",
        _custom("leave.request.status", employee_code=ess_enabled[6]["employee_code"], leave_type="Annual",
                start_date="2026-05-18", expected_status="pending"),
        preconditions=[{"type": "leave_balance.available", "employee_code": ess_enabled[6]["employee_code"],
                         "leave_type": "Annual", "min_days": 1}],
    ))
    tasks.append(_task(
        "RECOVERY-END-DATE-BEFORE-START-DATE-REJECTED", "L4", [ess_enabled[7]["username"] if len(ess_enabled) > 7 else ess_enabled[0]["username"]],
        "Attempt to submit a leave request whose end date is before its start date. "
        "Confirm it is rejected with a clear validation message.",
        f"{TARGET_BASE}/leave/requests/new",
        _state_diff({"leave_requests": {"created": []}}),
    ))
    tasks.append(_task(
        "RECOVERY-FAULT-INJECTED-TIMESHEET-APPROVAL-ERROR-RETRY", "L4", [supervisors[1]["username"]],
        "Approving a submitted timesheet may fail with a server error on the first attempt. "
        "If so, reload and try again. Confirm it ends up approved with no duplicate entries.",
        f"{TARGET_BASE}/time/timesheets?view=team",
        _custom("timesheet.status", employee_code=ess_enabled[1]["employee_code"],
                week_start_date="2026-03-09", expected_status="approved"),
    ))
    tasks.append(_task(
        "RECOVERY-WRONG-DEPARTMENT-FILTER-THEN-CORRECT", "L4", ["admin"],
        f"Filter the Employees list by the wrong department first, notice "
        f"{employees[22]['employee_code']} is not there, correct the filter to "
        f"{employees[22]['department']}, and confirm the employee is now found.",
        f"{TARGET_BASE}/pim/employees",
        _custom("employee.field_equals", employee_code=employees[22]["employee_code"], field="department",
                value=employees[22]["department"]),
    ))
    return tasks


def build_l5(employees, users, candidates):
    tasks = []
    ess_enabled = [u for u in users if u["role"] == "ess" and u["enabled"]]
    supervisors = [u for u in users if u["role"] == "supervisor"]

    tasks.append(_task(
        "EXPERT-MULTI-ACTOR-LEAVE-THEN-REPORT-VERIFY", "L5", [ess_enabled[8]["username"] if len(ess_enabled) > 8 else ess_enabled[0]["username"], supervisors[4]["username"], "admin"],
        f'Log in as "{ess_enabled[0]["username"]}" and submit a leave request. Log in as '
        f'supervisor "{supervisors[4]["username"]}" and approve it. Log in as admin, export '
        "the Leave Report as CSV, and confirm the approved request appears in it.",
        f"{TARGET_BASE}/login",
        _custom("leave.request.status", employee_code=ess_enabled[0]["employee_code"], leave_type="Annual",
                start_date="2026-06-01", expected_status="approved"),
        preconditions=[{"type": "leave_balance.available", "employee_code": ess_enabled[0]["employee_code"],
                         "leave_type": "Annual", "min_days": 2}],
    ))
    tasks.append(_task(
        "EXPERT-PRIVILEGE-ESCALATION-BLOCKED", "L5", [ess_enabled[1]["username"]],
        f'Log in as "{ess_enabled[1]["username"]}" and attempt to create a new admin user '
        "account by submitting a request directly to the Admin Users creation endpoint. "
        "Confirm it is denied and no user is created.",
        f"{TARGET_BASE}/admin/users/new",
        _state_diff({"users": {"created": []}}),
        forbidden_changes=["user.create"],
    ))
    tasks.append(_task(
        "EXPERT-CONCURRENT-TIMESHEET-APPROVAL", "L5", [supervisors[0]["username"], supervisors[1]["username"]],
        "James Wilson's submitted timesheet is approved by two different actors in rapid "
        "succession (simulating two supervisors racing to approve). Confirm the final "
        "state is a single consistent 'approved' status, not a corrupted or double-applied "
        "state.",
        f"{TARGET_BASE}/time/timesheets?view=team",
        _custom("timesheet.status", employee_code="EMP-0008", week_start_date="2026-01-05", expected_status="approved"),
    ))
    tasks.append(_task(
        "EXPERT-FULL-HIRE-TO-FIRST-LEAVE-LIFECYCLE", "L5", ["admin", ess_enabled[2]["username"], supervisors[2]["username"]],
        "Hire a shortlisted candidate into a new employee record, create a linked ESS user "
        "account for them, assign a supervisor, log in as the new employee and submit a "
        "leave request, then log in as their supervisor and approve it. Verify the leave "
        "request is approved.",
        f"{TARGET_BASE}/recruitment/vacancies",
        _state_diff({"employees": {"created": [{}]}, "users": {"created": [{}]}, "leave_requests": {"created": [{}]}}),
        allowed_changes=["candidate.status", "employee.create", "user.create", "leave_request.create", "leave_request.status"],
    ))
    tasks.append(_task(
        "EXPERT-COMPLEX-FILTER-TO-PAGE-3-MATCH", "L5", ["admin"],
        f"Using combined department and status filters, locate employee "
        f"{employees[41]['employee_code']} on the Employees list, which is not on page 1.",
        f"{TARGET_BASE}/pim/employees",
        _custom("employee.field_equals", employee_code=employees[41]["employee_code"], field="status",
                value=employees[41]["status"]),
    ))
    tasks.append(_task(
        "EXPERT-DOCUMENT-FULL-LIFECYCLE", "L5", ["admin"],
        f"Upload a document to employee {employees[13]['employee_code']}, attempt a "
        "duplicate-filename upload (rejected), delete the original, re-upload the same "
        "filename (now accepted), then download it and confirm the content is correct.",
        f"{TARGET_BASE}/documents/employees/{employees[13]['id']}",
        _custom("document.exists", employee_code=employees[13]["employee_code"], filename="offer_letter.txt"),
    ))
    tasks.append(_task(
        "EXPERT-MULTI-TAB-SESSION-ISOLATION", "L5", ["admin", ess_enabled[3]["username"]],
        f'With two browser tabs, log in as "admin" in one and "{ess_enabled[3]["username"]}" '
        "in the other. Confirm actions in one tab do not affect or leak into the other's "
        "session (each tab keeps its own logged-in identity and permitted view).",
        f"{TARGET_BASE}/login",
        _redirect("/dashboard"),
    ))
    tasks.append(_task(
        "EXPERT-CROSS-MODULE-RECRUITMENT-PERFORMANCE-REPORT", "L5", ["admin"],
        "Hire a shortlisted candidate, then confirm the new employee appears correctly in "
        "the Employees Report export with the right department and job title.",
        f"{TARGET_BASE}/recruitment/vacancies",
        _download("employees_report", "text/csv"),
        allowed_changes=["candidate.status", "employee.create"],
    ))
    tasks.append(_task(
        "EXPERT-FAULT-INJECTED-LEAVE-APPROVAL-RECOVERY", "L5", [supervisors[3]["username"]],
        "A leave approval you attempt may be interrupted by a simulated conflict error. "
        "Recover from it (reload, retry) and confirm the leave request ends up approved "
        "with no duplicate or corrupted state.",
        f"{TARGET_BASE}/leave/requests",
        _custom("leave.request.status", employee_code="EMP-0025", leave_type="Sick",
                start_date="2026-01-12", expected_status="approved"),
    ))
    tasks.append(_task(
        "EXPERT-LONG-WORKFLOW-REVIEW-CYCLE-CLOSE", "L5", [ess_enabled[4]["username"], supervisors[4]["username"], "admin"],
        f'Log in as "{ess_enabled[4]["username"]}" and submit a self-review. Log in as '
        f'supervisor "{supervisors[4]["username"]}" and submit the manager review. Log in '
        "as admin and close the entire review cycle. Confirm the cycle status is closed "
        "and the review is finalized.",
        f"{TARGET_BASE}/login",
        _custom("performance.review.finalized", employee_code=ess_enabled[4]["employee_code"], cycle_name="2026 H1 Review"),
    ))
    return tasks


def main():
    employees, users, candidates, vacancies = load_fixture_data()

    by_level = {
        "L1": build_l1(employees, users),
        "L2": build_l2(employees, users, candidates),
        "L3": build_l3(employees, users, candidates),
        "L4": build_l4(employees, users),
        "L5": build_l5(employees, users, candidates),
    }

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    counts = {}
    for level, tasks in by_level.items():
        path = TASKS_DIR / f"{level}.yaml"
        path.write_text(yaml.dump(tasks, sort_keys=False, allow_unicode=True, width=100))
        counts[level] = len(tasks)
        total += len(tasks)

    manifest = {
        "catalog_version": "v1",
        "status": "draft",
        "task_count": total,
        "level_distribution": counts,
        "fixture": "base",
    }
    (CATALOG_DIR / "catalog.yaml").write_text(yaml.dump(manifest, sort_keys=False))

    print(f"Generated {total} tasks: {counts}")


if __name__ == "__main__":
    main()
