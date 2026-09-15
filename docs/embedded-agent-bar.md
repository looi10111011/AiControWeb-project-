# Agent Bar inside the benchmark page

Run `.venv\Scripts\python.exe run.py web` from the project directory. This starts
the agent API on port 8000 and benchmark on port 8100, then opens only the benchmark
in the default browser. Sign in with a demo account and type into the floating bar.
Chrome debugging flags, port 9222, and a separate controlled browser are unnecessary.

Examples:

- `ไปหน้า PIM` — the agent clicks the PIM menu in the current tab.
- `ค้นหาพนักงานชื่อ James` — it fills the search field, clicks Search, and checks the result.

The model/provider configuration is the existing backend configuration. The backend
still runs as a service, but actions run inside the website's own document.

## Implementation

`page-bridge.js` collects visible text and indexed controls, excludes the assistant UI,
and hides password values. The widget sends this snapshot with `POST /tasks` as
`embedded_page`. That path bypasses browser acquisition and CDP entirely.

The backend uses the existing LLM provider/tool interface and permission classifier.
A random task token protects `POST /tasks/{id}/page`. The widget polls for one command,
executes a whitelisted DOM action, and acknowledges with the resulting snapshot.
The model receives the observed result before deciding the next step or finishing.

Task ID, token and pending command are stored in tab-scoped sessionStorage. When a menu
or form loads a new document, the new widget acknowledges the previous command and
continues the same task. Commands reference a document ID to reject stale targets.
Pending approvals and task completion continue to use the existing task SSE stream.

## Boundaries

- Supports DOM clicks, field entry, native select/checkbox controls, common keyboard
  events, scrolling, waiting, reading visible content and same-origin navigation.
- Does not execute arbitrary model-generated JavaScript or open another browser/tab.
- Browser-native dialogs, file pickers, screenshots and cross-origin frames are not
  controlled by this transport. Synthetic mouse/key events cannot emulate every
  browser-native interaction (for example CSS-only hover).
- The widget is currently mounted on signed-in benchmark pages. If the session expires
  and redirects to login, sign in again to resume, or restart the command if it timed out.
- After backend changes, restart the backend; refresh the benchmark to load updated JS.

## Verification (2026-09-15)

Live in an ordinary in-app browser with no CDP: `ไปหน้า PIM` moved Dashboard to
`/pim/employees`; `ค้นหาพนักงานชื่อ James` produced one result, James Wilson, with
`q=James` in the URL. The assistant resumed after both navigations and reported completion.
