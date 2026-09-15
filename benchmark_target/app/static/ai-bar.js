/**
 * AI bar — a floating, top-center prompt box wired straight into this project's own
 * Hermes agent backend (backend/app/api/routes.py, port 8000). Mounted only on
 * authenticated pages (base.html includes this script only inside the `{% if user %}`
 * branch, i.e. never on /login).
 *
 * Deliberately built in a Shadow DOM, not injected as plain elements into the page's own
 * DOM: the host page's CSS can't leak in and clobber the bar's layout, and the bar's own
 * styles can't leak out and affect the host page — the two are visually/style-isolated
 * even though they share one document (a true cross-origin iframe would isolate the JS
 * globals too, but would also need postMessage plumbing for zero benefit here, since the
 * bar talks to the backend over plain fetch/EventSource either way).
 *
 * The page-bridge executes whitelisted DOM actions in this very document. The backend
 * receives snapshots and chooses the next action. No CDP or separate browser is used.
 *
 * No plan-confirmation step (confirm_plan=false, per the request) — the agent starts
 * acting immediately. auto_approve stays false: a real permission prompt (SAFE /
 * NEEDS_CONFIRMATION / BLOCKED, see permission/rules.py) pops up centered on screen and
 * genuinely waits for a real click, via the existing approval_request/respond mechanism
 * this backend already has (TaskManager.request_approval/resolve_approval) — nothing new
 * needed there, this widget is just the first browser-side client for it outside the
 * original Test Console.
 */
(function () {
  const BACKEND_URL = window.location.protocol + "//" + window.location.hostname + ":8000";
  const STORAGE_SESSION_ID = "hermesAiBarSessionId";
  const STORAGE_OWNER_TOKEN = "hermesAiBarOwnerToken";
  const STORAGE_TASK_ID = "hermesAiBarTaskId";
  const STORAGE_BRIDGE_TOKEN = "hermesAiBarBridgeToken";
  const STORAGE_COMMAND = "hermesAiBarCommand";
  const bridge = window.HermesPageBridge;
  const tabId = crypto.randomUUID();

  function getOrCreateSession() {
    let sessionId = sessionStorage.getItem(STORAGE_SESSION_ID);
    let ownerToken = sessionStorage.getItem(STORAGE_OWNER_TOKEN);
    if (!sessionId || !ownerToken) {
      sessionId = crypto.randomUUID();
      ownerToken = crypto.randomUUID();
      sessionStorage.setItem(STORAGE_SESSION_ID, sessionId);
      sessionStorage.setItem(STORAGE_OWNER_TOKEN, ownerToken);
    }
    return { sessionId, ownerToken };
  }

  const host = document.createElement("div");
  host.id = "hermes-ai-bar-host";
  host.dataset.tabId = tabId;
  document.body.appendChild(host);
  const root = host.attachShadow({ mode: "open" });

  root.innerHTML = `
    <style>
      :host { all: initial; }
      * { box-sizing: border-box; font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
      .bar-wrap {
        position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
        z-index: 2147483000; width: min(560px, 90vw);
      }
      .bar {
        display: flex; gap: 0.5rem; align-items: center;
        background: #1e2530; border-radius: 999px; padding: 0.5rem 0.6rem 0.5rem 1rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
      }
      .bar input {
        flex: 1; border: none; outline: none; background: transparent; color: #fff;
        font-size: 0.9rem; min-width: 0;
      }
      .bar input::placeholder { color: #9aa4b2; }
      .bar button {
        border: none; border-radius: 999px; padding: 0.45rem 0.9rem; font-size: 0.85rem;
        cursor: pointer; font-weight: 600;
      }
      .send-btn { background: #4c7bf3; color: #fff; }
      .send-btn:disabled { background: #3a4150; color: #7c8494; cursor: not-allowed; }
      .stop-btn { background: #c0392b; color: #fff; }
      .status-line {
        margin-top: 0.4rem; background: #1e2530; color: #cdd4e0; border-radius: 10px;
        padding: 0.5rem 0.8rem; font-size: 0.78rem; max-height: 140px; overflow-y: auto;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25); display: none;
      }
      .status-line.visible { display: block; }
      .status-line .line { padding: 0.1rem 0; }
      .status-line .line.error { color: #ff8a80; }

      .overlay {
        position: fixed; inset: 0; z-index: 2147483001; background: rgba(0,0,0,0.55);
        display: flex; align-items: center; justify-content: center;
      }
      .overlay[hidden] { display: none; }
      .modal {
        background: #fff; border-radius: 10px; padding: 1.25rem; width: min(480px, 90vw);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
      }
      .modal h2 { margin: 0 0 0.5rem; font-size: 1.05rem; color: #1f2933; }
      .modal p.subtitle { margin: 0 0 0.75rem; color: #667080; font-size: 0.85rem; }
      .modal pre {
        background: #f4f6f9; border-radius: 6px; padding: 0.6rem; font-size: 0.78rem;
        max-height: 220px; overflow: auto; white-space: pre-wrap; word-break: break-word;
        color: #1f2933; margin: 0 0 1rem;
      }
      .modal .answer-input {
        width: 100%; padding: 0.5rem 0.6rem; margin: 0 0 1rem; font-size: 0.85rem;
        border: 1px solid #dde3ea; border-radius: 6px; color: #1f2933;
      }
      .modal .actions { display: flex; gap: 0.5rem; justify-content: flex-end; }
      .modal button { border: none; border-radius: 6px; padding: 0.5rem 1rem; font-size: 0.85rem; cursor: pointer; }
      .approve-btn { background: #1e8e5a; color: #fff; }
      .deny-btn { background: #c0392b; color: #fff; }
    </style>

    <div class="bar-wrap">
      <div class="bar">
        <input type="text" placeholder="บอก agent ว่าต้องการให้ทำอะไร..." />
        <button class="send-btn">Send</button>
        <button class="stop-btn" hidden>Stop</button>
      </div>
      <div class="status-line"></div>
    </div>

    <div class="overlay" hidden>
      <div class="modal">
        <h2 class="modal-title">Agent ต้องขออนุมัติก่อนทำ action นี้</h2>
        <p class="subtitle modal-subtitle">ตรวจสอบรายละเอียดด้านล่างก่อนกด Approve</p>
        <pre></pre>
        <input type="text" class="answer-input" placeholder="พิมพ์คำตอบ..." hidden />
        <div class="actions">
          <button class="deny-btn">Deny</button>
          <button class="approve-btn">Approve</button>
        </div>
      </div>
    </div>
  `;

  const input = root.querySelector(".bar input");
  const sendBtn = root.querySelector(".send-btn");
  const stopBtn = root.querySelector(".stop-btn");
  const statusLine = root.querySelector(".status-line");
  const overlay = root.querySelector(".overlay");
  const modalTitle = root.querySelector(".modal-title");
  const modalSubtitle = root.querySelector(".modal-subtitle");
  const modalPre = root.querySelector(".modal pre");
  const answerInput = root.querySelector(".answer-input");
  const approveBtn = root.querySelector(".approve-btn");
  const denyBtn = root.querySelector(".deny-btn");

  let currentTaskId = null;
  let currentEventSource = null;
  let pendingRequestId = null;
  let pendingCmdType = null;
  let bridgeToken = null;
  let pageLeaving = false;
  let polling = false;
  let pendingResult = null;

  function finishTask(event) {
    if (!currentTaskId) return;
    const success = event.status === "done" && event.result?.success !== false;
    logLine((success ? "เสร็จแล้ว: " : "งานยังไม่สำเร็จ: ") +
      (event.error || event.result?.message || event.status), !success);
    currentTaskId = null;
    bridgeToken = null;
    pendingResult = null;
    for (const key of [STORAGE_TASK_ID, STORAGE_BRIDGE_TOKEN, STORAGE_COMMAND]) sessionStorage.removeItem(key);
    closeStream();
    hideApprovalModal();
    setRunning(false);
  }

  async function pollPage() {
    if (polling || !bridge || !bridgeToken || !currentTaskId || pageLeaving) return;
    polling = true;
    try {
      const taskId = currentTaskId;
      const response = await fetch(`${BACKEND_URL}/tasks/${taskId}/page`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: bridgeToken, snapshot: bridge.snapshot(), ...(pendingResult || {}) }),
      });
      if (response.status === 410) {
        const status = await fetch(`${BACKEND_URL}/tasks/${taskId}`);
        const result = status.ok ? await status.json() : { status: "error", error: "backend เริ่มใหม่แล้ว กรุณาสั่งงานอีกครั้ง" };
        finishTask(result);
        return;
      }
      if (!response.ok) throw new Error(`page bridge: HTTP ${response.status}`);
      const { command } = await response.json();
      if (pageLeaving || currentTaskId !== taskId) return;
      if (pendingResult) {
        pendingResult = null;
        sessionStorage.removeItem(STORAGE_COMMAND);
      }
      if (command) {
        // Persist before a click can unload this document. A new document ACKs it
        // with its fresh snapshot instead of executing the same click twice.
        sessionStorage.setItem(STORAGE_COMMAND, JSON.stringify({
          id: command.id, document_id: bridge.documentId, task_id: taskId,
        }));
        try {
          const message = await bridge.execute(command);
          pendingResult = { command_id: command.id, success: true, message };
        } catch (error) {
          pendingResult = { command_id: command.id, success: false, message: String(error) };
        }
      }
    } catch (error) {
      if (!pageLeaving) logLine("เชื่อมต่อหน้าเว็บขัดข้อง กำลังลองใหม่: " + error, true);
    } finally {
      polling = false;
      if (currentTaskId && bridgeToken && !pageLeaving) setTimeout(pollPage, 400);
    }
  }

  function logLine(text, isError) {
    statusLine.classList.add("visible");
    const div = document.createElement("div");
    div.className = "line" + (isError ? " error" : "");
    div.textContent = text;
    statusLine.appendChild(div);
    statusLine.scrollTop = statusLine.scrollHeight;
  }

  function setRunning(running) {
    input.disabled = running;
    sendBtn.disabled = running;
    stopBtn.hidden = !running;
  }

  // Different cmd.type values need different responses, not just Approve/Deny:
  //  - "request_user_input" (orchestrator.py::_request_user_input, e.g. mid-task CAPTCHA
  //    or a value the site rejected) needs a typed answer, sent back as `answer_text`.
  //  - "confirm_plan" and ordinary permission-confirmation cmds are plain yes/no.
  // Found live: the first real run of this widget hit exactly a request_user_input cmd
  // (the domain guard blocked a navigation and the agent asked for the current URL) —
  // this branch didn't exist yet at that point, which is how the gap got caught.
  function showApprovalModal(requestId, cmd) {
    pendingRequestId = requestId;
    pendingCmdType = cmd && cmd.type;
    modalPre.textContent = JSON.stringify(cmd, null, 2);
    if (pendingCmdType === "request_user_input") {
      modalTitle.textContent = "Agent ต้องการข้อมูลเพิ่มเติม";
      modalSubtitle.textContent = (cmd && cmd.prompt) || "พิมพ์คำตอบแล้วกด Approve เพื่อส่งกลับ";
      answerInput.hidden = false;
      answerInput.value = "";
      approveBtn.textContent = "Send";
      denyBtn.textContent = "Skip";
    } else {
      modalTitle.textContent = "Agent ต้องขออนุมัติก่อนทำ action นี้";
      modalSubtitle.textContent = "ตรวจสอบรายละเอียดด้านล่างก่อนกด Approve";
      answerInput.hidden = true;
      approveBtn.textContent = "Approve";
      denyBtn.textContent = "Deny";
    }
    overlay.hidden = false;
  }

  function hideApprovalModal() {
    pendingRequestId = null;
    pendingCmdType = null;
    overlay.hidden = true;
  }

  async function respondApproval(approved) {
    if (!pendingRequestId || !currentTaskId) return;
    const requestId = pendingRequestId;
    const cmdType = pendingCmdType;
    const answerText = cmdType === "request_user_input" && approved ? answerInput.value : null;
    hideApprovalModal();
    try {
      await fetch(`${BACKEND_URL}/tasks/${currentTaskId}/respond`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: requestId, approved, answer_text: answerText }),
      });
      logLine(approved ? "ส่งคำตอบแล้ว — agent ทำงานต่อ" : "ข้ามแล้ว — agent จะหาทางอื่น");
    } catch (e) {
      logLine("ตอบ approval ไม่สำเร็จ: " + e, true);
    }
  }

  approveBtn.addEventListener("click", () => respondApproval(true));
  denyBtn.addEventListener("click", () => respondApproval(false));

  function closeStream() {
    if (currentEventSource) {
      currentEventSource.close();
      currentEventSource = null;
    }
  }

  function startStream(taskId) {
    const es = new EventSource(`${BACKEND_URL}/tasks/${taskId}/stream`);
    currentEventSource = es;
    es.onmessage = (ev) => {
      let event;
      try {
        event = JSON.parse(ev.data);
      } catch (e) {
        return;
      }
      switch (event.kind) {
        case "approval_request":
          showApprovalModal(event.request_id, event.cmd);
          logLine("รอการอนุมัติ...");
          break;
        case "auto_approved":
          logLine("อนุมัติอัตโนมัติ: " + (event.cmd && event.cmd.type ? event.cmd.type : ""));
          break;
        case "approval_timeout":
          logLine("หมดเวลาขออนุมัติ", true);
          hideApprovalModal();
          break;
        case "chat_reply":
          logLine(event.message || "");
          break;
        case "task_done":
          finishTask(event);
          break;
        case "step_start":
          logLine(`ขั้นตอน ${event.step}: ${event.cmd.type} ${event.label || ""}`);
          break;
        case "step":
          logLine(`ขั้นตอน ${event.step}: ${event.success ? "ทำแล้ว" : "ยังทำไม่ได้"}`, !event.success);
          break;
        default:
          if (event.kind) logLine(event.kind);
      }
    };
    es.onerror = () => {
      logLine("การเชื่อมต่อ stream หลุด", true);
    };
  }

  async function sendPrompt() {
    if (sendBtn.disabled || currentTaskId) return;
    const goal = input.value.trim();
    if (!goal) return;
    if (!bridge) {
      logLine("ไม่พบ page bridge กรุณารีเฟรชหน้าเว็บ", true);
      return;
    }
    const { sessionId, ownerToken } = getOrCreateSession();

    setRunning(true);
    statusLine.innerHTML = "";
    logLine("ส่ง goal ไปที่ agent...");

    let resp;
    try {
      const capabilities = await fetch(`${BACKEND_URL}/page-bridge`);
      if (!capabilities.ok || (await capabilities.json()).execution !== "in_page") {
        throw new Error("กรุณารีสตาร์ท backend ให้รองรับการทำงานในหน้าเว็บก่อน");
      }
      resp = await fetch(`${BACKEND_URL}/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: window.location.href,
          goal: goal,
          confirm_plan: false,
          auto_approve: false,
          embedded_page: bridge.snapshot(),
          session_id: sessionId,
          session_owner_token: ownerToken,
          max_steps: 30,
        }),
      });
    } catch (e) {
      logLine("เรียก backend ไม่สำเร็จ (server รันอยู่ไหม? ที่ " + BACKEND_URL + "): " + e, true);
      setRunning(false);
      return;
    }

    if (!resp.ok) {
      const text = await resp.text();
      logLine(`backend ตอบ ${resp.status}: ${text}`, true);
      setRunning(false);
      return;
    }

    const data = await resp.json();
    if (!data.embedded_token) {
      logLine("backend ยังเป็นเวอร์ชันเดิม กรุณารีสตาร์ท backend ก่อนใช้ผู้ช่วยในหน้าเว็บ", true);
      setRunning(false);
      return;
    }
    currentTaskId = data.task_id;
    bridgeToken = data.embedded_token;
    sessionStorage.setItem(STORAGE_BRIDGE_TOKEN, bridgeToken);
    sessionStorage.setItem(STORAGE_TASK_ID, currentTaskId);
    input.value = "";
    logLine(`task ${currentTaskId} เริ่มทำงานแล้ว`);
    startStream(currentTaskId);
    pollPage();
  }

  sendBtn.addEventListener("click", sendPrompt);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendPrompt();
  });

  stopBtn.addEventListener("click", async () => {
    if (!currentTaskId) return;
    try {
      await fetch(`${BACKEND_URL}/tasks/${currentTaskId}/stop`, { method: "POST" });
      logLine("ส่งคำสั่งหยุดแล้ว...");
    } catch (e) {
      logLine("หยุด task ไม่สำเร็จ: " + e, true);
    }
  });

  // A menu click loads a new document. Resume the same task and pending approvals.
  const savedTaskId = sessionStorage.getItem(STORAGE_TASK_ID);
  bridgeToken = sessionStorage.getItem(STORAGE_BRIDGE_TOKEN);
  if (savedTaskId && bridgeToken && bridge) {
    currentTaskId = savedTaskId;
    try {
      const saved = JSON.parse(sessionStorage.getItem(STORAGE_COMMAND) || "null");
      if (saved && saved.task_id === savedTaskId) {
        pendingResult = { command_id: saved.id, success: saved.document_id !== bridge.documentId,
          message: "หน้าเว็บโหลดใหม่หลังคำสั่ง กรุณาตรวจผลจาก snapshot ล่าสุด" };
      }
    } catch (_) { sessionStorage.removeItem(STORAGE_COMMAND); }
    setRunning(true);
    logLine("เชื่อมต่อกับงานที่กำลังทำต่อ...");
    startStream(savedTaskId);
    pollPage();
  } else {
    sessionStorage.removeItem(STORAGE_TASK_ID);
    sessionStorage.removeItem(STORAGE_COMMAND);
  }
  window.addEventListener("pagehide", () => { pageLeaving = true; closeStream(); });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) {
      pageLeaving = false;
      if (currentTaskId && !currentEventSource) startStream(currentTaskId);
      pollPage();
    }
  });
})();
