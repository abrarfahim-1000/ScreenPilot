# Local to Cloud Project Plan (Gemini Live Agent Challenge 2026)

**Project codename:** Local-to-Cloud Terminal Navigator (UI Navigator category)  
**Goal:** Build a *screen-native* agent that can **see the UI**, **plan**, **take OS-level actions**, **verify outcomes**, and **recover** across **browser + terminal + cloud console**.  
**Core differentiator:** Not "macro clicking" — it's **visual precision + verification + safety guardrails**.

---

## 0) What you are building (one-paragraph pitch)

A cross-application UI Navigator that treats your desktop as a canvas. You give a high-level instruction (e.g., "Run my local model tests, upload the logs, and deploy the service to Cloud Run"), and the agent visually interprets screenshots/screen-stream frames, generates an action plan, executes OS-level macro actions (click/type/hotkeys), verifies each step via re-observation (visual precision checks), and recovers from UI changes or errors (context-aware visual recovery). All orchestration runs on Google Cloud and uses Gemini multimodal for perception and planning.

---

## 1) Category fit + compliance checklist (pass Stage One)

### Must-have requirements (non-negotiable)
- [ ] **Category:** UI Navigator (agent interprets screenshots/screen recordings and outputs executable actions)
- [ ] **Multimodal:** Uses Gemini multimodal (vision + text; optionally audio for "live feel")
- [ ] **Beyond text:** Demonstrates real UI understanding + actions in demo video (no mockups)
- [ ] **Developer tools:** Uses **Google GenAI SDK** in code
- [ ] **Google Cloud service:** Uses at least one GCP service (recommend **Cloud Run + Firestore + GCS**)
- [ ] **Hosted on GCP:** Backend/orchestrator is deployed to GCP (Cloud Run ideal)
- [ ] **Public code repo:** GitHub/GitLab public repository
- [ ] **Reproducibility:** README with step-by-step spin-up instructions
- [ ] **Proof of deployment:** Short recording showing GCP deployment/logs OR code link proving GCP usage
- [ ] **Architecture diagram:** Clear system diagram in submission
- [ ] **Demo video:** ≤ 4 minutes, English (or subtitles), shows real working system

### Disqualification risks to avoid
- Secret leakage (tokens/passwords shown or copied)
- "Agent" that is basically a scripted macro with no perception/verification
- Unsafe actions without confirmation (delete, stop services, deploy)
- Project not newly created within contest period

> **Note on ADK:** Plain GenAI SDK is the right choice for this project. ADK is listed as an alternative to the SDK, not an upgrade — and it adds framework overhead (~2 days ramp-up) without solving any of the actual hard problems here (coordinate resolution, frame compression, terminal output parsing, verification loops). The judging rubric rewards cloud-native architecture quality and visual precision, not framework choice. Use plain GenAI SDK and invest those 2 days in verification and recovery logic instead.

---

## 2) MVP scope (what to build in 15 days)

### The "killer workflow" (design your demo around this)
**Workflow A: Local test → extract metrics → upload logs → Cloud Run deploy → verify service**
1. Focus Terminal window
2. Run `make test 2>&1 | tee /tmp/test_out.log` (pipe stdout to file for reliable parsing)
3. Parse log file for summary (pass/fail, timing, key metrics) — do NOT rely on screenshot-based terminal reading
4. Upload log file to **Google Cloud Storage**
5. Deploy to **Cloud Run** via `gcloud run deploy` CLI command (stable, auditable, allowlisted) — NOT browser navigation of GCP Console
6. Open GCP Console in browser to visually verify "Deployed" status + service URL (demo-only visual moment)
7. Generate a short "deployment report" (links + metrics) and copy to clipboard

> This demonstrates: terminal vision + cloud deployment + verification. CLI deploy is intentionally chosen over Console navigation for stability — GCP Console UI can change between Day 9 and demo day.

### What "success" looks like (judge-visible)
- The agent **locates UI elements** correctly using **visual precision** (tabs/buttons/terminal prompt)
- The agent **executes actions** (click/type/hotkeys)
- The agent **verifies each step** by re-checking the UI state (**visual precision checks**)
- The agent **recovers** from a mismatch via **context-aware visual recovery** (scroll, search, switch tabs, ask user)
- The agent shows at least **one deliberate recovery moment in the demo** — don't engineer a flawless run; a recovery looks like a real agent, not a scripted macro
- The agent supports **interruptions**: "Stop—don't deploy, just upload logs."

---

## 3) Recommended tech stack (CV-grade + fastest)

### Client (Desktop "hands + eyes")
**Python (fastest path)**
- Screen capture: `mss`
- Frame change detection: compare frame hashes locally — only send frame to cloud when delta exceeds threshold (reduces API cost and latency significantly; implement on Day 1, not as polish)
- Input control: `pyautogui` (or `pynput`)
- Window focus helpers: `pygetwindow` (Windows), OS-specific fallbacks
- Terminal output: pipe stdout to file (`make test 2>&1 | tee /tmp/test_out.log`) and read directly — do NOT rely on screenshot-based OCR of terminal text
- Optional local OCR: `rapidfuzz` for fuzzy matching only if needed (keep vision primary)

### Backend (Google Cloud)
- **Cloud Run**: orchestrator API (session manager, policy engine, planner)
- **Firestore**: session state + audit logs (actions, decisions, verification results)
- **Cloud Storage**: store screenshots, session traces, logs, demo artifacts
- **Gemini via Google GenAI SDK**: perception + planning + action selection

> **Pub/Sub is deprioritized.** Cloud Run + Firestore + GCS fully satisfies the "Google Cloud Native" judging criterion. Add Pub/Sub only if time remains after Day 11 — it adds complexity without improving what judges score.

---

## 4) Architecture (reference design)

### High-level components
1. **Desktop Client**
   - Captures frames
   - **Runs local frame-diff check — skips transmission if screen unchanged**
   - Sends frames + metadata to backend
   - Receives structured actions
   - Executes actions
   - Sends action results + new frame back

2. **Cloud Orchestrator (Cloud Run)**
   - Maintains session state
   - Applies safety policies
   - Calls Gemini for perception/planning
   - Chooses actions
   - Writes audit logs (Firestore)
   - Stores artifacts (GCS)

3. **Gemini Multimodal**
   - UI element detection (semantic labels + optional boxes) — returns priority-ranked element list (primary target + 2 fallbacks) to avoid extra roundtrips on miss
   - Plan generation
   - Action proposal + verification questions
   - **Pre-step modal check**: every perception prompt includes "Is there an unexpected modal or dialog visible?" — zero extra code, catches OS interruptions

### Data flow (loop)
**Frame → [local diff check] → Perception → Plan → Action → Execute → VERIFY → Recover → (repeat)**

---

## 5) Action interface (executable actions output)

Implement a strict JSON action schema. `VERIFY` is a **first-class action type** — not a post-step behavior — so verification is visible in logs and auditable. Example:

```json
{
  "session_id": "abc",
  "step_id": 12,
  "actions": [
    {"type": "FOCUS_WINDOW", "title_contains": "Terminal", "reason": "Need to run tests"},
    {"type": "CLICK", "x": 312, "y": 988, "reason": "Place cursor in terminal prompt"},
    {"type": "TYPE", "text": "make test 2>&1 | tee /tmp/test_out.log\n", "reason": "Run test suite, pipe output to file"},
    {"type": "WAIT", "ms": 1800, "reason": "Wait for command output"},
    {"type": "VERIFY", "method": "read_file", "path": "/tmp/test_out.log", "reason": "Visual precision check: confirm test summary in output file"}
  ],
  "expected": {
    "must_see": ["PASS", "Summary", "exit code 0"],
    "timeout_ms": 15000,
    "max_retries": 3,
    "on_failure": "HAND_OFF_TO_USER"
  }
}
```

### Supported action types
- `FOCUS_WINDOW {title_contains}`
- `CLICK {x,y}` — always followed by a VERIFY step after major actions
- `DOUBLE_CLICK {x,y}`
- `RIGHT_CLICK {x,y}`
- `TYPE {text}`
- `HOTKEY {keys:[...]}`
- `SCROLL {dx,dy}`
- `WAIT {ms}`
- `DRAG {from:{x,y}, to:{x,y}}`
- `COPY`, `PASTE` (use carefully)
- `VERIFY {method, ...}` ← **first-class action, not a side effect**
- `ABORT` (for interruptions)
- `HAND_OFF_TO_USER` (triggered after max_retries exceeded)

### Execution constraints (safety)
- Max actions per step: **≤ 6**
- Confidence gating: only execute if model confidence ≥ threshold
- Always include a VERIFY action after every "major step" (navigation, deploy, stop service)
- **Click with verification pattern:** click predicted coords → re-capture → confirm correct element activated → if miss, retry with small search spiral using fallback elements from the ranked perception output

---

## 6) Perception output format (what Gemini returns)

Ask Gemini to return:
- A concise description of the screen
- **Whether an unexpected modal or dialog is present** (pre-step check, included in every prompt)
- A **priority-ranked** list of actionable UI elements (primary target + up to 2 fallbacks) with approximate location hints
- Current task-relevant state (e.g., "Cloud Run deploy button visible")

Example:

```json
{
  "screen_summary": "Brave browser open to Google Cloud Run service page. 'Deploy' button visible top-right.",
  "unexpected_modal": null,
  "elements": [
    {"label": "Deploy button", "hint": "top-right", "bbox": [1510, 110, 1650, 165], "confidence": 0.84, "priority": 1},
    {"label": "Deploy button (alternate)", "hint": "top-right expanded menu", "bbox": [1490, 170, 1660, 210], "confidence": 0.61, "priority": 2}
  ],
  "risks": ["Potential destructive action: deployment"],
  "next_best_action": "Ask for confirmation before deploying"
}
```

> Returning a priority-ranked fallback list means the action executor has alternatives without an additional Gemini roundtrip if the primary click misses.

---

## 7) Verification and recovery (the "agentic" part)

**This is the primary differentiator for the "visual precision" judging criterion. Every verification step should be described as a "visual precision check" in logs and in the submission text.**

### Verification strategy (required)
After every major action:
1. Re-capture frame
2. Execute a `VERIFY` action (first-class — logged and auditable)
3. Ask Gemini: "Did the expected state occur?"
4. If yes → proceed
5. If no → recovery plan (bounded — see below)

### Recovery playbook (deterministic first, model second)
The recovery loop is **bounded**: max 3 retries per step, then `HAND_OFF_TO_USER`. Without this, the agent can loop indefinitely.

- If target element not found:
  - try fallback elements from ranked perception output first (no extra Gemini call)
  - scroll down/up
  - zoom out (CTRL-)
  - switch tab (CTRL+TAB)
  - use in-page search (CTRL+F)
  - try `CTRL+L` (address bar), search, open correct URL
- If wrong window focused:
  - ALT+TAB / focus window by title
- If terminal output indicates error:
  - read error from log file, summarize, propose fix, ask user to approve re-run
- If max_retries exceeded:
  - execute `HAND_OFF_TO_USER` with summary of what was attempted

**Deliberately show one recovery moment in the demo.** A recovery demonstrates visual precision better than a flawless run.

---

## 8) Safety guardrails (do not skip — secret redaction moves to Day 1)

### Policy engine rules (enforced in Cloud Run)
**Allowlist**
- Allowed apps/windows: Terminal, Brave/Chrome, GCP Console, VS Code (optional)
- Allowed domains: `console.cloud.google.com`, `github.com`, your docs sites
- **Disallowed window titles list**: add patterns for password managers, banking tabs, etc. — prevents accidental action on sensitive windows

**Command restrictions**
- Allowlist terminal commands OR "review before execute"
- Block patterns: `rm -rf`, `del /s`, `shutdown`, resource deletion commands, unknown scripts
- `gcloud run deploy` is explicitly allowlisted

**Secret redaction — implement on Day 1**
- Regex scan on any TYPE action text before execution (patterns: `[A-Z0-9]{20,}`, `Bearer `, `-----BEGIN`, JWT patterns)
- Blur/mask screen regions matching secret patterns before frames are sent to cloud
- Never copy/paste secrets
- Mask in logs and artifacts

**Destructive actions confirmation**
- Stop services, deploy, delete: require explicit user confirmation step

### Audit trail (for judging + robustness)
Log (stored in Firestore; large artifacts in GCS):
- Prompts sent to Gemini (sanitized)
- Model outputs
- Actions executed (including all VERIFY actions)
- Before/after screenshots
- Verification results (visual precision check outcomes)
- Errors + recovery attempts (context-aware visual recovery log)

---

## 9) Implementation plan (15-day schedule)

> **Key changes from original:** Secret redaction moved to Day 1. Terminal workflow moved to Days 3–5. Verification/recovery loop moved to Days 6–8. This ensures verification is built on a solid terminal foundation, not the other way around.

### Days 1–2: Skeleton + local loop + secret redaction
- [x] Desktop client captures screenshot every 2–3 seconds
- [x] **Local frame-diff check: skip transmission if pixel hash delta below threshold**
- [x] **Secret redaction: regex scan on TYPE actions + frame masking**
- [x] Cloud Run endpoint receives frame + metadata
- [x] Gemini call returns "next action" JSON (with modal pre-check in prompt)
- [x] Client executes actions (click/type/wait)
- [x] Basic UI to start/stop session
- [x] Frame compression: JPEG 70%, max 1280×720

Deliverable: **Agent can open a webpage and click a visible button. Secrets never leave the client unredacted.**

### Days 3–5: Terminal workflow
- [x] Add focus window by title
- [x] Implement safe command execution flow (allowlist enforced)
- [x] **Pipe terminal output to file (`make test 2>&1 | tee /tmp/test_out.log`) — read file directly, not screenshot OCR**
- [x] Parse test summary from log file + write deployment report

Deliverable: **Agent runs tests and extracts a reliable summary from file output.**

### Days 6–8: Planning + verification loop
- [x] Add "plan first, then act" mode
- [x] **Add `VERIFY` as first-class action type in schema and executor**
- [x] **Add `max_retries` and `on_failure: HAND_OFF_TO_USER` to all `expected` blocks**
- [x] Add bounded recovery playbook (deterministic fallbacks → model → hand off)
- [x] Add session state in Firestore
- [x] **Add priority-ranked element list to perception output (primary + 2 fallbacks)**
- [x] **Add click-with-verification pattern in executor**

Deliverable: **Agent navigates GCP Console reliably with visual precision checks. Recovery loop has a defined exit condition.**

### Days 9–11: Cloud integration
- [ ] Cloud Storage uploads for logs/screenshots
- [ ] **Deploy flow uses `gcloud run deploy` CLI (allowlisted) — not Console browser navigation**
- [ ] Console navigation used only for post-deploy visual verification of service URL

Deliverable: **End-to-end workflow A completes (with confirmation). CLI deploy is stable and demo-safe.**

### Days 12–13: Guardrails + polish
- [ ] Destructive action confirmation UI
- [ ] Recovery playbook improvements
- [ ] Further latency optimization if needed

Deliverable: **Feels safe, robust, and demo-ready.**

### Days 14–15: Submission assets
- [ ] Architecture diagram
- [ ] README spin-up instructions
- [ ] Proof of GCP deployment recording
- [ ] 4-minute demo video script + recording
- [ ] Public repo cleanup

Deliverable: **Final Devpost submission package.**

---

## 10) Demo video script (≤ 4 minutes)

**0:00–0:20** Problem statement: "Cross-app workflows are slow; automation breaks across apps."  
**0:20–0:40** Show architecture diagram quickly (Cloud Run + Gemini + Firestore + GCS + Desktop client).

**0:40–1:45** Live demo Workflow A (first half):
- Give instruction: "Run tests, upload logs, deploy to Cloud Run."
- Agent narrates plan briefly.
- Agent acts in terminal — show it reading from the log file, not the terminal screen.
- Show visual precision check: "Test PASS confirmed (visual precision check ✓)"

**1:45–2:15** Interruption — **move this earlier; it's the strongest differentiator**:
- User: "Stop—skip deploy, just upload logs and open the report."
- Agent halts and replans instantly.
- Show the ABORT + replan in the action log.

**2:15–2:45** Live demo Workflow A (second half, post-replan):
- GCS upload confirmed (visual precision check ✓)
- Show one deliberate **context-aware visual recovery**: agent misses a button, scrolls, finds it, proceeds. Don't hide this — it proves the agent isn't scripted.

**2:45–3:20** Proof of Cloud:
- Show Cloud Run logs, Firestore session entry with VERIFY actions logged.

**3:20–3:40** Close: what you learned + future improvements (permissions, enterprise policy).

---

## 11) Repo structure (recommended)

```
ui-navigator/
  client/
    main.py
    capture.py          ← includes frame-diff logic
    executor.py         ← includes VERIFY action handler + click-with-verification
    window_focus.py
    redaction.py        ← secret redaction (regex scan + frame masking)
    config.example.json
  server/
    app.py (Cloud Run)
    policy.py
    planner.py
    gemini.py
    schemas.py          ← includes VERIFY action type + max_retries/on_failure fields
  infra/
    cloudrun.yaml (or Dockerfile)
    deploy.sh           ← make deploy script (counts toward automated deployment bonus)
  diagrams/
    architecture.png
  README.md
  LICENSE
```

---

## 12) README.md checklist (what judges want)

- [ ] One-line pitch + gif/screenshot
- [ ] Features list — use phrase **"visual precision"** explicitly (maps directly to judging rubric language)
- [ ] Architecture diagram
- [ ] Setup: local client run steps
- [ ] Setup: backend deploy steps (Cloud Run)
- [ ] Env vars: Gemini key / GCP project
- [ ] Demo workflow instructions (exact steps to reproduce)
- [ ] Safety notes + allowlist behavior
- [ ] Troubleshooting section

> **Framing note:** Use the phrase "visual precision" throughout the submission text and README — it's the exact language in the UI Navigator judging rubric. Also describe recovery moments as "context-aware visual recovery." Same features, language that maps directly to how they're scored.

---

## 13) Optional bonus points (nice boost)

### Blog/video (up to +0.6)
Publish a short build log:
- "Built for #GeminiLiveAgentChallenge"
- Explain architecture and safety design
- Include a clip of interruption handling and a recovery moment

### Automated deployment (up to +0.2)
Add:
- `make deploy` / `deploy.sh` script
- Already in repo structure above — don't forget to include it

---

## 14) Final "Do not forget" list (submission)

- [ ] Public repo link
- [ ] Demo video link (≤ 4 min, English)
- [ ] Architecture diagram included
- [ ] Proof of GCP deployment (screen recording OR code)
- [ ] Testing instructions + credentials if needed (prefer no login)
- [ ] Third-party tools listed explicitly (PyAutoGUI, MSS, etc.)
- [ ] Submission text uses "visual precision" and "context-aware visual recovery" language

---

## 15) Stretch ideas (only if time remains)
- Screen overlay showing detected elements + next action
- "Action confidence" meter
- Multi-monitor support
- Task templates ("Deploy", "Run tests", "Collect logs")
- Sandboxed "practice mode" for safety

---

**End of plan.**
