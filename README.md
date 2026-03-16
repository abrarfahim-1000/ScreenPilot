# ScreenPilot - See it. Plan it. Do it

A cross-application desktop agent that treats your screen as a canvas. Give it a high-level instruction — *"Run my tests, upload the logs, and deploy to Cloud Run"* — and it visually interprets screenshots, plans a sequence of OS-level actions, executes them (click, type, hotkeys), verifies each step by re-observing the screen (**visual precision checks**), and recovers automatically from UI changes or errors (**context-aware visual recovery**).

Built for the **Gemini Live Agent Challenge 2026** (UI Navigator category).

**Architecture:** Desktop client (Python) → Cloud Run orchestrator → Gemini multimodal → actions back to desktop. Session state in Firestore, artifacts in Cloud Storage.

---

## How it works

1. The desktop client captures your screen every 2–3 seconds and sends JPEG frames to the Cloud Run backend
2. The backend calls Gemini (vision + text) to perceive the screen and plan the next actions
3. Actions are sent back to the client and executed — clicks, keystrokes, terminal commands, file reads
4. Every major action is followed by a `VERIFY` step; failures trigger a bounded recovery playbook before handing off to the user
5. Secrets are scanned and redacted before any text leaves the client

---

## Prerequisites

- Python 3.11+
- A [Gemini API key](https://aistudio.google.com/app/apikey)
- `make` installed (for `make test`) — Windows: `winget install GnuWin32.Make`

---

## Quick start

### 1 — Clone and install dependencies

```bash
git clone https://github.com/abrarfahim-1000/local-to-cloud
cd local-to-cloud
pip install -r requirements.txt
```

### 2 — Start the client

The backend is already deployed to Google Cloud Run — you do not need to run the server locally.

```bash
python client/main.py
```

On first **Start Session**, a dialog will ask for your Gemini API key. The key is saved once to the OS credential store (Windows Credential Manager / macOS Keychain / Linux Secret Service) and is not requested again.

The **Server URL** field is pre-set to the live Cloud Run deployment:
```
https://ui-navigator-314272999720.asia-southeast1.run.app
```

### 3 — Verify the backend is reachable

```bash
curl https://ui-navigator-314272999720.asia-southeast1.run.app/health
# → {"status":"ok","version":"0.1.0"}
```

### 4 — Run a session

1. Enter a task goal in the **Task Goal** field, for example:
   ```
   Run make test, upload the log to GCS, and deploy to Cloud Run
   ```
2. Click **▶ Start Session**
3. Watch the Action Log — the agent will perceive your screen, plan actions, and execute them live

---

## Run the test suite

```bash
make test
# or directly:
pytest tests/ -v --tb=short
```

530 tests across 8 modules — capture, compression, redaction, log parsing, command policy, server endpoints, verification loop, and cloud integration — all run offline with no real GCP credentials needed.

---

## Run the server locally (optional)

If you want to run the backend on your own machine instead of using the Cloud Run deployment:

```bash
# Set your Gemini API key
export GEMINI_API_KEY="AIza..."          # macOS / Linux
$env:GEMINI_API_KEY = "AIza..."          # Windows PowerShell

# Start the server
uvicorn server.app:app --port 8080 --reload
```

Then change the Server URL in the client UI to `http://localhost:8080`.

---

## Deploy to Cloud Run (re-deploy after changes)

```bash
make deploy
```

Or manually:

```bash
gcloud run deploy ui-navigator \
  --source . \
  --region asia-southeast1 \
  --platform managed \
  --no-allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=localtocloud-489708,GCS_BUCKET=ltc-12,GEMINI_MODEL=gemini-2.5-flash" \
  --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
  --memory 512Mi \
  --timeout 120
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes (client) | Forwarded per-request via `X-Gemini-Api-Key` header — never stored server-side |
| `GOOGLE_CLOUD_PROJECT` | Server only | GCP project ID for Firestore and GCS |
| `GCS_BUCKET` | Server only | GCS bucket name for log and artifact uploads |
| `GEMINI_MODEL` | Optional | Gemini model override (default: `gemini-2.5-flash`) |
| `NAVIGATOR_SERVER` | Optional | Pre-populate the Server URL field on client startup |
| `NAVIGATOR_GOAL` | Optional | Pre-populate the Task Goal field on client startup |

---

## Project structure

```
ui-navigator/
  client/
    main.py           ← entry point, Qt app
    capture.py        ← screen capture + frame diff
    executor.py       ← action execution + click-with-verification
    session.py        ← perception-action loop (QThread)
    redaction.py      ← secret scanning + frame masking
    command_policy.py ← terminal command allowlist / blocklist
    log_parser.py     ← test log parsing + deployment report
    window_focus.py   ← cross-platform window focus
    keystore.py       ← OS credential store wrapper
    ui.py             ← PyQt6 control panel
  server/
    app.py            ← Cloud Run FastAPI orchestrator
    gemini.py         ← Gemini perception + planning client
    schemas.py        ← Pydantic action schemas
    firestore_session.py ← session state + audit log
    gcs_storage.py    ← GCS artifact store
  tests/              ← 530 tests, all offline
  Makefile
  Dockerfile
  requirements.txt
```

---

## Safety

- **Secret redaction:** Every `TYPE` action is scanned for credentials (API keys, tokens, PEM blocks, JWTs) before execution. Detected secrets are replaced with `[REDACTED:pattern_name]` and the action is blocked — secrets never reach the clipboard, the terminal, or the cloud.
- **Command policy:** Terminal commands pass through a two-tier allowlist/blocklist gate. `rm -rf`, `shutdown`, `gcloud delete`, and `curl | bash` are hard-blocked regardless of any other setting.
- **Destructive action confirmation:** `DEPLOY_CLOUD_RUN` always requires explicit user confirmation via a dialog before the `gcloud` command is executed.
- **Bounded recovery:** VERIFY failures trigger a deterministic recovery playbook (scroll → zoom → tab switch → address bar). After `max_retries` the agent hands off to the user with a full summary — it never loops indefinitely.
