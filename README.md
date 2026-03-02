# Local To Cloud

## Prerequisites

- Python 3.11+
- A [Gemini API key](https://aistudio.google.com/app/apikey)

---

## 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2 — Start the server

```bash
cd server
uvicorn app:app --port 8080 --reload
```

The server does **not** need a `GEMINI_API_KEY` environment variable — the key is forwarded per-request by the desktop client.

To verify it is running:

```bash
curl http://localhost:8080/health
# → {"status":"ok","version":"0.1.0"}
```

---

## 3 — Start the UI

Open a second terminal:

```bash
cd client
python main.py
```

On first **Start Session**, a dialog will ask for your Gemini API key. The key is saved once to the OS credential store (Windows Credential Manager / macOS Keychain / Linux Secret Service) and is not requested again on subsequent starts.
