# 📌 Epital ElderGuard – Care Assistance API

**Clinically-inspired caregiver support endpoints for fast fall response workflows.**

---

## 📖 Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [License](#license)

---

## 🔥 Introduction

Epital ElderGuard’s production system blends on-device fall-detection models with a caregiver co-pilot. This repo showcases the **public-facing caregiver assistance API**: a FastAPI service that taps DuckDuckGo’s Instant Answer API to deliver contextual health information and emergency facility lookups. It’s intentionally lightweight to let visitors try the endpoints live.

---

## 🚀 Key Features

✅ **FastAPI microservice** with two curated caregiver endpoints  
✅ **DuckDuckGo Instant Answer integration** (no proprietary data required)  
✅ **Clean `.gitignore` & slim repo** for painless GitHub/portfolio sharing  
✅ **One-command local run** via `uvicorn`  
✅ **Edge-friendly narrative**—ties into the broader ElderGuard story without shipping private artifacts

---

## 🛠 Tech Stack

- **Backend:** FastAPI, Pydantic
- **Search integration:** DuckDuckGo Instant Answer API (requests)
- **Runtime:** Uvicorn
- **Packaging:** Python 3.10+, virtualenv

---

## ⚡ Quick Start

```bash
git clone https://github.com/petitmj/epital-elderguard.git
cd epital-elderguard
python -m venv .venv && .\.venv\Scripts\activate  # Windows
# source .venv/bin/activate                      # macOS/Linux
pip install -r requirements.txt
uvicorn care_assist_api:app --reload
```

The service boots at `http://127.0.0.1:8000`. Visit `/docs` for the interactive Swagger UI.

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`  | Service metadata & navigation hints |
| `POST` | `/care-info` | Provide a caregiving topic to receive curated health guidance cards |
| `POST` | `/emergency-facilities` | Provide a location query to surface nearby emergency resources |

### Example payloads

```json
POST /care-info
{
  "topic": "fall prevention checklist",
  "audience": "elderly caregivers",
  "max_results": 5
}
```

```json
POST /emergency-facilities
{
  "location_query": "Nairobi, Kenya",
  "max_results": 3
}
```

Each endpoint returns normalized `InfoCard` objects: `{ "title", "snippet", "url", "metadata" }`.

---

## 🚀 Deployment

### Local / Demo

1. Follow **Quick Start**.
2. Keep the server running and expose it via `ngrok` or a similar tunnel for quick demos.

### Portfolio Hosting

1. Push this repo to GitHub (the `.gitignore` already excludes bulky artifacts).
2. Deploy to a free-tier host (Render, Fly.io, Railway, etc.):
   - Set the start command to `uvicorn care_assist_api:app --host 0.0.0.0 --port 8000`.
   - Add `PORT` env var if required by the host.
3. Embed the live API link or Swagger UI iframe on your portfolio page so visitors can try it instantly.

---

## 🗂️ Project Structure

```text
.
├── care_assist_api.py      # FastAPI application exposing caregiver endpoints
├── duckduckgo_service.py   # DuckDuckGo Instant Answer client + InfoCard helpers
├── requirements.txt        # Minimal runtime dependencies
├── LICENSE
├── .gitignore              # Keeps notebooks, datasets, and models out of Git
└── README.md               # You are here
```

> Looking for the training notebooks or Snapdragon deployment scripts? Those live in private/internal repos. This public snapshot focuses on the reproducible experience that complements your portfolio showcase.

---

## License

**MIT License** – free to use, remix, and extend.

---

🚀 **Show prospective collaborators how ElderGuard supports caregivers—even without the heavyweight training stack.**  
