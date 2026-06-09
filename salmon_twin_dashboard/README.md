# Salmon Twin AI Operations Dashboard

FastAPI + React/Vite prototype dashboard for a closed-loop salmon farm digital twin.

## Features

- Omniverse WebRTC stream area
- Recent 1-hour sensor query from SQL DB
- Raw sensor trend chart
- LLM + RAG chat through Ollama
- AI action proposal with confirm/decline
- AI operations report generation
- Dark/light mode

## Architecture

```text
DB sensor table -> FastAPI -> React Dashboard
                     |          |
                     |          -> confirm/decline UI
                     v
              Ollama Gemma + optional RAG endpoint
                     |
                     v
              action payload for Omniverse controller
```

## 1. Configure backend

```bash
cd backend
cp .env.example .env
```

Edit `.env`:

```env
DATABASE_URL=sqlite:///./sensor.db
SENSOR_TABLE=sensor_readings
TIME_COLUMN=timestamp
SENSOR_COLUMNS=temperature,do,ph,ammonia,salinity,turbidity
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4
RAG_ENDPOINT=
CORS_ORIGIN=http://localhost:5173
```

For Postgres:

```env
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/salmon
```

If running with Docker and Ollama is on the host machine, use:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

## 2. Configure frontend

```bash
cd frontend
cp .env.example .env
```

Edit `.env`:

```env
VITE_API_BASE=http://localhost:8000
VITE_OMNIVERSE_STREAM_URL=http://localhost:8011/streaming/webrtc-client
```

Replace `VITE_OMNIVERSE_STREAM_URL` with your actual Omniverse WebRTC Streaming Client URL.

## 3. Run locally without Docker

Terminal 1:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## 4. Run with Docker Compose

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
# edit both .env files

docker compose up --build
```

## 5. Required DB shape

Your sensor table should have a timestamp column and six numeric columns.

Example:

```sql
CREATE TABLE sensor_readings (
  timestamp TEXT NOT NULL,
  temperature REAL,
  do REAL,
  ph REAL,
  ammonia REAL,
  salinity REAL,
  turbidity REAL
);
```

The backend runs this query pattern:

```sql
SELECT timestamp, temperature, do, ph, ammonia, salinity, turbidity
FROM sensor_readings
WHERE timestamp >= :cutoff
ORDER BY timestamp ASC
LIMIT :limit;
```

If your table/column names differ, edit `SENSOR_TABLE`, `TIME_COLUMN`, and `SENSOR_COLUMNS` in `backend/.env`.

## 6. Omniverse control integration point

The confirm endpoint currently returns a payload like:

```json
{
  "status": "confirmed",
  "control_payload": {
    "actions": [
      {
        "variable": "temperature",
        "direction": "down",
        "amount": 0.5,
        "unit": "C",
        "rationale": "..."
      }
    ]
  }
}
```

Replace the stub in `backend/app/main.py` inside `/api/ai/decision` with your actual Omniverse bridge:

- WebSocket to Kit extension
- MQTT command publish
- DB command table insert
- HTTP call to your Omniverse controller process

Recommended command-table schema for quick prototype:

```sql
CREATE TABLE control_commands (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  payload_json TEXT NOT NULL
);
```

Then Omniverse can poll `pending` commands every 0.5 s and apply variable up/down changes.

## 7. RAG endpoint contract

If your RAG pipeline already has an HTTP server, set `RAG_ENDPOINT`.

Expected request:

```json
{"query": "..."}
```

Accepted response examples:

```json
{"context": "retrieved guideline text..."}
```

or:

```json
{"answer": "retrieved answer..."}
```

If `RAG_ENDPOINT` is empty, the app still works with Ollama only.
