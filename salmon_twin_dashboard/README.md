# Salmon Twin AI Operations Dashboard

FastAPI + React/Vite prototype dashboard for a closed-loop salmon indoor aquaculture digital twin.

This dashboard is designed for the following workflow:

```text
Omniverse digital twin
  -> virtual sensor DB
  -> FastAPI backend
  -> Ollama Gemma + optional RAG
  -> AI action proposal
  -> confirm / decline
  -> Omniverse control payload
  -> updated digital twin state
```

## Features

- Omniverse WebRTC stream display area
- Recent 1-hour sensor query from SQL DB
- Raw sensor trend chart
- LLM + RAG chat through Ollama
- AI action proposal for six controllable variables
- Confirm / decline workflow
- AI operations report generation button
- Dark / light mode

## Project structure

```text
salmon_twin_dashboard/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── db.py
│   │   ├── ollama_client.py
│   │   ├── config.py
│   │   └── schemas.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   ├── index.html
│   └── .env.example
├── docker-compose.yml
└── README.md
```

> If `.env.example` files are missing in your copied directory, create the `.env` files manually using the templates below.

---

## 1. Prerequisites

### Backend

- Python 3.10+
- Running SQL database or local SQLite file
- Ollama running locally or on a reachable host
- Pulled model, for example `gemma4`

Example:

```bash
ollama list
ollama run gemma4
```

### Frontend

Use Node.js **20.19+** or **22.12+**. Node 18 will not work with recent Vite versions.

Recommended:

```bash
node -v
npm -v
```

Expected example:

```text
v22.22.3
10.9.8
```

If you use `nvm`, Node should resolve to an nvm path, for example:

```text
/home/user/.nvm/versions/node/v22.22.3/bin/node
```

---

## 2. Configure backend

From the project root:

```bash
cd backend
cp .env.example .env
```

If `.env.example` is missing, create `backend/.env` manually:

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

Adjust these values to your actual DB schema.

### PostgreSQL example

```env
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/salmon
```

### MySQL example

```env
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/salmon
```

### Docker + Ollama on host machine

If you run the backend in Docker but Ollama is running on the host OS, use:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

---

## 3. Configure frontend

From the project root:

```bash
cd frontend
cp .env.example .env
```

If `.env.example` is missing, create `frontend/.env` manually:

```env
VITE_API_BASE=http://localhost:8000
VITE_OMNIVERSE_STREAM_URL=http://localhost:8011/streaming/webrtc-client
```

Replace `VITE_OMNIVERSE_STREAM_URL` with your actual Omniverse WebRTC Streaming Client URL.

---

## 4. Run locally without Docker

### Terminal 1: backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API should be available at:

```text
http://localhost:8000
```

### Terminal 2: frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

---

## 5. Important Node/Vite troubleshooting

### Symptom

You may see this error when running `npm run dev`:

```text
You are using Node.js 18.19.1.
Vite requires Node.js version 20.19+ or 22.12+.
Please upgrade your Node.js version.

ReferenceError: CustomEvent is not defined
```

This is not caused by dashboard code or `.env` settings. It means Vite is being executed with Node 18, or `node_modules` was installed under an old Node environment.

### Step 1: verify the actual Node path

Run this inside `frontend/`:

```bash
which node
node -v
which npm
npm -v
```

A working setup should look similar to:

```text
/home/user/.nvm/versions/node/v22.22.3/bin/node
v22.22.3
/home/user/.nvm/versions/node/v22.22.3/bin/npm
10.9.8
```

### Step 2: reinstall frontend dependencies

If Node is already v22 but the error still appears, delete the old install artifacts and reinstall:

```bash
cd frontend
rm -rf node_modules package-lock.json
npm cache verify
npm install
npm run dev
```

This fixed the observed issue where the shell pointed to Node v22, but the frontend still failed with a Node 18/Vite error.

### Step 3: optional nvm version pin

To prevent the project from accidentally using Node 18 again:

```bash
cd frontend
echo "22" > .nvmrc
nvm use
```

If `nvm use` is not available, reload your shell:

```bash
source ~/.bashrc
```

or, if using zsh:

```bash
source ~/.zshrc
```

### Step 4: diagnostic command

If the error persists, check what Node npm uses internally:

```bash
cd frontend
npm exec node -v
npm exec vite --version
```

If `npm exec node -v` shows `v18.x.x`, your npm execution environment is still bound to Node 18.

---

## 6. Run with Docker Compose

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
# edit both .env files

docker compose up --build
```

If the `.env.example` files are missing, create `backend/.env` and `frontend/.env` manually using the templates in sections 2 and 3.

---

## 7. Required DB shape

Your sensor table should have one timestamp column and six numeric sensor columns.

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

The backend uses this query pattern:

```sql
SELECT timestamp, temperature, do, ph, ammonia, salinity, turbidity
FROM sensor_readings
WHERE timestamp >= :cutoff
ORDER BY timestamp ASC
LIMIT :limit;
```

If your actual DB uses different names, update these values in `backend/.env`:

```env
SENSOR_TABLE=your_table_name
TIME_COLUMN=your_time_column
SENSOR_COLUMNS=temperature,dissolved_oxygen,ph,ammonia,salinity,turbidity
```

Important: the names in `SENSOR_COLUMNS` must match the actual DB column names exactly.

---

## 8. Omniverse control integration point

The confirm endpoint returns a payload like this:

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
        "rationale": "High water temperature is increasing salmon stress."
      }
    ]
  }
}
```

Replace the stub in `backend/app/main.py` inside `/api/ai/decision` with your actual Omniverse bridge.

Recommended options:

- WebSocket to an Omniverse Kit extension
- MQTT command publish
- DB command-table insert
- HTTP call to your Omniverse controller process

Recommended command-table schema for a quick prototype:

```sql
CREATE TABLE control_commands (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  payload_json TEXT NOT NULL
);
```

Then the Omniverse side can poll `pending` commands every 0.5 seconds and apply variable up/down changes.

---

## 9. RAG endpoint contract

If your RAG pipeline already has an HTTP server, set `RAG_ENDPOINT` in `backend/.env`.

Expected request:

```json
{
  "query": "current salmon farm water quality condition and recommended operation"
}
```

Accepted response examples:

```json
{
  "context": "retrieved guideline text..."
}
```

or:

```json
{
  "answer": "retrieved answer..."
}
```

If `RAG_ENDPOINT` is empty, the app still works with Ollama only.

---

## 10. Suggested development order

1. Start backend and verify `/api/health`.
2. Start frontend and verify dashboard loads.
3. Check the recent 1-hour DB query.
4. Confirm the raw sensor chart updates.
5. Test LLM chat with Ollama only.
6. Connect RAG endpoint.
7. Generate an AI action proposal.
8. Confirm the proposal and inspect the returned control payload.
9. Connect the payload to Omniverse through your preferred bridge.
