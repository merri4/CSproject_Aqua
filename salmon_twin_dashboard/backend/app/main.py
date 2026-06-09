from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import append_action_log, get_latest_sensor_row, get_recent_sensor_rows, sensor_columns
from .ollama_client import chat_with_operator, generate_report, propose_actions
from .schemas import ActionDecision, ActionProposal, ChatRequest, ChatResponse, ControlPayload, ReportRequest, ReportResponse, SensorResponse

app = FastAPI(title="Salmon Twin AI Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CORS_ORIGIN, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_proposal: ActionProposal | None = None


@app.get("/health")
def health():
    return {"ok": True, "model": settings.OLLAMA_MODEL, "sensor_columns": sensor_columns()}


@app.get("/api/sensors/recent", response_model=SensorResponse)
def recent_sensors(hours: float = 1.0, limit: int = 7200):
    rows = get_recent_sensor_rows(hours=hours, limit=limit)
    return SensorResponse(columns=[settings.TIME_COLUMN] + sensor_columns(), rows=rows)


@app.get("/api/sensors/latest")
def latest_sensor():
    row = get_latest_sensor_row()
    if not row:
        raise HTTPException(status_code=404, detail="No sensor data found")
    return row


@app.post("/api/ai/propose", response_model=ActionProposal)
async def ai_propose():
    global last_proposal
    rows = get_recent_sensor_rows(hours=1.0, limit=7200)
    last_proposal = await propose_actions(rows, sensor_columns())
    return last_proposal


@app.post("/api/ai/decision")
def ai_decision(payload: ActionDecision):
    append_action_log(payload.proposal.model_dump(), payload.decision)
    if payload.decision == "decline":
        return {"status": "declined", "message": "Proposal logged but no control action was applied."}

    # Important integration point:
    # Replace this stub with your Omniverse/Kit bridge, WebSocket, MQTT, or DB command table.
    # The frontend receives the action and can also call your Omniverse controller separately.
    control_payload = ControlPayload(actions=payload.proposal.actions)
    return {
        "status": "confirmed",
        "message": "Proposal confirmed. Send this payload to Omniverse controller.",
        "control_payload": control_payload.model_dump(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    rows = get_recent_sensor_rows(hours=1.0, limit=7200)
    reply = await chat_with_operator(req.message, rows)
    return ChatResponse(reply=reply)


@app.post("/api/report", response_model=ReportResponse)
async def report(req: ReportRequest):
    rows = get_recent_sensor_rows(hours=1.0, limit=7200)
    proposal = last_proposal or await propose_actions(rows, sensor_columns())
    text = await generate_report(rows, proposal, req.operator_note)
    return ReportResponse(report=text, proposal=proposal)
