from __future__ import annotations

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import append_action_log, append_control_command, database_status, get_latest_sensor_row, get_recent_sensor_rows, sensor_columns
from .ollama_client import chat_with_operator, generate_report, ollama_status, propose_actions
from .rag_adapter import rag_status
from .schemas import ActionDecision, ActionProposal, ChatRequest, ChatResponse, ControlPayload, ReportRequest, ReportResponse, SensorResponse

app = FastAPI(title="Salmon Twin AI Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CORS_ORIGIN, "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_proposal: ActionProposal | None = None


@app.get("/health")
async def health():
    return {"ok": True, "model": settings.OLLAMA_MODEL, "sensor_columns": sensor_columns()}


@app.get("/api/status")
async def status():
    return {
        "db": database_status(),
        "llm": await ollama_status(),
        "rag": rag_status(),
        "omniverse": {
            "stream_url_configured": True,
            "control_endpoint": settings.OMNIVERSE_CONTROL_ENDPOINT or None,
            "control_queue": "control_commands",
        },
    }


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
async def ai_decision(payload: ActionDecision):
    append_action_log(payload.proposal.model_dump(), payload.decision)
    if payload.decision == "decline":
        return {"status": "declined", "message": "Proposal logged but no control action was applied."}

    control_payload = ControlPayload(actions=payload.proposal.actions).model_dump()
    command_id = append_control_command(control_payload)
    bridge_result = None

    if settings.OMNIVERSE_CONTROL_ENDPOINT:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                res = await client.post(settings.OMNIVERSE_CONTROL_ENDPOINT, json=control_payload)
                res.raise_for_status()
                try:
                    bridge_result = res.json()
                except Exception:
                    bridge_result = {"text": res.text}
        except Exception as exc:
            bridge_result = {"error": str(exc)}

    return {
        "status": "confirmed",
        "message": "Proposal confirmed and queued for Omniverse control.",
        "command_id": command_id,
        "control_payload": control_payload,
        "bridge_result": bridge_result,
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
