from pydantic import BaseModel, Field
from typing import Any, Literal


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


class ActionItem(BaseModel):
    variable: str
    direction: Literal["up", "down"]
    amount: float
    unit: str | None = None
    rationale: str


class ActionProposal(BaseModel):
    summary: str
    risk_level: Literal["normal", "watch", "warning", "critical"] = "watch"
    actions: list[ActionItem] = Field(default_factory=list)
    raw_model_output: str | None = None


class ActionDecision(BaseModel):
    proposal: ActionProposal
    decision: Literal["confirm", "decline"]


class ControlPayload(BaseModel):
    actions: list[ActionItem]


class ReportRequest(BaseModel):
    operator_note: str | None = None


class ReportResponse(BaseModel):
    report: str
    proposal: ActionProposal | None = None


class SensorResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
