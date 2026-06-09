from __future__ import annotations

import json
from typing import Any

import httpx

from .config import settings
from .schemas import ActionProposal


ACTION_SYSTEM_PROMPT = """
You are an expert operator for indoor salmon aquaculture water-quality management.
You receive recent sensor data and RAG context. Decide whether controlled variables need adjustment.
Return STRICT JSON only. No markdown. No comments.

Schema:
{
  "summary": "short Korean summary for operator",
  "risk_level": "normal|watch|warning|critical",
  "actions": [
    {
      "variable": "one of the provided variable names",
      "direction": "up|down",
      "amount": number,
      "unit": "optional unit string",
      "rationale": "Korean rationale"
    }
  ]
}

Rules:
- Do not invent variables outside the provided variable list.
- Keep action amount conservative. This is a prototype closed-loop controller.
- If state is stable, return an empty actions list.
- Prefer actionable operating controls, not vague advice.
""".strip()


REPORT_SYSTEM_PROMPT = """
You are an expert report writer for indoor salmon aquaculture operations.
Write a concise Korean operations report based on recent sensor data, RAG context, and proposed actions.
Include: current state, likely cause, risk, recommended action, expected feedback effect, and follow-up check.
""".strip()


async def rag_context(query: str) -> str:
    if not settings.RAG_ENDPOINT:
        return ""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            res = await client.post(settings.RAG_ENDPOINT, json={"query": query})
            res.raise_for_status()
            try:
                payload = res.json()
                if isinstance(payload, dict):
                    return str(payload.get("context") or payload.get("answer") or payload)
                return str(payload)
            except Exception:
                return res.text
    except Exception as exc:
        return f"[RAG unavailable: {exc}]"


async def ollama_chat(messages: list[dict[str, str]], temperature: float = 0.2) -> str:
    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    async with httpx.AsyncClient(timeout=90) as client:
        res = await client.post(url, json=payload)
        res.raise_for_status()
        data = res.json()
        return data.get("message", {}).get("content", "")


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("JSON\n", "", 1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


async def propose_actions(sensor_rows: list[dict[str, Any]], variable_names: list[str]) -> ActionProposal:
    latest = sensor_rows[-1] if sensor_rows else {}
    query = f"연어 실내양식장 현재 수질 상태 평가 및 제어 조치. 최신 센서값: {latest}"
    context = await rag_context(query)

    user_prompt = {
        "variable_names": variable_names,
        "latest_sensor": latest,
        "recent_sensor_sample": sensor_rows[-120:],
        "rag_context": context[:5000],
        "task": "현재 상태를 평가하고 조정할 변수를 결정하라. 반드시 JSON만 출력하라.",
    }
    raw = await ollama_chat(
        [
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        temperature=0.1,
    )
    try:
        parsed = _extract_json(raw)
        parsed["raw_model_output"] = raw
        return ActionProposal(**parsed)
    except Exception:
        return ActionProposal(
            summary="모델 출력 JSON 파싱에 실패했습니다. 원문을 확인하세요.",
            risk_level="watch",
            actions=[],
            raw_model_output=raw,
        )


async def chat_with_operator(message: str, sensor_rows: list[dict[str, Any]]) -> str:
    latest = sensor_rows[-1] if sensor_rows else {}
    context = await rag_context(message)
    return await ollama_chat(
        [
            {"role": "system", "content": "연어 실내양식장 운영 전문가처럼 한국어로 간결하고 실행 가능하게 답하라."},
            {"role": "user", "content": json.dumps({"message": message, "latest_sensor": latest, "rag_context": context[:5000]}, ensure_ascii=False)},
        ],
        temperature=0.3,
    )


async def generate_report(sensor_rows: list[dict[str, Any]], proposal: ActionProposal | None, operator_note: str | None) -> str:
    context = await rag_context("연어 실내양식장 수질 관리 리포트 작성 기준")
    prompt = {
        "recent_sensor_rows": sensor_rows[-240:],
        "proposal": proposal.model_dump() if proposal else None,
        "operator_note": operator_note,
        "rag_context": context[:5000],
    }
    return await ollama_chat(
        [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.25,
    )
