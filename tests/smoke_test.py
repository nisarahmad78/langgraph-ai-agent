"""Smoke tests that run without an LLM API key.

Covers:
  * calculator, datetime and knowledge-lookup tools (direct calls)
  * graph construction (import + compile)
  * FastAPI TestClient: /health, /tools, the web UI at /, and the graceful
    /chat error returned when the LLM is not configured

Run from the project root:  python tests/smoke_test.py
Exits with code 0 when everything passes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure no real key leaks into the test run: the point is to verify the
# no-key behaviour, so drop the variable if the shell happens to export one.
os.environ.pop("OPENAI_API_KEY", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def main() -> int:
    from app.tools import (
        calculate_expression,
        get_current_datetime,
        lookup_knowledge_impl,
    )

    # --- Calculator -------------------------------------------------------
    check("calculator: precedence", calculate_expression("2 + 3 * 4") == "14")
    check("calculator: parentheses + sqrt",
          calculate_expression("(12 * 8) + sqrt(144)") == "108")
    check("calculator: power", calculate_expression("2 ** 10") == "1024")
    check("calculator: division by zero is a graceful error",
          calculate_expression("1 / 0").startswith("Error"))
    check("calculator: rejects code injection",
          calculate_expression("__import__('os')").startswith("Error"))

    # --- Datetime ----------------------------------------------------------
    dt = get_current_datetime("Asia/Karachi")
    check("datetime: returns a value for Asia/Karachi",
          "Asia/Karachi" in dt and not dt.startswith("Error"), dt)
    check("datetime: bad timezone is a graceful error",
          get_current_datetime("Not/AZone").startswith("Error"))

    # --- Knowledge lookup --------------------------------------------------
    kb = lookup_knowledge_impl("What tools does the agent have?")
    check("knowledge lookup: finds the tools entry",
          "calculator" in kb.lower(), kb[:80])
    kb2 = lookup_knowledge_impl("What is the license of this project?")
    check("knowledge lookup: finds the license entry", "MIT" in kb2, kb2[:80])

    # --- Graph construction ------------------------------------------------
    from app.graph import build_graph, llm_configured

    check("llm_configured() is False without a key", llm_configured() is False)
    graph = build_graph()
    check("graph compiles without an API key", graph is not None)

    # --- FastAPI via TestClient -------------------------------------------
    from fastapi.testclient import TestClient

    from app.server import app

    client = TestClient(app)
    health = client.get("/health")
    check("GET /health -> 200", health.status_code == 200)
    check("GET /health reports llm_configured=false",
          health.json().get("llm_configured") is False)

    tools_resp = client.get("/tools")
    names = [t["name"] for t in tools_resp.json().get("tools", [])]
    check("GET /tools lists all four tools",
          names == ["calculator", "current_datetime", "web_search", "knowledge_lookup"],
          str(names))

    chat = client.post("/chat", json={"session_id": "smoke", "message": "hello"})
    check("POST /chat without a key -> 503 graceful error",
          chat.status_code == 503 and "OPENAI_API_KEY" in chat.json().get("detail", ""),
          f"status={chat.status_code} body={chat.text[:120]}")

    # --- Web UI -------------------------------------------------------------
    root = client.get("/")
    check("GET / serves the chat UI",
          root.status_code == 200 and "LangGraph AI Agent" in root.text
          and 'id="messages"' in root.text,
          f"status={root.status_code}")
    js = client.get("/static/app.js")
    check("GET /static/app.js -> 200", js.status_code == 200,
          f"status={js.status_code}")
    css = client.get("/static/style.css")
    check("GET /static/style.css -> 200", css.status_code == 200,
          f"status={css.status_code}")

    print()
    if FAILURES:
        print(f"SMOKE TESTS FAILED: {len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
