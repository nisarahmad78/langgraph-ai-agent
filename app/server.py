"""FastAPI server exposing the agent over HTTP.

Endpoints:
    GET  /         -- browser chat UI (static frontend)
    GET  /health   -- service status and whether the LLM is configured
    GET  /tools    -- list the tools the agent can call
    POST /chat     -- chat with the agent (keeps per-session history)

Run with:  uvicorn app.server:app --reload --port 8000
Then open http://localhost:8000/ for the chat UI.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .graph import build_graph, llm_configured
from .tools import TOOLS

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="LangGraph AI Agent",
    description="A tool-using AI agent built with LangGraph, exposed as an API.",
    version="1.0.0",
)

_graph = None  # Compiled lazily on first use.


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Conversation identifier; history is kept per session.")
    message: str = Field(..., description="The user's message.")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tools_used: list[str] = Field(default_factory=list, description="Tools called while producing this answer.")


@app.get("/")
def index() -> FileResponse:
    """Serve the browser chat UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_configured": llm_configured()}


@app.get("/tools")
def list_tools() -> dict:
    return {
        "tools": [
            {"name": t.name, "description": (t.description or "").strip()}
            for t in TOOLS
        ]
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not llm_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM is not configured. Set OPENAI_API_KEY (and optionally "
                "OPENAI_BASE_URL and LLM_MODEL) in the environment or in a "
                ".env file, then restart the server."
            ),
        )
    graph = get_graph()
    config = {"configurable": {"thread_id": request.session_id}}

    # Count existing messages so we only report tools used in this turn.
    existing = graph.get_state(config)
    offset = len(existing.values.get("messages", [])) if existing else 0

    result = graph.invoke(
        {"messages": [{"role": "user", "content": request.message}]},
        config=config,
    )
    new_messages = result["messages"][offset:]

    tools_used: list[str] = []
    for message in new_messages:
        name = getattr(message, "name", None)  # ToolMessage carries the tool name.
        if getattr(message, "type", None) == "tool" and name and name not in tools_used:
            tools_used.append(name)

    answer = result["messages"][-1].content if result["messages"] else ""
    return ChatResponse(
        session_id=request.session_id,
        answer=answer if isinstance(answer, str) else str(answer),
        tools_used=tools_used,
    )


# Static assets for the chat UI (CSS/JS). Mounted last so it never shadows
# the API routes above.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
