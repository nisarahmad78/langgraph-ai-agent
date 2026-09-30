r"""LangGraph agent graph definition.

Graph layout:

    START -> agent --(tool calls)--> tools -> agent -> ... -> END
                  \--(no tool calls)--> END

The ``agent`` node is a chat model with the tools bound to it. The ``tools``
node is LangGraph's prebuilt ``ToolNode``. Conversation state is kept per
session with an in-memory checkpointer (``MemorySaver``), keyed by the
``thread_id`` passed at invoke time.
"""

from __future__ import annotations

import os

try:  # Load a local .env file when python-dotenv is installed.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is in requirements.txt
    pass

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .tools import TOOLS

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

Guidelines:
- Use the calculator tool for any arithmetic instead of computing mentally.
- Use current_datetime when the user asks about the date or time.
- Use knowledge_lookup first for questions about this project itself
  (features, tools, setup, sessions, license).
- Use web_search for current events or facts outside the knowledge base.
- Answer concisely and mention which tool you used when it matters.
"""

DEFAULT_MODEL = "gpt-4o-mini"


def llm_configured() -> bool:
    """True when an API key for the LLM provider is present in the environment."""
    return bool(os.getenv("OPENAI_API_KEY"))


def build_model() -> ChatOpenAI:
    """Create the chat model from environment configuration.

    Works with any OpenAI-compatible API: set OPENAI_BASE_URL to point at a
    different endpoint and LLM_MODEL to choose the model. A placeholder key is
    used when none is configured so the graph can still be constructed (for
    example in tests); calls will fail until a real key is provided.
    """
    kwargs: dict = {
        "model": os.getenv("LLM_MODEL", DEFAULT_MODEL),
        "api_key": os.getenv("OPENAI_API_KEY") or "not-configured",
        "base_url": os.getenv("OPENAI_BASE_URL") or None,
        "temperature": 0,
    }
    try:
        return ChatOpenAI(**kwargs)
    except Exception:
        # Some sandboxed environments export proxy variables that httpx
        # cannot parse, which breaks construction of the default client.
        # Retry with explicit clients that ignore environment proxies.
        import httpx

        kwargs["http_client"] = httpx.Client(trust_env=False)
        kwargs["http_async_client"] = httpx.AsyncClient(trust_env=False)
        return ChatOpenAI(**kwargs)


def build_graph():
    """Build and compile the agent graph with in-memory session state."""
    model = build_model().bind_tools(TOOLS)

    def call_model(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [model.invoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    # tools_condition routes to "tools" when the model requested tool calls,
    # and to END when it produced a final answer.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=MemorySaver())
