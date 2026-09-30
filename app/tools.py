"""Tools available to the agent.

Each tool is implemented as a plain Python function (easy to unit test and to
call without an LLM) and then wrapped with LangChain's ``@tool`` decorator so
it can be bound to a chat model inside the LangGraph graph.

Tools:
    calculator        -- safe arithmetic evaluation (no ``eval``)
    current_datetime  -- current date/time in any IANA timezone
    web_search        -- DuckDuckGo search, free and no API key required
    knowledge_lookup  -- keyword search over a bundled FAQ knowledge base
"""

from __future__ import annotations

import ast
import math
import operator
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node, rejecting anything not whitelisted."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
    ):
        return _FUNCS[node.func.id](*[_eval_node(arg) for arg in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculate_expression(expression: str) -> str:
    """Evaluate an arithmetic expression safely and return the result as text."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:  # malformed or unsupported expression
        return f"Error: could not evaluate '{expression}': {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.

    Supports +, -, *, /, //, %, ** , parentheses, the constants pi, e, tau and
    the functions sqrt, log, log10, sin, cos, tan, abs, round.
    Example input: "(12 * 8) + sqrt(144)".
    """
    return calculate_expression(expression)


# ---------------------------------------------------------------------------
# Current date / time
# ---------------------------------------------------------------------------


def get_current_datetime(timezone_name: str = "UTC") -> str:
    """Return the current date and time for an IANA timezone name."""
    try:
        tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        return f"Error: unknown timezone '{timezone_name}'."
    now = datetime.now(tz)
    return (
        f"{now.strftime('%A, %d %B %Y, %H:%M:%S')} "
        f"{now.tzname()} (timezone: {timezone_name})"
    )


@tool
def current_datetime(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Pass an IANA timezone name such as "UTC", "Asia/Karachi" or
    "America/New_York". Defaults to UTC.
    """
    return get_current_datetime(timezone)


# ---------------------------------------------------------------------------
# Web search (DuckDuckGo -- free, no API key)
# ---------------------------------------------------------------------------


def web_search_impl(query: str, max_results: int = 5) -> str:
    """Search the web with DuckDuckGo and return a compact text summary."""
    try:
        from ddgs import DDGS
    except ImportError:
        return (
            "Web search is unavailable: the 'ddgs' package is not installed. "
            "Install it with: pip install ddgs"
        )
    # Honour a proxy configured in the environment, when present.
    proxy = (
        os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("ALL_PROXY")
        or os.getenv("all_proxy")
    )
    # Try the search backends in order; some are unreachable from restricted
    # networks, so fall through to the next one on failure or empty results.
    results: list[dict] = []
    last_error: Exception | None = None
    for backend in ("duckduckgo", "bing", "yahoo"):
        try:
            with DDGS(proxy=proxy) as client:
                results = list(
                    client.text(query, max_results=max_results, backend=backend)
                )
        except Exception as exc:  # backend unreachable, rate limited, etc.
            last_error = exc
            results = []
        if results:
            break
    if not results:
        if last_error is not None:
            return f"Web search failed: {last_error}"
        return f"No web results found for '{query}'."
    lines = [f"Search results for '{query}':"]
    for item in results:
        title = item.get("title", "").strip()
        body = item.get("body", "").strip()
        href = item.get("href", "").strip()
        lines.append(f"- {title}: {body} ({href})")
    return "\n".join(lines)


@tool
def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo.

    Use this for facts that may have changed recently or that are outside the
    local knowledge base. Input is a plain search query.
    """
    return web_search_impl(query)


# ---------------------------------------------------------------------------
# Local knowledge lookup (bundled FAQ file)
# ---------------------------------------------------------------------------

_FAQ_PATH = Path(__file__).resolve().parent.parent / "data" / "faq.txt"
_STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "how", "what", "which",
    "can", "i", "you", "it", "of", "to", "in", "on", "for", "and", "or",
}


def _load_faq() -> list[tuple[str, str]]:
    """Parse data/faq.txt into (question, answer) pairs."""
    if not _FAQ_PATH.exists():
        return []
    entries: list[tuple[str, str]] = []
    for block in _FAQ_PATH.read_text(encoding="utf-8").split("\n\n"):
        question, answer = "", ""
        for line in block.strip().splitlines():
            if line.startswith("Q:"):
                question = line[2:].strip()
            elif line.startswith("A:"):
                answer = line[2:].strip()
        if question and answer:
            entries.append((question, answer))
    return entries


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in _STOPWORDS
    }


def lookup_knowledge_impl(query: str) -> str:
    """Find the FAQ entry whose question best matches the query keywords."""
    entries = _load_faq()
    if not entries:
        return "The local knowledge base is empty."
    query_tokens = _tokens(query)
    best: tuple[int, str, str] | None = None
    for question, answer in entries:
        score = len(query_tokens & _tokens(question))
        if best is None or score > best[0]:
            best = (score, question, answer)
    if best is None or best[0] == 0:
        return (
            "No matching entry in the local knowledge base. "
            "Try rephrasing, or use web_search for general questions."
        )
    return f"Q: {best[1]}\nA: {best[2]}"


@tool
def knowledge_lookup(query: str) -> str:
    """Look up information in the local knowledge base (a bundled FAQ file).

    Use this first for questions about this project: what it does, its tools,
    how to run it, supported models, sessions and licensing.
    """
    return lookup_knowledge_impl(query)


# All tools, in the order they are bound to the model.
TOOLS = [calculator, current_datetime, web_search, knowledge_lookup]
