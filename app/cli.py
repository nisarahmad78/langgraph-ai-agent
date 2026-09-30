"""Command-line interface for the agent.

Usage:
    python -m app.cli "What is 25 * 17 + sqrt(81)?"
    python -m app.cli --session my-chat "And what time is it in Tokyo?"

Conversation history is kept for the given --session id for the duration of
the process when used interactively via repeated calls in code; the CLI runs
one question per invocation by default. Use --interactive for a REPL that
keeps a single session alive across questions.
"""

from __future__ import annotations

import argparse
import sys

from .graph import build_graph, llm_configured


def _ask(graph, session_id: str, question: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": question}]}, config=config
    )
    tools_used = [
        m.name
        for m in result["messages"]
        if getattr(m, "type", None) == "tool" and getattr(m, "name", None)
    ]
    answer = result["messages"][-1].content
    return (answer if isinstance(answer, str) else str(answer)), tools_used


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="app.cli", description="Chat with the LangGraph AI Agent."
    )
    parser.add_argument("question", nargs="?", help="The question to ask.")
    parser.add_argument(
        "--session", default="cli", help="Session id (default: 'cli')."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive session (type 'exit' to quit).",
    )
    args = parser.parse_args(argv)

    if not llm_configured():
        print(
            "Error: the LLM is not configured.\n"
            "Copy .env.example to .env and set OPENAI_API_KEY (plus optional "
            "OPENAI_BASE_URL / LLM_MODEL), then try again.",
            file=sys.stderr,
        )
        return 1

    graph = build_graph()

    if args.interactive:
        print("Interactive session started. Type 'exit' to quit.")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            answer, tools_used = _ask(graph, args.session, question)
            if tools_used:
                print(f"[tools used: {', '.join(dict.fromkeys(tools_used))}]")
            print(f"Agent: {answer}")
        return 0

    if not args.question:
        parser.error("a question is required (or use --interactive)")

    answer, tools_used = _ask(graph, args.session, args.question)
    if tools_used:
        print(f"[tools used: {', '.join(dict.fromkeys(tools_used))}]")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
