# LangGraph AI Agent

A tool-using AI agent built with **LangGraph** and **LangChain**. The agent reasons about each question, decides which tools it needs, calls them, and uses the results to compose a final answer. It runs in three modes: a **browser chat UI**, a **command-line chat**, and a **FastAPI web service** with per-session conversation memory.

## Screenshots

![LangGraph AI Agent UI](docs/screenshot.png)

The browser chat UI: the sidebar lists the agent's tools (calculator, current datetime, web search, knowledge lookup) and the current session, while the main panel shows the welcome message with clickable example questions.

## Features

- **ReAct-style agent graph** built with LangGraph — model node, tool node, conditional routing
- **Four built-in tools**
  - `calculator` — safe arithmetic evaluation (AST-based, never uses `eval`)
  - `current_datetime` — current date and time in any IANA timezone
  - `web_search` — DuckDuckGo search; free and no API key required
  - `knowledge_lookup` — keyword search over a bundled FAQ knowledge base (`data/faq.txt`)
- **Any OpenAI-compatible LLM** — configured purely through environment variables
- **Three interfaces**: browser chat UI (`http://localhost:8000/`), CLI (`python -m app.cli`), and REST API (`POST /chat`)
- **Web chat UI** — vanilla HTML/CSS/JS (no build step, no CDN): message bubbles, a live tool panel fed by `GET /tools`, the tools used for each reply shown as steps, a persistent session id with a "New chat" reset, and a typing indicator
- **Per-session memory** — conversations are kept separate by `session_id`
- **Graceful no-key behaviour** — tools and the API stay usable for inspection; chat returns a clear error until a key is configured

## How the Graph Works

The agent is a small state graph with two working nodes:

```
            +---------+
            |  START  |
            +----+----+
                 |
                 v
            +---------+      tool calls       +---------+
            |  agent  | --------------------> |  tools  |
            |  (LLM)  |                       | ToolNode|
            |         | <-------------------- |         |
            +----+----+      tool results     +---------+
                 |
                 | final answer (no tool calls)
                 v
            +---------+
            |   END   |
            +---------+
```

- **`agent`** — a chat model with all tools bound to it. It receives the conversation (plus a system prompt) and either answers directly or emits tool calls.
- **`tools`** — LangGraph's prebuilt `ToolNode`, which executes every requested tool and returns the results as tool messages.
- **Routing** — `tools_condition` inspects the model's last message: if it contains tool calls, control goes to `tools`; otherwise the run ends. After `tools`, control always returns to `agent`, so the model can chain several tool calls before answering.
- **State** — `MessagesState` (the message list) plus an in-memory checkpointer (`MemorySaver`). The API's `session_id` is used as the graph `thread_id`, which is what keeps conversations separate and persistent across requests.

## Tech Stack

| Layer        | Technology                                   |
|--------------|----------------------------------------------|
| Agent graph  | LangGraph, LangChain Core                    |
| LLM client   | `langchain-openai` (any OpenAI-compatible API) |
| API server   | FastAPI, Uvicorn, Pydantic v2                |
| Web UI       | Vanilla HTML / CSS / JavaScript (no build step) |
| Web search   | DuckDuckGo via `ddgs` (no API key)           |
| Config       | Environment variables / `.env` (`python-dotenv`) |
| Language     | Python 3.10+                                 |

## Setup

```bash
# 1. Clone and enter the project
git clone <your-fork-url> langgraph-agent
cd langgraph-agent

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure the LLM
cp .env.example .env
# edit .env and set OPENAI_API_KEY (and optionally OPENAI_BASE_URL / LLM_MODEL)
```

### Configuration

| Variable          | Required | Default       | Purpose                                   |
|-------------------|----------|---------------|-------------------------------------------|
| `OPENAI_API_KEY`  | Yes      | —             | API key for the LLM provider              |
| `OPENAI_BASE_URL` | No       | OpenAI default| Base URL of any OpenAI-compatible endpoint |
| `LLM_MODEL`       | No       | `gpt-4o-mini` | Model name                                |

## Usage

### Command line

```bash
python -m app.cli "What is 25 * 17 + sqrt(81)?"
# [tools used: calculator]
# 25 * 17 is 425, and sqrt(81) is 9, so the result is 434.

python -m app.cli --interactive
# Interactive session started. Type 'exit' to quit.
# You: What time is it in Tokyo right now?
```

### API server + web UI

```bash
uvicorn app.server:app --port 8000
```

Open **`http://localhost:8000/`** in a browser for the chat UI: type a question, watch which tools the agent uses for each reply, and start a fresh conversation with **New chat**. The UI talks to the same REST API described below, so it needs the LLM configured (see Setup) for chat; without a key it still loads and shows the graceful error.

Interactive API documentation is available at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "What tools do you have?"}'
```

```json
{
  "session_id": "demo",
  "answer": "I have four tools: a calculator, a date/time lookup, a web search, and a local knowledge base lookup...",
  "tools_used": ["knowledge_lookup"]
}
```

Because the same `session_id` is reused, follow-up questions keep their context:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "And what is 15 percent of 240?"}'
# -> tools_used: ["calculator"]
```

## API Reference

### `GET /`

Serves the browser chat UI (static HTML/CSS/JS from `app/static/`).

### `GET /health`

Returns service status.

```json
{ "status": "ok", "llm_configured": true }
```

### `GET /tools`

Lists the tools bound to the agent, with their descriptions.

### `POST /chat`

| Field        | Type   | Description                                  |
|--------------|--------|----------------------------------------------|
| `session_id` | string | Conversation identifier (history is kept per session) |
| `message`    | string | The user's message                           |

**Response:** `session_id`, `answer` (the agent's reply), and `tools_used` (names of the tools called while producing this answer).

If the LLM is not configured, the endpoint responds with **HTTP 503** and a message explaining which environment variable to set.

## Example Questions

- "What is 15% of 2,340 plus the square root of 12,996?" → `calculator`
- "What time is it in Karachi and in New York right now?" → `current_datetime`
- "How do I add a new tool to this project?" → `knowledge_lookup`
- "What were the top AI headlines this week?" → `web_search`
- "Search for the current LangGraph version, then tell me what day it is in London." → `web_search` + `current_datetime`

## Project Structure

```
langgraph-agent/
├── app/
│   ├── __init__.py
│   ├── graph.py        # LangGraph graph: agent node, tool node, routing, memory
│   ├── tools.py        # Tool implementations (plain functions + @tool wrappers)
│   ├── server.py       # FastAPI app: / (chat UI), /health, /tools, /chat
│   ├── cli.py          # Command-line interface (single question or interactive)
│   └── static/         # Browser chat UI (vanilla HTML/CSS/JS, no build step)
│       ├── index.html
│       ├── style.css
│       └── app.js
├── data/
│   └── faq.txt         # Local knowledge base used by knowledge_lookup
├── tests/
│   └── smoke_test.py   # Key-free smoke tests (tools, graph, API)
├── .env.example
├── requirements.txt
└── README.md
```

## Extending the Agent

**Add a tool:** write a function in `app/tools.py`, decorate it with `@tool`, give it a clear docstring (the model reads it), and add it to the `TOOLS` list. The graph picks it up automatically.

**Extend the knowledge base:** add `Q:` / `A:` entries to `data/faq.txt`, separated by blank lines.

## Testing

Key-free smoke tests cover the standalone tools, graph construction, and the API surface:

```bash
python tests/smoke_test.py
```

## License

Released under the [MIT License](LICENSE).

---
**More projects by Nisar Ahmad** — [GitHub profile](https://github.com/nisarahmad78) · [Portfolio site](https://nisarahmad78.github.io)
- [VOCALIQ — AI Voice Customer Experience Platform](https://github.com/nisarahmad78/VOCALIQ)
- [RAG Document Q&A](https://github.com/nisarahmad78/rag-document-qa)
- [LangGraph AI Agent](https://github.com/nisarahmad78/langgraph-ai-agent)
- [MCP Server Suite](https://github.com/nisarahmad78/mcp-server-suite)
- [AI Support Desk](https://github.com/nisarahmad78/ai-support-desk)
- [LLM Gateway](https://github.com/nisarahmad78/llm-gateway)
