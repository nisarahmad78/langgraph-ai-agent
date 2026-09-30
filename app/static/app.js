// LangGraph AI Agent — chat UI logic (vanilla JS, no dependencies).
//
// Talks to the FastAPI backend:
//   GET  /health  -> { status, llm_configured }
//   GET  /tools   -> { tools: [{ name, description }] }
//   POST /chat    -> { session_id, answer, tools_used }   (body: { session_id, message })

"use strict";

const SESSION_KEY = "langgraph-agent-session";

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const composerEl = document.getElementById("composer");
const toolsListEl = document.getElementById("tools-list");
const sessionIdEl = document.getElementById("session-id");
const newChatBtn = document.getElementById("new-chat-btn");
const statusDotEl = document.getElementById("status-dot");
const statusTextEl = document.getElementById("status-text");
const errorBannerEl = document.getElementById("error-banner");

let sessionId = loadSessionId();
let sending = false;

// ---------------------------------------------------------------------------
// Session handling
// ---------------------------------------------------------------------------

function newSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return "session-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}

function loadSessionId() {
  try {
    const saved = window.localStorage.getItem(SESSION_KEY);
    if (saved) return saved;
  } catch (err) {
    /* localStorage may be unavailable; fall through to a fresh id */
  }
  const fresh = newSessionId();
  saveSessionId(fresh);
  return fresh;
}

function saveSessionId(id) {
  try {
    window.localStorage.setItem(SESSION_KEY, id);
  } catch (err) {
    /* non-fatal: the session simply will not persist across reloads */
  }
}

function renderSessionId() {
  sessionIdEl.textContent = sessionId;
}

// ---------------------------------------------------------------------------
// Message rendering
// ---------------------------------------------------------------------------

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message " + role;

  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : role === "error" ? "Error" : "Agent";
  wrapper.appendChild(label);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);

  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function addSteps(wrapper, toolsUsed) {
  if (!toolsUsed || toolsUsed.length === 0) return;
  const steps = document.createElement("div");
  steps.className = "steps";
  const label = document.createElement("span");
  label.textContent = toolsUsed.length === 1 ? "Tool used:" : "Tools used:";
  steps.appendChild(label);
  toolsUsed.forEach(function (name) {
    const chip = document.createElement("span");
    chip.className = "step-chip";
    chip.textContent = name;
    steps.appendChild(chip);
  });
  wrapper.appendChild(steps);
  scrollToBottom();
}

function showTyping() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";
  wrapper.id = "typing-row";
  const bubble = document.createElement("div");
  bubble.className = "bubble typing";
  bubble.innerHTML = "<span></span><span></span><span></span>";
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
}

function hideTyping() {
  const row = document.getElementById("typing-row");
  if (row) row.remove();
}

function showErrorBanner(text) {
  errorBannerEl.textContent = text;
  errorBannerEl.hidden = false;
}

function hideErrorBanner() {
  errorBannerEl.hidden = true;
  errorBannerEl.textContent = "";
}

function showWelcome() {
  const wrapper = addMessage(
    "assistant",
    "Hi! I am a LangGraph agent with tools for calculations, the current date and time, web search, and a local knowledge base. Ask me something:"
  );
  const examples = document.createElement("div");
  examples.className = "examples";
  [
    "What is 15% of 240 plus sqrt(144)?",
    "What time is it in Karachi right now?",
    "What tools do you have?",
  ].forEach(function (question) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "example-chip";
    chip.textContent = question;
    chip.addEventListener("click", function () {
      inputEl.value = question;
      composerEl.requestSubmit();
    });
    examples.appendChild(chip);
  });
  wrapper.appendChild(examples);
  scrollToBottom();
}

// ---------------------------------------------------------------------------
// Server calls
// ---------------------------------------------------------------------------

async function loadHealth() {
  try {
    const resp = await fetch("/health");
    if (!resp.ok) throw new Error("health check failed");
    const data = await resp.json();
    if (data.llm_configured) {
      statusDotEl.className = "status-dot status-ok";
      statusTextEl.textContent = "Connected — LLM configured";
    } else {
      statusDotEl.className = "status-dot status-warn";
      statusTextEl.textContent = "Server up — no LLM key set (chat will return an error until OPENAI_API_KEY is configured)";
    }
  } catch (err) {
    statusDotEl.className = "status-dot status-down";
    statusTextEl.textContent = "Server unreachable";
  }
}

async function loadTools() {
  try {
    const resp = await fetch("/tools");
    if (!resp.ok) throw new Error("tools request failed");
    const data = await resp.json();
    toolsListEl.innerHTML = "";
    (data.tools || []).forEach(function (tool) {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.className = "tool-name";
      name.textContent = tool.name;
      const desc = document.createElement("span");
      desc.className = "tool-desc";
      // Keep the sidebar compact: show the first sentence of the description.
      desc.textContent = (tool.description || "").split(". ")[0].replace(/\.$/, "") + ".";
      li.appendChild(name);
      li.appendChild(desc);
      toolsListEl.appendChild(li);
    });
    if (!data.tools || data.tools.length === 0) {
      toolsListEl.innerHTML = '<li class="tools-error">No tools reported by the server.</li>';
    }
  } catch (err) {
    toolsListEl.innerHTML = '<li class="tools-error">Could not load the tool list.</li>';
  }
}

async function sendMessage(text) {
  hideErrorBanner();
  addMessage("user", text);
  sending = true;
  sendBtn.disabled = true;
  showTyping();

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });

    let data = null;
    try {
      data = await resp.json();
    } catch (err) {
      /* non-JSON error body; handled below */
    }

    hideTyping();

    if (!resp.ok) {
      const detail =
        (data && data.detail) || "The server returned an error (HTTP " + resp.status + ").";
      addMessage("error", detail);
      return;
    }

    const wrapper = addMessage("assistant", data.answer || "(empty reply)");
    addSteps(wrapper, data.tools_used);
  } catch (err) {
    hideTyping();
    addMessage("error", "Could not reach the server. Is it running? (" + err.message + ")");
  } finally {
    sending = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

// ---------------------------------------------------------------------------
// Composer behaviour
// ---------------------------------------------------------------------------

function autoResize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
}

composerEl.addEventListener("submit", function (event) {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text || sending) return;
  inputEl.value = "";
  autoResize();
  sendMessage(text);
});

inputEl.addEventListener("keydown", function (event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composerEl.requestSubmit();
  }
});

inputEl.addEventListener("input", autoResize);

newChatBtn.addEventListener("click", function () {
  if (sending) return;
  sessionId = newSessionId();
  saveSessionId(sessionId);
  renderSessionId();
  hideErrorBanner();
  messagesEl.innerHTML = "";
  showWelcome();
  inputEl.focus();
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

renderSessionId();
showWelcome();
loadHealth();
loadTools();
inputEl.focus();
