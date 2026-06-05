const defaults = {
  style: "reasoning",
  useHistory: false,
  webResearch: false,
  researchDepth: "quick",
  systemPrompt: "",
  temperature: 0,
  topK: 0,
  topP: 1,
  repetitionPenalty: 1,
  maxTokens: 256,
};

const elements = {
  composer: document.querySelector("#composer"),
  messageInput: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
  messageList: document.querySelector("#messageList"),
  emptyState: document.querySelector("#emptyState"),
  conversation: document.querySelector("#conversation"),
  newChatButton: document.querySelector("#newChatButton"),
  resetSettingsButton: document.querySelector("#resetSettingsButton"),
  styleInput: document.querySelector("#styleInput"),
  historyInput: document.querySelector("#historyInput"),
  webResearchInput: document.querySelector("#webResearchInput"),
  researchDepthInput: document.querySelector("#researchDepthInput"),
  systemPromptInput: document.querySelector("#systemPromptInput"),
  temperatureInput: document.querySelector("#temperatureInput"),
  temperatureOutput: document.querySelector("#temperatureOutput"),
  topKInput: document.querySelector("#topKInput"),
  topPInput: document.querySelector("#topPInput"),
  topPOutput: document.querySelector("#topPOutput"),
  repetitionInput: document.querySelector("#repetitionInput"),
  repetitionOutput: document.querySelector("#repetitionOutput"),
  maxTokensInput: document.querySelector("#maxTokensInput"),
  modelLabel: document.querySelector("#modelLabel"),
  deviceLabel: document.querySelector("#deviceLabel"),
  settingsPanel: document.querySelector("#settingsPanel"),
  mobileSettingsButton: document.querySelector("#mobileSettingsButton"),
  panelBackdrop: document.querySelector("#panelBackdrop"),
};

let settings = { ...defaults, ...loadJson("hrm-settings", defaults) };
let messages = loadJson("hrm-messages", []);
let generating = false;

function loadJson(key, fallback) {
  try {
    const value = JSON.parse(localStorage.getItem(key));
    return value ?? structuredClone(fallback);
  } catch {
    return structuredClone(fallback);
  }
}

function saveState() {
  localStorage.setItem("hrm-settings", JSON.stringify(settings));
  localStorage.setItem("hrm-messages", JSON.stringify(messages.slice(-50)));
}

function applySettings() {
  elements.styleInput.value = settings.style;
  elements.historyInput.checked = settings.useHistory;
  elements.webResearchInput.checked = settings.webResearch;
  elements.researchDepthInput.value = settings.researchDepth;
  elements.researchDepthInput.disabled = !settings.webResearch;
  elements.systemPromptInput.value = settings.systemPrompt;
  elements.temperatureInput.value = settings.temperature;
  elements.topKInput.value = settings.topK;
  elements.topPInput.value = settings.topP;
  elements.repetitionInput.value = settings.repetitionPenalty;
  elements.maxTokensInput.value = settings.maxTokens;
  elements.temperatureOutput.value = Number(settings.temperature).toFixed(2);
  elements.topPOutput.value = Number(settings.topP).toFixed(2);
  elements.repetitionOutput.value = Number(settings.repetitionPenalty).toFixed(2);
}

function readSettings() {
  settings = {
    style: elements.styleInput.value,
    useHistory: elements.historyInput.checked,
    webResearch: elements.webResearchInput.checked,
    researchDepth: elements.researchDepthInput.value,
    systemPrompt: elements.systemPromptInput.value,
    temperature: Number(elements.temperatureInput.value),
    topK: Math.max(0, Number(elements.topKInput.value) || 0),
    topP: Number(elements.topPInput.value),
    repetitionPenalty: Number(elements.repetitionInput.value),
    maxTokens: Math.max(1, Number(elements.maxTokensInput.value) || 256),
  };
  applySettings();
  saveState();
}

function createMessage(message, index) {
  const article = document.createElement("article");
  article.className = `message ${message.role}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = message.role === "assistant" ? "H" : "YOU";

  const card = document.createElement("div");
  card.className = "message-card";

  const header = document.createElement("div");
  header.className = "message-header";
  const author = document.createElement("strong");
  author.textContent = message.role === "assistant" ? "HRM-Text" : "You";
  header.append(author);

  if (message.role === "assistant" && message.content) {
    const copy = document.createElement("button");
    copy.type = "button";
    copy.textContent = "Copy";
    copy.addEventListener("click", async () => {
      await navigator.clipboard.writeText(message.content);
      copy.textContent = "Copied";
      setTimeout(() => { copy.textContent = "Copy"; }, 1200);
    });
    header.append(copy);
  }

  const content = document.createElement("pre");
  content.className = "message-content";
  if (message.pending && !message.content) {
    const status = document.createElement("span");
    status.className = "research-status";
    status.textContent = message.status || "Generating response...";
    const thinking = document.createElement("span");
    thinking.className = "thinking";
    thinking.append(document.createElement("i"), document.createElement("i"), document.createElement("i"));
    content.append(status, thinking);
  } else {
    content.textContent = message.content;
  }

  card.append(header, content);
  if (message.role === "assistant" && Array.isArray(message.sources) && message.sources.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "source-list";

    const sourceHeading = document.createElement("div");
    sourceHeading.className = "source-heading";
    sourceHeading.textContent = `${message.sources.length} web source${message.sources.length === 1 ? "" : "s"}`;
    sourceList.append(sourceHeading);

    for (const source of message.sources) {
      const safeUrl = safeHttpUrl(source.url);
      if (!safeUrl) continue;

      const link = document.createElement("a");
      link.className = "source-card";
      link.href = safeUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";

      const indexLabel = document.createElement("span");
      indexLabel.className = "source-index";
      indexLabel.textContent = `[${source.id}]`;

      const sourceBody = document.createElement("span");
      sourceBody.className = "source-body";
      const title = document.createElement("strong");
      title.textContent = source.title || safeUrl;
      const domain = document.createElement("small");
      domain.textContent = new URL(safeUrl).hostname;
      const excerpt = document.createElement("span");
      excerpt.textContent = source.excerpt || "";
      sourceBody.append(title, domain, excerpt);

      link.append(indexLabel, sourceBody);
      sourceList.append(link);
    }
    card.append(sourceList);
  }
  article.append(avatar, card);
  article.dataset.index = index;
  return article;
}

function safeHttpUrl(rawUrl) {
  try {
    const url = new URL(rawUrl);
    return ["http:", "https:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function renderMessages() {
  elements.messageList.replaceChildren(...messages.map(createMessage));
  elements.emptyState.hidden = messages.length > 0;
  elements.messageList.hidden = messages.length === 0;
  requestAnimationFrame(scrollToBottom);
}

function scrollToBottom() {
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

function autoSizeInput() {
  elements.messageInput.style.height = "auto";
  elements.messageInput.style.height = `${Math.min(elements.messageInput.scrollHeight, 180)}px`;
}

function setGenerating(value) {
  generating = value;
  elements.sendButton.disabled = value;
  elements.messageInput.disabled = value;
}

async function sendMessage(rawMessage) {
  const message = rawMessage.trim();
  if (!message || generating) return;

  const history = messages
    .filter((entry) => !entry.pending && entry.content)
    .slice(-8)
    .map(({ role, content }) => ({ role, content }));

  messages.push({ role: "user", content: message });
  messages.push({
    role: "assistant",
    content: "",
    pending: true,
    status: settings.webResearch ? "Searching the web..." : "Generating response...",
  });
  elements.messageInput.value = "";
  autoSizeInput();
  setGenerating(true);
  renderMessages();
  saveState();

  const assistantIndex = messages.length - 1;
  try {
    let researchContext = "";
    if (settings.webResearch) {
      const researchResponse = await fetch("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: message,
          depth: settings.researchDepth,
        }),
      });
      const research = await researchResponse.json().catch(() => ({}));
      if (!researchResponse.ok) {
        throw new Error(research.error || `Web research failed with status ${researchResponse.status}`);
      }
      researchContext = research.context || "";
      messages[assistantIndex].sources = Array.isArray(research.sources) ? research.sources : [];
      messages[assistantIndex].status = `HRM is synthesizing ${messages[assistantIndex].sources.length} sources...`;
      renderMessages();
      saveState();
    }

    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history,
        researchContext,
        ...settings,
      }),
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || `Request failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      messages[assistantIndex].pending = false;
      messages[assistantIndex].status = "";
      messages[assistantIndex].content += decoder.decode(value, { stream: true });
      renderMessages();
    }
    messages[assistantIndex].pending = false;
    messages[assistantIndex].status = "";
    messages[assistantIndex].content += decoder.decode();
    if (!messages[assistantIndex].content.trim()) {
      messages[assistantIndex].content = "The model ended the response without producing visible text.";
    }
  } catch (error) {
    messages[assistantIndex].pending = false;
    messages[assistantIndex].content = `Request failed: ${error.message}`;
  } finally {
    setGenerating(false);
    saveState();
    renderMessages();
    elements.messageInput.focus();
  }
}

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(elements.messageInput.value);
});

elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage(elements.messageInput.value);
  }
});

elements.messageInput.addEventListener("input", autoSizeInput);

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

[
  elements.styleInput,
  elements.historyInput,
  elements.webResearchInput,
  elements.researchDepthInput,
  elements.systemPromptInput,
  elements.temperatureInput,
  elements.topKInput,
  elements.topPInput,
  elements.repetitionInput,
  elements.maxTokensInput,
].forEach((input) => input.addEventListener("input", readSettings));

elements.newChatButton.addEventListener("click", () => {
  if (generating) return;
  messages = [];
  saveState();
  renderMessages();
  elements.messageInput.focus();
});

elements.resetSettingsButton.addEventListener("click", () => {
  settings = structuredClone(defaults);
  applySettings();
  saveState();
});

function togglePanel(open) {
  elements.settingsPanel.classList.toggle("open", open);
  elements.panelBackdrop.classList.toggle("open", open);
}

elements.mobileSettingsButton.addEventListener("click", () => togglePanel(true));
elements.panelBackdrop.addEventListener("click", () => togglePanel(false));

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    elements.modelLabel.textContent = status.model;
    elements.deviceLabel.textContent = `${status.device} / ${status.contextLength} token context`;
  } catch {
    elements.deviceLabel.textContent = "Local CUDA runtime unavailable";
  }
}

applySettings();
renderMessages();
autoSizeInput();
loadStatus();
