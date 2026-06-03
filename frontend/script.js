document.addEventListener("DOMContentLoaded", () => {
    const API_BASE_URL = window.location.origin;
    const CHAT_STORAGE_KEY = "healthcare_rag_chat_history_v1";
    const REQUEST_TIMEOUT_MS = 90000;

    const inputField = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const chatHistoryContainer = document.getElementById("chat-history");
    const clearChatBtn = document.getElementById("clear-chat-btn");
    const exportChatBtn = document.getElementById("export-chat-btn");
    const historyStatus = document.getElementById("history-status");

    const chunkSizeSelect = document.getElementById("chunk-size");
    const embeddingModelSelect = document.getElementById("embedding-model");
    const llmModelSelect = document.getElementById("llm-model");

    const pipelineSteps = [
        document.getElementById("step-ingestion"),
        document.getElementById("step-chunking"),
        document.getElementById("step-embeddings"),
        document.getElementById("step-vectordb"),
        document.getElementById("step-llm")
    ].filter(Boolean);
    const pipelineLines = [
        document.getElementById("line-1"),
        document.getElementById("line-2"),
        document.getElementById("line-3"),
        document.getElementById("line-4")
    ].filter(Boolean);
    const pipelineStatuses = [
        document.getElementById("status-ingestion"),
        document.getElementById("status-chunking"),
        document.getElementById("status-embeddings"),
        document.getElementById("status-vectordb"),
        document.getElementById("status-llm")
    ].filter(Boolean);

    let chatMessages = [];
    let isProcessing = false;
    let pipelineResetTimer = null;

    function generateId() {
        return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function formatTime(isoTimestamp) {
        const timestamp = new Date(isoTimestamp);
        if (Number.isNaN(timestamp.getTime())) return "Unknown time";
        return timestamp.toLocaleString([], {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit"
        });
    }

    function roleLabel(role) {
        if (role === "user") return "You";
        if (role === "system") return "System";
        return "Assistant";
    }

    function setHistoryStatus(text, isError = false) {
        historyStatus.textContent = text;
        historyStatus.classList.toggle("status-error", isError);
    }

    function safeParseJson(rawText) {
        try {
            return JSON.parse(rawText);
        } catch {
            return null;
        }
    }

    function saveChatHistory() {
        try {
            const payload = {
                version: 1,
                saved_at: new Date().toISOString(),
                messages: chatMessages
            };
            localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(payload));
            setHistoryStatus("Auto-saved locally");
        } catch (error) {
            setHistoryStatus("Save failed (storage unavailable)", true);
            console.error("Unable to save chat history:", error);
        }
    }

    function normalizeMessage(record) {
        const allowedRoles = ["user", "assistant", "system"];
        return {
            id: typeof record.id === "string" ? record.id : generateId(),
            role: allowedRoles.includes(record.role) ? record.role : "assistant",
            text: typeof record.text === "string" ? record.text : "",
            timestamp: typeof record.timestamp === "string" ? record.timestamp : new Date().toISOString(),
            meta: record.meta && typeof record.meta === "object" ? record.meta : null
        };
    }

    function loadChatHistory() {
        const rawHistory = localStorage.getItem(CHAT_STORAGE_KEY);
        if (!rawHistory) return [];

        const parsed = safeParseJson(rawHistory);
        if (!parsed) {
            setHistoryStatus("Corrupted history reset", true);
            localStorage.removeItem(CHAT_STORAGE_KEY);
            return [];
        }

        const records = Array.isArray(parsed) ? parsed : parsed.messages;
        if (!Array.isArray(records)) return [];

        return records.map(normalizeMessage);
    }

    function scrollHistoryToBottom() {
        chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
    }

    function buildMetaDetails(meta) {
        if (!meta) return null;

        const details = document.createElement("details");
        details.className = "rag-details";

        const summary = document.createElement("summary");
        summary.textContent = "RAG Details";
        details.appendChild(summary);

        const list = document.createElement("ul");
        list.className = "rag-list";

        function appendField(label, value) {
            if (value === undefined || value === null || value === "") return;
            const item = document.createElement("li");
            const labelNode = document.createElement("strong");
            labelNode.textContent = `${label}:`;
            item.appendChild(labelNode);
            item.appendChild(document.createTextNode(` ${value}`));
            list.appendChild(item);
        }

        appendField("Model", meta.model_used);
        appendField("Response Time", meta.response_time_ms ? `${meta.response_time_ms} ms` : null);
        appendField("Precision", meta.precision);
        appendField("Relevance", meta.relevance);

        if (meta.config) {
            appendField(
                "Config",
                `chunk=${meta.config.chunk_size}, embedding=${meta.config.embedding}, db=${meta.config.db}, llm=${meta.config.llm}`
            );
        }

        if (Array.isArray(meta.sources) && meta.sources.length > 0) {
            const sourceText = meta.sources
                .map((entry) => `${entry.source} (${entry.score})`)
                .join(", ");
            appendField("Top Sources", sourceText);
        }

        if (list.children.length === 0) return null;

        details.appendChild(list);
        return details;
    }

    function renderMessage(message) {
        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${message.role === "user" ? "bubble-user" : message.role === "system" ? "bubble-system" : "bubble-ai"}`;
        bubble.dataset.messageId = message.id;

        const metaRow = document.createElement("div");
        metaRow.className = "chat-meta-row";

        const role = document.createElement("span");
        role.className = "chat-role";
        role.textContent = roleLabel(message.role);

        const time = document.createElement("span");
        time.className = "chat-time";
        time.textContent = formatTime(message.timestamp);

        metaRow.appendChild(role);
        metaRow.appendChild(time);

        const text = document.createElement("div");
        text.className = "chat-text";
        text.textContent = message.text;

        bubble.appendChild(metaRow);
        bubble.appendChild(text);

        if (message.role !== "system") {
            const deleteBtn = document.createElement("button");
            deleteBtn.className = "delete-msg-btn";
            deleteBtn.type = "button";
            deleteBtn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
            deleteBtn.addEventListener("click", () => {
                deleteMessage(message.id);
            });
            bubble.appendChild(deleteBtn);
        }

        if (message.role === "assistant") {
            const details = buildMetaDetails(message.meta);
            if (details) bubble.appendChild(details);
        }

        chatHistoryContainer.appendChild(bubble);
    }

    function renderChatHistory() {
        chatHistoryContainer.innerHTML = "";

        if (chatMessages.length === 0) {
            const emptyState = document.createElement("div");
            emptyState.className = "chat-empty-state";
            emptyState.textContent = "No chat history yet. Ask your first healthcare question.";
            chatHistoryContainer.appendChild(emptyState);
            return;
        }

        chatMessages.forEach((message) => renderMessage(message));
        scrollHistoryToBottom();
    }

    function makeMessage(role, text, meta = null) {
        return {
            id: generateId(),
            role,
            text,
            timestamp: new Date().toISOString(),
            meta
        };
    }

    function appendMessage(message, shouldPersist = true) {
        chatMessages.push(message);
        renderChatHistory();
        if (shouldPersist) saveChatHistory();
    }

    function replaceMessageById(messageId, replacement) {
        const index = chatMessages.findIndex((entry) => entry.id === messageId);
        if (index === -1) return;
        replacement.id = messageId;
        chatMessages[index] = replacement;
        renderChatHistory();
        saveChatHistory();
    }

    function deleteMessage(messageId) {
        chatMessages = chatMessages.filter((entry) => entry.id !== messageId);
        renderChatHistory();
        saveChatHistory();
    }

    function getSelectedConfig() {
        return {
            chunk_size: parseInt(chunkSizeSelect.value, 10),
            embedding: embeddingModelSelect.value,
            db: document.querySelector('input[name="db-type"]:checked').value,
            llm: llmModelSelect.value
        };
    }

    function resetPipeline() {
        pipelineSteps.forEach((step) => step.classList.remove("active"));
        pipelineLines.forEach((line) => line.classList.remove("active"));
        pipelineStatuses.forEach((status) => status.classList.add("hidden"));
    }

    function schedulePipelineReset(delayMs = 2400) {
        if (pipelineResetTimer) clearTimeout(pipelineResetTimer);
        pipelineResetTimer = setTimeout(() => {
            resetPipeline();
            pipelineResetTimer = null;
        }, delayMs);
    }

    function animatePipelineStep(index, durationMs = 240) {
        return new Promise((resolve) => {
            const step = pipelineSteps[index];
            if (step) step.classList.add("active");

            const status = pipelineStatuses[index];
            if (status) status.classList.remove("hidden");

            const line = index > 0 ? pipelineLines[index - 1] : null;
            if (line) line.classList.add("active");

            setTimeout(resolve, durationMs);
        });
    }

    async function runPipelineAnimation(fetchPromise) {
        if (pipelineSteps.length === 0) return fetchPromise;

        await animatePipelineStep(0, 220);
        await animatePipelineStep(1, 220);
        await animatePipelineStep(2, 220);
        await animatePipelineStep(3, 220);

        const response = await fetchPromise;
        await animatePipelineStep(4, 520);
        return response;
    }

    function setProcessingState(processing) {
        isProcessing = processing;
        sendBtn.disabled = processing;
        inputField.disabled = processing;
    }

    function buildAssistantText(isError, responseStatus, responseData, payload) {
        let answer = payload && typeof payload.answer === "string" ? payload.answer : "";
        if (!answer) answer = "No answer returned by backend.";

        if (payload && payload.system_note) {
            answer = `${payload.system_note}\n\n${answer}`;
        }

        if (isError) {
            const errorLabel = responseData && responseData.error ? responseData.error : "Unknown error";
            answer = `API Error (status ${responseStatus}): ${errorLabel}\n\n${answer}`;
        }

        if (!isError && payload && payload.metrics && payload.metrics.precision < 0.2) {
            answer = `Warning: Retrieval precision is low. Validate this answer manually.\n\n${answer}`;
        }

        return answer;
    }

    function buildAssistantMeta(config, payload) {
        if (!payload || typeof payload !== "object") {
            return { config };
        }

        const sourceEntries = Array.isArray(payload.retrieved_chunks)
            ? payload.retrieved_chunks.slice(0, 5).map((entry) => ({
                  source: entry.source || "Unknown source",
                  score: Number.isFinite(entry.score) ? entry.score.toFixed(3) : "n/a"
              }))
            : [];

        return {
            model_used: payload.model_used || null,
            response_time_ms: payload.response_time_ms || null,
            precision: payload.metrics ? payload.metrics.precision : null,
            relevance: payload.metrics ? payload.metrics.relevance : null,
            config,
            sources: sourceEntries
        };
    }

    async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
        try {
            return await fetch(url, { ...options, signal: controller.signal });
        } finally {
            clearTimeout(timeoutId);
        }
    }

    async function verifyBackendHealth(showInlineError = true) {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            if (!response.ok) throw new Error(`Health status: ${response.status}`);
            return true;
        } catch (error) {
            if (showInlineError) {
                appendMessage(
                    makeMessage(
                        "system",
                        "Backend reachable nahi hai. `python api.py` run karke port 8000 active karein."
                    )
                );
            }
            console.error("Health check failed:", error);
            return false;
        }
    }

    async function processQuery() {
        const query = inputField.value.trim();
        if (!query || isProcessing) return;

        if (pipelineResetTimer) {
            clearTimeout(pipelineResetTimer);
            pipelineResetTimer = null;
        }
        resetPipeline();

        const config = getSelectedConfig();
        setProcessingState(true);

        appendMessage(makeMessage("user", query));
        inputField.value = "";

        const pendingMessage = makeMessage("assistant", "Processing your query...");
        appendMessage(pendingMessage, false);

        try {
            const fetchPromise = fetchWithTimeout(`${API_BASE_URL}/ask`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query, config })
            });
            const response = await runPipelineAnimation(fetchPromise);

            let responseData = null;
            try {
                responseData = await response.json();
            } catch {
                responseData = { error: "Invalid JSON response from backend." };
            }

            const isError = !response.ok;
            const payload = isError && responseData.fallback ? responseData.fallback : responseData;

            const assistantText = buildAssistantText(isError, response.status, responseData, payload);
            const assistantMeta = buildAssistantMeta(config, payload);

            replaceMessageById(
                pendingMessage.id,
                makeMessage("assistant", assistantText, assistantMeta)
            );
        } catch (error) {
            const errorMessage = error.name === "AbortError"
                ? `Request timeout after ${Math.round(REQUEST_TIMEOUT_MS / 1000)}s. Try Groq model for faster response.`
                : error.message;
            replaceMessageById(
                pendingMessage.id,
                makeMessage(
                    "assistant",
                    `Backend request failed.\n${errorMessage}`,
                    { config }
                )
            );
            console.error("Query processing failed:", error);
        } finally {
            schedulePipelineReset(2400);
            setProcessingState(false);
            inputField.focus();
        }
    }

    function clearChatHistory() {
        chatMessages = [];
        localStorage.removeItem(CHAT_STORAGE_KEY);
        renderChatHistory();
        setHistoryStatus("History cleared");
    }

    function exportChatHistory() {
        if (chatMessages.length === 0) {
            setHistoryStatus("No messages to export", true);
            return;
        }

        const exportPayload = {
            exported_at: new Date().toISOString(),
            total_messages: chatMessages.length,
            messages: chatMessages
        };

        const blob = new Blob([JSON.stringify(exportPayload, null, 2)], {
            type: "application/json"
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        link.href = url;
        link.download = `healthcare-rag-chat-${timestamp}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        setHistoryStatus("History exported");
    }

    sendBtn.addEventListener("click", processQuery);
    inputField.addEventListener("keydown", (event) => {
        if (event.key === "Enter") processQuery();
    });

    clearChatBtn.addEventListener("click", () => {
        if (chatMessages.length === 0) return;
        const shouldClear = window.confirm("Clear complete chat history from local storage?");
        if (shouldClear) clearChatHistory();
    });

    exportChatBtn.addEventListener("click", exportChatHistory);

    chatMessages = loadChatHistory();
    renderChatHistory();
    resetPipeline();
    verifyBackendHealth(chatMessages.length === 0);
    inputField.focus();
});
