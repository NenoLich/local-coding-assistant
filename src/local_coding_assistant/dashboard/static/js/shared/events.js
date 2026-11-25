// Shared event preview/filter utilities for LOCCA dashboard

export const EVENT_PREVIEW_KEYS = {
    llm_complete: [
        "model",
        "provider",
        "finish_reason",
        "input_tokens",
        "output_tokens",
    ],
    tool_start: [
        "tool_call_id",
        "tool_name",
    ],
    tool_result: [
        "tool_call_id",
        "tool_name",
        "success",
    ],
    turn_complete: [
        "status",
        "final_answer",
        "total_input_tokens",
        "total_output_tokens",
        "total_frames",
        "finish_reason",
    ],
    frame_complete: [
        "status",
        "finish_reason",
        "tool_call_count",
    ],
    turn_start: ["user_input", "turn_index", "agent_mode", "model_name"],
    frame_start: ["iteration"],
    session_start: ["user_input", "agent_mode", "model_name"],
    error: ["error_type", "error_message", "recoverable"],
    continuation_decision: ["decision", "reason"],
};

export const EVENT_COLORS = {
    session_start: "primary",
    turn_start: "primary",
    frame_start: "info",
    llm_start: "primary",
    llm_complete: "success",
    tool_start: "warning",
    tool_result: "warning",
    approval_required: "warning",
    approval_granted: "success",
    approval_denied: "danger",
    error: "danger",
    continuation_decision: "info",
    frame_complete: "success",
    turn_complete: "success",
    session_resume: "info",
    session_interrupt: "warning",
};

export const EVENT_ICONS = {
    session_start: "bi-play-circle",
    turn_start: "bi-play-circle",
    frame_start: "bi-box-arrow-in-right",
    llm_start: "bi-cpu",
    llm_complete: "bi-check-circle",
    tool_start: "bi-tools",
    tool_result: "bi-tools",
    approval_required: "bi-question-circle",
    approval_granted: "bi-check2-circle",
    approval_denied: "bi-x-circle",
    error: "bi-exclamation-triangle",
    continuation_decision: "bi-arrow-repeat",
    frame_complete: "bi-check2-square",
    turn_complete: "bi-check-all",
    session_resume: "bi-arrow-clockwise",
    session_interrupt: "bi-stop-circle",
};

export function getEventColor(type) {
    return EVENT_COLORS[type] || "secondary";
}

export function getEventIcon(type) {
    return EVENT_ICONS[type] || "bi-circle";
}

export function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

export function formatEventPreview(eventType, payload) {
    if (!payload || typeof payload !== "object") return {};
    const preview = {};
    const keys = EVENT_PREVIEW_KEYS[eventType] || [];
    for (const key of keys) {
        if (key in payload) {
            preview[key] = payload[key];
        }
    }

    if (eventType === "llm_complete") {
        if (payload.content) preview.content = String(payload.content).slice(0, 200);
    } else if (eventType === "approval_required") {
        preview.approval_requests = (payload.approval_requests || []).length;
    } else if (eventType === "approval_granted") {
        preview.approval_token = payload.approval_token;
        preview.tool_call_id = payload.tool_call_id;
    } else if (eventType === "approval_denied") {
        preview.approval_token = payload.approval_token;
        preview.tool_call_id = payload.tool_call_id;
        preview.reason = payload.reason;
    }

    return preview;
}

export function renderPreviewHtml(preview) {
    const parts = [];
    for (const [key, value] of Object.entries(preview)) {
        let displayValue = value;
        if (typeof value === "object") {
            displayValue = JSON.stringify(value);
        }
        parts.push(
            `<strong>${escapeHtml(key.replace(/_/g, " "))}:</strong> ${escapeHtml(
                String(displayValue)
            )}`
        );
    }
    return parts.join("<br>");
}

export function renderTimelineHtml(events, toggleJsonCb) {
    if (!events || events.length === 0) {
        return '<div class="text-center text-muted">No events</div>';
    }
    return events
        .map((event, idx) => {
            const previewHtml =
                Object.keys(event.preview || {}).length > 0
                    ? `<div class="event-payload-preview">${renderPreviewHtml(
                          event.preview
                      )}</div>`
                    : "";
            const eventKey = event.event_id ?? event.seq ?? idx;
            const expandHtml = event.has_payload
                ? `<div class="expandable-json" data-idx="${idx}" data-event-key="${eventKey}" onclick="toggleEventJson(this)">Show full JSON</div><div id="evt-json-${eventKey}" data-event-key="${eventKey}" class="event-json" style="display: none;">${escapeHtml(JSON.stringify(event.payload || {}, null, 2))}</div>`
                : "";
            return `
            <div class="timeline-item d-flex">
                <div class="timeline-marker me-3">
                    <div class="rounded-circle bg-${getEventColor(
                        event.type
                    )} text-white d-flex align-items-center justify-content-center" style="width: 36px; height: 36px;">
                        <i class="bi ${getEventIcon(event.type)}"></i>
                    </div>
                </div>
                <div class="timeline-content flex-grow-1">
                    <div class="card mb-2">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h6 class="mb-0"><span class="event-type-badge">${escapeHtml(
                                event.type
                            )}</span></h6>
                            <small class="text-muted">Seq: ${event.seq} | ${
                                event.timestamp
                                    ? new Date(event.timestamp).toLocaleTimeString()
                                    : "-"
                            }</small>
                        </div>
                        <div class="card-body">
                            ${previewHtml}
                            ${expandHtml}
                        </div>
                    </div>
                </div>
            </div>
        `;
        })
        .join("");
}

export function getSelectedEventTypes() {
    const allCheckbox = document.querySelector('.event-type-checkbox[value=""]');
    const specificCheckboxes = document.querySelectorAll(
        '.event-type-checkbox:not([value=""]):checked'
    );
    const types = [];
    specificCheckboxes.forEach((cb) => {
        types.push(cb.value);
    });
    if (allCheckbox && allCheckbox.checked) {
        return [];
    }
    return types;
}

export function setupEventFilterHandlers(applyHandler, clearHandler) {
    const applyBtn = document.getElementById("event-apply-filters");
    const clearBtn = document.getElementById("event-clear-filters");
    if (applyBtn) {
        applyBtn.removeEventListener("click", applyHandler);
        applyBtn.addEventListener("click", applyHandler);
    }
    if (clearBtn) {
        clearBtn.removeEventListener("click", clearHandler);
        clearBtn.addEventListener("click", clearHandler);
    }

    // "All" checkbox toggles all specific types
    const allCheckbox = document.querySelector('.event-type-checkbox[value=""]');
    if (allCheckbox) {
        allCheckbox.removeEventListener("change", handleAllCheckboxChange);
        allCheckbox.addEventListener("change", handleAllCheckboxChange);
    }

    // Unchecking any specific type unchecks "All"
    document
        .querySelectorAll('.event-type-checkbox:not([value=""])')
        .forEach((cb) => {
            cb.removeEventListener("change", handleSpecificCheckboxChange);
            cb.addEventListener("change", handleSpecificCheckboxChange);
        });
}

function handleAllCheckboxChange() {
    const specificCheckboxes = document.querySelectorAll(
        '.event-type-checkbox:not([value=""])'
    );
    specificCheckboxes.forEach((scb) => {
        scb.checked = this.checked;
    });
}

function handleSpecificCheckboxChange() {
    if (!this.checked) {
        const allCheckbox = document.querySelector('.event-type-checkbox[value=""]');
        if (allCheckbox) allCheckbox.checked = false;
    }
}

export function toggleEventJson(elem) {
    const key = elem.getAttribute("data-event-key") ?? elem.getAttribute("data-idx");
    const id = "evt-json-" + key;
    const el = document.getElementById(id);
    if (el.style.display === "none") {
        el.style.display = "block";
        elem.textContent = "Hide JSON";
    } else {
        el.style.display = "none";
        elem.textContent = "Show full JSON";
    }
}

export function truncateText(text, maxLen = 150) {
    if (!text) return "";
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen) + "...";
}

export function toggleText(btn) {
    const container = btn.closest(".truncate-container");
    if (!container) return;
    const fullEl = container.querySelector(".text-full");
    const shortEl = container.querySelector(".text-short");
    if (!fullEl || !shortEl) return;
    if (btn.textContent === "Show more") {
        shortEl.style.display = "none";
        fullEl.style.display = "block";
        btn.textContent = "Show less";
    } else {
        shortEl.style.display = "block";
        fullEl.style.display = "none";
        btn.textContent = "Show more";
    }
}

// Make toggleText available globally for inline onclick handlers
if (typeof window !== "undefined") {
    window.toggleText = toggleText;
    window.toggleJson = toggleJson;
    window.toggleToolData = toggleToolData;
    window.toggleEventJson = toggleEventJson;
}

export function renderTruncatedText(text, maxLen = 150, expandable = true) {
    if (!text) return '<span class="text-muted">(empty)</span>';
    if (!expandable) return escapeHtml(truncateText(text, maxLen));
    if (text.length <= maxLen) return escapeHtml(text);
    return (
        '<div class="truncate-container">' +
        '<span class="text-short">' +
        escapeHtml(truncateText(text, maxLen)) +
        '</span>' +
        '<span class="text-full" style="display: none;">' +
        escapeHtml(text) +
        '</span>' +
        '<div class="expandable-text" onclick="toggleText(this)">Show more</div>' +
        '</div>'
    );
}

export function toggleJson(elem, id) {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.style.display === "none") {
        el.style.display = "block";
        elem.textContent = "Hide metadata";
    } else {
        el.style.display = "none";
        elem.textContent = "Show metadata";
    }
}

export function toggleToolData(elem, id) {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.style.display === "none") {
        el.style.display = "block";
        elem.textContent = "Hide data";
    } else {
        el.style.display = "none";
        elem.textContent = "Show data";
    }
}
