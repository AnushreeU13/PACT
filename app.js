const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const batchFromFileBtn = document.getElementById('batch-from-file-btn');
const attachDocBtn = document.getElementById('attach-doc-btn');
const fileInput = document.getElementById('file-input');
const attachmentPreview = document.getElementById('attachment-preview');
const attachedFilename = document.getElementById('attached-filename');
const removeAttachmentBtn = document.getElementById('remove-attachment');
const apiKeyInput = document.getElementById('api-key-input');
const testKeyBtn = document.getElementById('test-key-btn');
const keyStatus = document.getElementById('key-status');

// Load saved API key from session storage
if (apiKeyInput) {
    const savedKey = sessionStorage.getItem('pact_openai_key');
    if (savedKey) {
        apiKeyInput.value = savedKey;
        updateKeyStatus('active', '');
        console.log("DEBUG: Restored API key from session storage.");
    }
}
const llamaStatusBadge = document.getElementById('llama-status-badge');
const llamaStatusDot = document.getElementById('llama-status-dot');
const llamaStatusText = document.getElementById('llama-status-text');

let currentAttachmentText = "";
let currentAttachmentName = "";

// Toggles
const toggles = {
    identity: document.getElementById('toggle-identity'),
    location: document.getElementById('toggle-location'),
    demographic: document.getElementById('toggle-demographic'),
    health: document.getElementById('toggle-health'),
    financial: document.getElementById('toggle-financial')
};

// Auto-resize textarea
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = (userInput.scrollHeight) + 'px';
});

async function sendMessage() {
    let text = userInput.value.trim();
    if (!text && !currentAttachmentText) return;

    // Combine document and question if attachment exists
    let fullQuery = text;
    if (currentAttachmentText) {
        fullQuery = `[Document Content: ${currentAttachmentName}]\n${currentAttachmentText}\n\n[User Question]: ${text}`;
    }

    // Add user message to UI (show a shorter version if it's a huge doc)
    const displayMessage = currentAttachmentName ? `[Attached: ${currentAttachmentName}]\n${text}` : text;
    appendMessage('user', displayMessage);
    userInput.value = '';
    userInput.style.height = 'auto';

    // Show typing indicator or placeholder
    const botMsgId = appendMessage('bot', 'Sanitizing and thinking...');

    try {
        // Prepare request body
        const keyEl = document.getElementById('api-key-input');
        let userApiKey = keyEl ? keyEl.value.trim() : null;
        
        // Fallback to session storage if the input exists but is empty
        if (!userApiKey) {
            userApiKey = sessionStorage.getItem('pact_openai_key');
        }
        
        const auThresholdEl = document.getElementById('au-threshold-select');
        const auThreshold = auThresholdEl ? parseFloat(auThresholdEl.value) : 0.8;

        const payload = {
            query: fullQuery,
            is_document: !!currentAttachmentText,
            api_key: userApiKey,
            au_threshold: auThreshold,
            settings: {
                identity: toggles.identity.checked,
                location: toggles.location.checked,
                demographic: toggles.demographic.checked,
                health: toggles.health.checked,
                financial: toggles.financial.checked
            }
        };

        console.log("DEBUG: Sending payload to /chat", {
            query_length: payload.query.length,
            has_api_key: !!payload.api_key,
            api_key_length: payload.api_key ? payload.api_key.length : 0
        });

        const response = await fetch(BACKEND_URL + '/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json().catch(() => null);
        if (!response.ok) {
            const detail = data && data.detail ? data.detail : `Request failed with status ${response.status}`;
            updateMessage(botMsgId, escapeHtml(String(detail)), []);
            return;
        }

        updateMessage(
            botMsgId,
            data.response,
            data.sanitizations || [],
            data.pipeline_trace || null
        );

        // Clear attachment after successful send
        clearAttachment();

    } catch (error) {
        console.error('Error:', error);
        updateMessage(
            botMsgId,
            "Sorry, I couldn't connect to the backend sanitizer. Make sure the Python server is running.",
            []
        );
    }
}

let _msgCounter = 0;
function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    const id = 'msg-' + (++_msgCounter);
    msgDiv.id = id;
    msgDiv.className = `message ${role}-message`;
    msgDiv.textContent = text;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function appendPipelineTrace(container, trace) {
    if (!trace || typeof trace !== 'object') return;

    const wrap = document.createElement('details');
    wrap.style.marginTop = '12px';
    wrap.style.border = '1px solid var(--glass-border, rgba(255,255,255,0.12))';
    wrap.style.borderRadius = '8px';
    wrap.style.padding = '8px 10px';
    wrap.style.background = 'rgba(0,0,0,0.15)';

    const sum = document.createElement('summary');
    sum.style.cursor = 'pointer';
    sum.style.fontSize = '0.8rem';
    sum.style.fontWeight = '600';
    sum.style.color = 'var(--accent-cyan, #5eead4)';
    sum.style.display = 'flex';
    sum.style.alignItems = 'center';
    sum.style.gap = '10px';

    // AU Probe badge in summary line
    const auProbe = trace.au_probe;
    let badgeHtml = '';
    if (auProbe) {
        const score = (auProbe.score * 100).toFixed(1);
        const isUncertain = auProbe.triggered;
        const badgeColor = isUncertain ? '#ef4444' : '#22d3ee';
        const badgeBg   = isUncertain ? 'rgba(239,68,68,0.15)' : 'rgba(34,211,238,0.15)';
        badgeHtml = `<span style="
            background:${badgeBg};
            color:${badgeColor};
            border:1px solid ${badgeColor};
            border-radius:5px;
            padding:1px 7px;
            font-size:0.72rem;
            font-weight:700;
            letter-spacing:0.02em;
            white-space:nowrap;
        ">AU Score: ${score}% - ${auProbe.status.toUpperCase()}</span>`;
    }

    sum.innerHTML = `<span>Pipeline: module masks - Groq - GPT</span>${badgeHtml}`;

    const body = document.createElement('div');
    body.style.marginTop = '10px';
    body.style.display = 'flex';
    body.style.flexDirection = 'column';
    body.style.gap = '8px';

    // Helper to create a section block
    const makeSection = (title, content, color = 'rgba(255,255,255,0.06)') => {
        const sec = document.createElement('div');
        sec.style.borderRadius = '6px';
        sec.style.background = color;
        sec.style.padding = '6px 10px';

        const h = document.createElement('div');
        h.style.fontSize = '0.68rem';
        h.style.fontWeight = '700';
        h.style.color = 'var(--accent-cyan, #5eead4)';
        h.style.marginBottom = '4px';
        h.style.textTransform = 'uppercase';
        h.style.letterSpacing = '0.05em';
        h.textContent = title;

        const pre = document.createElement('pre');
        pre.style.margin = '0';
        pre.style.whiteSpace = 'pre-wrap';
        pre.style.wordBreak = 'break-word';
        pre.style.fontSize = '0.66rem';
        pre.style.lineHeight = '1.4';
        pre.style.color = 'rgba(255,255,255,0.8)';
        pre.textContent = typeof content === 'string' ? content : JSON.stringify(content, null, 2);

        sec.appendChild(h);
        sec.appendChild(pre);
        return sec;
    };

    // AU Probe section (prominent, colored)
    if (auProbe) {
        const isUncertain = auProbe.triggered;
        const sectionColor = isUncertain ? 'rgba(239,68,68,0.12)' : 'rgba(34,211,238,0.08)';
        const scoreBar = document.createElement('div');
        scoreBar.style.borderRadius = '6px';
        scoreBar.style.background = sectionColor;
        scoreBar.style.padding = '8px 10px';

        const h = document.createElement('div');
        h.style.fontSize = '0.68rem';
        h.style.fontWeight = '700';
        h.style.color = isUncertain ? '#ef4444' : '#22d3ee';
        h.style.marginBottom = '6px';
        h.style.textTransform = 'uppercase';
        h.style.letterSpacing = '0.05em';
        h.textContent = 'AU-Probe Uncertainty';
        scoreBar.appendChild(h);

        // Score bar
        const barWrap = document.createElement('div');
        barWrap.style.display = 'flex';
        barWrap.style.alignItems = 'center';
        barWrap.style.gap = '10px';

        const barTrack = document.createElement('div');
        barTrack.style.flex = '1';
        barTrack.style.height = '8px';
        barTrack.style.borderRadius = '4px';
        barTrack.style.background = 'rgba(255,255,255,0.1)';
        barTrack.style.position = 'relative';
        barTrack.style.overflow = 'hidden';

        const barFill = document.createElement('div');
        const scorePercent = Math.round(auProbe.score * 100);
        barFill.style.width = `${scorePercent}%`;
        barFill.style.height = '100%';
        barFill.style.borderRadius = '4px';
        barFill.style.background = isUncertain
            ? 'linear-gradient(90deg, #f97316, #ef4444)'
            : 'linear-gradient(90deg, #22d3ee, #6366f1)';
        barFill.style.transition = 'width 0.4s ease';

        // Threshold marker at 80%
        const marker = document.createElement('div');
        marker.style.position = 'absolute';
        marker.style.left = '80%';
        marker.style.top = '0';
        marker.style.bottom = '0';
        marker.style.width = '2px';
        marker.style.background = 'rgba(255,255,255,0.4)';
        marker.title = 'Threshold: 80%';

        barTrack.appendChild(barFill);
        barTrack.appendChild(marker);

        const scoreLabel = document.createElement('span');
        scoreLabel.style.fontSize = '0.75rem';
        scoreLabel.style.fontWeight = '700';
        scoreLabel.style.color = isUncertain ? '#ef4444' : '#22d3ee';
        scoreLabel.style.minWidth = '40px';
        scoreLabel.textContent = `${scorePercent}%`;

        const thresholdLabel = document.createElement('span');
        thresholdLabel.style.fontSize = '0.62rem';
        thresholdLabel.style.color = 'rgba(255,255,255,0.4)';
        thresholdLabel.textContent = `(threshold: ${Math.round(auProbe.threshold * 100)}%)`;

        barWrap.appendChild(barTrack);
        barWrap.appendChild(scoreLabel);
        barWrap.appendChild(thresholdLabel);
        scoreBar.appendChild(barWrap);
        body.appendChild(scoreBar);
    }

    // Final prompt
    if (trace.final_prompt_to_gpt) {
        body.appendChild(makeSection('Final Prompt → GPT', trace.final_prompt_to_gpt));
    }

    // Local Llama synthesis summary
    if (trace.local_llama) {
        const ll = trace.local_llama;
        const summary = `Mode: ${ll.synthesis_mode}\nOutput: ${ll.extracted_before_fallback || '-'}\nFallback: ${ll.used_fallback ? `YES (${ll.fallback_reason})` : 'No'}`;
        body.appendChild(makeSection('Groq Synthesis', summary));
    }

    // Module masks (compact)
    if (trace.module_masks) {
        body.appendChild(makeSection('Module Masks', trace.module_masks));
    }

    // Raw JSON (collapsible)
    const rawDetails = document.createElement('details');
    rawDetails.style.marginTop = '2px';
    const rawSum = document.createElement('summary');
    rawSum.style.fontSize = '0.62rem';
    rawSum.style.color = 'rgba(255,255,255,0.35)';
    rawSum.style.cursor = 'pointer';
    rawSum.textContent = 'Raw JSON';
    const rawPre = document.createElement('pre');
    rawPre.style.margin = '6px 0 0 0';
    rawPre.style.whiteSpace = 'pre-wrap';
    rawPre.style.wordBreak = 'break-word';
    rawPre.style.fontSize = '0.62rem';
    rawPre.style.lineHeight = '1.35';
    rawPre.style.maxHeight = 'min(40vh, 300px)';
    rawPre.style.overflow = 'auto';
    rawPre.style.color = 'rgba(255,255,255,0.5)';
    rawPre.textContent = JSON.stringify(trace, null, 2);
    rawDetails.appendChild(rawSum);
    rawDetails.appendChild(rawPre);
    body.appendChild(rawDetails);

    wrap.appendChild(sum);
    wrap.appendChild(body);
    container.appendChild(wrap);
}

function appendSanitizationsBlock(container, sanitizations) {
    if (!sanitizations || sanitizations.length === 0) return;
    const sanitizationList = document.createElement('div');
    sanitizationList.style.marginTop = '8px';
    sanitizationList.style.padding = '8px';
    sanitizationList.style.fontSize = '0.75rem';
    sanitizationList.style.background = 'rgba(0,0,0,0.1)';
    sanitizationList.style.borderRadius = '8px';
    sanitizationList.style.borderLeft = '2px solid var(--accent-cyan)';

    let html = '<strong>Privacy Actions:</strong><ul style="margin-left: 15px; margin-top: 4px;">';
    sanitizations.forEach((s) => {
        const orig = escapeHtml(s.original);
        html += `<li>Redacted financial info: <span style="color: var(--accent-cyan);">${orig}</span> → [REDACTED]</li>`;
    });
    html += '</ul>';
    sanitizationList.innerHTML = html;
    container.appendChild(sanitizationList);
}

function updateMessage(id, text, sanitizations = [], pipelineTrace = null) {
    const msgDiv = document.getElementById(id);
    if (!msgDiv) return;

    msgDiv.innerHTML = '';
    appendPipelineTrace(msgDiv, pipelineTrace);
    const body = document.createElement('div');
    body.className = 'bot-reply-body';
    body.innerHTML = text;
    msgDiv.appendChild(body);
    appendSanitizationsBlock(msgDiv, sanitizations);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function runBatchFromFile() {
    appendMessage('user', '[Run batch: data/queries.json]');
    const botMsgId = appendMessage('bot', 'Loading queries from JSON and sanitizing…');

    try {
        const keyEl = document.getElementById('api-key-input');
        let userApiKey = keyEl ? keyEl.value.trim() : null;
        if (!userApiKey) {
            userApiKey = sessionStorage.getItem('pact_openai_key');
        }

        const response = await fetch(BACKEND_URL + '/chat/batch-from-file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ queries: [], api_key: userApiKey })
        });
        const data = await response.json().catch(() => null);
        if (!response.ok) {
            const detail = data && data.detail ? data.detail : `Request failed with status ${response.status}`;
            updateMessage(botMsgId, escapeHtml(String(detail)), []);
            return;
        }

        const msgDiv = document.getElementById(botMsgId);
        if (!msgDiv) return;

        msgDiv.innerHTML = '';
        data.results.forEach((r, i) => {
            const block = document.createElement('div');
            block.className = 'batch-result';
            block.style.marginBottom = '1.25rem';
            block.style.paddingBottom = '1rem';
            block.style.borderBottom =
                i < data.results.length - 1 ? '1px solid var(--glass-border)' : 'none';

            const label = document.createElement('div');
            label.style.fontSize = '0.75rem';
            label.style.color = 'var(--text-muted)';
            label.style.marginBottom = '6px';
            label.textContent = `Query ${i + 1} of ${data.count}`;
            block.appendChild(label);

            appendPipelineTrace(block, r.pipeline_trace || null);

            const body = document.createElement('div');
            body.innerHTML = r.response;
            block.appendChild(body);
            appendSanitizationsBlock(block, r.sanitizations);
            msgDiv.appendChild(block);
        });

        chatMessages.scrollTop = chatMessages.scrollHeight;
    } catch (error) {
        console.error('Batch error:', error);
        updateMessage(
            botMsgId,
            'Could not run data/queries.json. Is the backend running and is the file present on the server?',
            []
        );
    }
}

sendBtn.addEventListener('click', sendMessage);
batchFromFileBtn.addEventListener('click', runBatchFromFile);

attachDocBtn.addEventListener('click', () => fileInput.click());

// API Key Helpers
function updateKeyStatus(mode, icon) {
    const keyStatus = document.getElementById('key-status');
    if (!keyStatus) return;
    if (mode === 'active') {
        keyStatus.textContent = icon + ' active';
        keyStatus.style.color = '#22d3ee'; // var(--accent-cyan)
        keyStatus.style.fontWeight = 'bold';
    } else if (mode === 'error') {
        keyStatus.textContent = icon + ' invalid';
        keyStatus.style.color = '#ff6b6b';
        keyStatus.style.fontWeight = 'bold';
    } else {
        keyStatus.textContent = '';
    }
}

if (apiKeyInput) {
    apiKeyInput.addEventListener('input', () => {
        const val = apiKeyInput.value.trim();
        if (val) {
            sessionStorage.setItem('pact_openai_key', val);
            updateKeyStatus('active', '•');
        } else {
            sessionStorage.removeItem('pact_openai_key');
            updateKeyStatus('empty', '');
        }
    });

    apiKeyInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const val = apiKeyInput.value.trim();
            if (val) {
                updateKeyStatus('active', '');
                apiKeyInput.blur();
            }
        }
    });
}

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const originalPlaceholder = userInput.placeholder;
    userInput.placeholder = "Reading attachment...";
    userInput.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(BACKEND_URL + '/extract/text', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) {
            alert(data.detail || "Extraction failed");
            return;
        }

        // Store internally, don't show all text in box
        currentAttachmentText = data.text;
        currentAttachmentName = file.name;
        
        // Show the badge
        attachedFilename.textContent = `Attached: ${file.name}`;
        attachmentPreview.style.display = 'block';
        userInput.placeholder = "Ask a question about this attachment...";
        userInput.focus();
        
    } catch (error) {
        console.error('Extraction Error:', error);
        alert("Failed to extract document text.");
    } finally {
        userInput.disabled = false;
        fileInput.value = '';
    }
});

function clearAttachment() {
    currentAttachmentText = "";
    currentAttachmentName = "";
    attachmentPreview.style.display = 'none';
    userInput.placeholder = "Type your message here...";
}

removeAttachmentBtn.addEventListener('click', clearAttachment);
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Preload local Llama on page load so the user sees progress immediately.
// First-time HF download + 8B load can take hours on a slow link; polling runs up to 24h.
(async function bootLocalLlama() {
    if (!chatMessages) return;

    const POLL_MS = 2500;
    // Match long server wait (default 6h); cap UI polling at 24h so tabs don't run forever.
    const LOAD_DEADLINE_MS = Date.now() + 24 * 60 * 60 * 1000;
    const started = Date.now();

    const msgId = appendMessage('bot', 'Connecting to Groq backend...');

    const setText = (t, status = 'loading') => {
        const el = document.getElementById(msgId);
        if (el) el.textContent = t;

        if (status === 'ready') {
            setHeaderStatus('ready', 'Groq Sanitizer Active');
        } else if (status === 'error') {
            setHeaderStatus('error', 'Groq Unavailable');
        } else {
            setHeaderStatus('loading', 'Connecting to Groq...');
        }
    };

    const setHeaderStatus = (status, text) => {
        if (!llamaStatusBadge) return;
        llamaStatusText.textContent = text;
        llamaStatusBadge.style.opacity = "1";
        if (status === 'ready') {
            llamaStatusDot.style.background = "var(--accent-cyan)";
            llamaStatusBadge.style.color = "var(--accent-cyan)";
        } else if (status === 'loading') {
            llamaStatusDot.style.background = "#fbbf24";
            llamaStatusBadge.style.color = "#fbbf24";
        } else {
            llamaStatusDot.style.background = "#ef4444";
            llamaStatusBadge.style.color = "#ef4444";
        }
    };

    const fmtElapsed = () => {
        const m = Math.floor((Date.now() - started) / 60000);
        const s = Math.floor(((Date.now() - started) % 60000) / 1000);
        return m > 0 ? `${m}m ${s}s` : `${s}s`;
    };

    let consecutiveErrors = 0;
    const MAX_CONSECUTIVE_ERRORS = 20; // ~50s of retries before giving up

    while (Date.now() < LOAD_DEADLINE_MS) {
        try {
            const statusResp = await fetch(BACKEND_URL + '/local-llama/status');
            if (!statusResp.ok) {
                consecutiveErrors++;
                if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                    setText('Groq backend status unavailable. (Check backend URL in config.js)', 'error');
                    return;
                }
                setText(`Groq backend unreachable (${statusResp.status}) - retrying... (${fmtElapsed()})`, 'loading');
                await new Promise((r) => setTimeout(r, POLL_MS));
                continue;
            }
            const st = await statusResp.json().catch(() => ({}));
            consecutiveErrors = 0;

            if (st.loaded) {
                setText('Groq backend: ready.', 'ready');
                return;
            }

            if (st.loading) {
                setText(`Connecting to Groq... (${fmtElapsed()} elapsed)`);
                await new Promise((r) => setTimeout(r, POLL_MS));
                continue;
            }

            if (st.load_error) {
                setText(`Groq backend failed: ${st.load_error}`, 'error');
                return;
            }

            // Not loaded and not currently loading -> kick off loading.
            await fetch(BACKEND_URL + '/local-llama/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            }).catch(() => {});

            setText(`Connecting to Groq... (${fmtElapsed()} elapsed)`);
        } catch (e) {
            consecutiveErrors++;
            console.error(`Local Llama status error (attempt ${consecutiveErrors}):`, e);
            if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                setText('Local Llama: status unavailable. (Check backend.)', 'error');
                return;
            }
            // Backend may still be cold-starting on Railway - keep retrying
            setText(`Local Llama: backend starting up… (${fmtElapsed()} elapsed)`, 'loading');
        }

        await new Promise((r) => setTimeout(r, POLL_MS));
    }

    setText(
        'Local Llama: UI wait limit (24h). Load may still run on the server - hard-refresh (Ctrl+Shift+R) to reload app.js, check GET /local-llama/status, or set LOCAL_LLM_LOAD_TIMEOUT_SEC on the backend.'
    );
})();
