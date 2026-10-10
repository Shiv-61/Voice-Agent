/**
 * University Voice Agent & Document Intelligence Web Application
 * Complete Interactive Suite: Hands-Free Auto-VAD, Text/Voice Dual Input,
 * Live Tool Visualizer, Telephony Simulator, PDF Chunk Inspector, and Student CRUD.
 */

class VoiceAgentApp {
  constructor() {
    this.ws = null;
    this.audioContext = null;
    this.mediaStream = null;
    this.audioInputNode = null;
    this.analyserNode = null;
    this.processorNode = null;

    this.audioChunks = [];
    this.isRecording = false;
    this.isCallActive = false;
    this.callMode = "auto_vad"; // 'auto_vad' | 'ptt'
    this.callStartTime = null;
    this.callTimerInterval = null;

    // Auto-VAD threshold & adaptive noise tracking state
    this.noiseFloor = 0.012;
    this.minSpeechThreshold = 0.024;
    this.vadThreshold = 0.024;
    this.consecutiveSpeechFrames = 0;
    this.requiredSpeechFrames = 2; // ~170ms of sustained speech energy
    this.consecutiveBargeInFrames = 0;
    this.requiredBargeInFrames = 3; // ~250ms of sustained loud speech to barge in
    this.vadSilenceTimer = null;
    this.vadSilenceDuration = 750; // 750ms natural turn completion pause
    this.isSpeechDetected = false;
    this.callStartupGrace = 0;
    this.playbackStartTime = 0;

    // Audio Playback Queue with Gapless Scheduling
    this.playbackQueue = [];
    this.isPlayingAudio = false;
    this.activeAudioSources = [];
    this.nextAudioStartTime = 0;
    this.workletSupported = false;

    // Canvas animation
    this.canvas = document.getElementById("waveformCanvas");
    this.canvasCtx = this.canvas ? this.canvas.getContext("2d") : null;
    this.animationFrameId = null;

    window.__appInstance = this;
    this.initElements();
    this.initEventListeners();
    this.initWebSocket();
    this.loadDocuments();
    this.loadDatabaseData();
    this.checkSystemStatus();
    this.loadCallHistory();
    this.loadRecentTranscripts();
    this.initUxEnhancements();

    this.startCanvasAnimation();
    // Fix #24: Smart polling — pause during active call, resume when idle
    this._statusPollInterval = setInterval(() => {
      if (!this.isCallActive) this.checkSystemStatus();
    }, 30000); // 30s when idle (was 10s unconditionally)
    this._historyPollInterval = setInterval(() => {
      if (!this.isCallActive) {
        this.loadCallHistory();
        this.loadRecentTranscripts();
      }
    }, 20000); // 20s when idle (paused during call)
  }

  initElements() {
    // Navigation
    this.tabButtons = document.querySelectorAll(".tab-btn");
    this.tabPanes = document.querySelectorAll(".tab-pane");

    // Controls
    this.startCallBtn = document.getElementById("startCallBtn");
    this.pushToTalkBtn = document.getElementById("pushToTalkBtn");
    this.interruptBtn = document.getElementById("interruptBtn");
    this.modeAutoVadBtn = document.getElementById("modeAutoVadBtn");
    this.modePttBtn = document.getElementById("modePttBtn");
    this.textQueryInput = document.getElementById("textQueryInput");
    this.sendTextQueryBtn = document.getElementById("sendTextQueryBtn");

    this.micOrb = document.getElementById("micOrb");
    this.agentStateText = document.getElementById("agentStateText");
    this.languageSelect = document.getElementById("languageSelect");
    this.themeToggle = document.getElementById("themeToggle");
    this.wsStatusBadge = document.getElementById("wsStatusBadge");
    this.callDuration = document.getElementById("callDuration");

    // Live Conversation Feed
    this.conversationFeed = document.getElementById("conversationFeed");
    this.clearConversationBtn = document.getElementById("clearConversationBtn");
    this.currentAssistantBubble = null;
    this.currentAssistantTextSpan = null;
    this.currentThinkingIndicator = null;

    // Call History
    this.callHistoryBody = document.getElementById("callHistoryBody");
    this.refreshCallsBtn = document.getElementById("refreshCallsBtn");

    // RAG Elements
    this.pdfDropzone = document.getElementById("pdfDropzone");
    this.pdfFileInput = document.getElementById("pdfFileInput");
    this.uploadProgressContainer = document.getElementById(
      "uploadProgressContainer",
    );
    this.uploadProgressBar = document.getElementById("uploadProgressBar");
    this.uploadProgressText = document.getElementById("uploadProgressText");
    this.uploadFeedback = document.getElementById("uploadFeedback");
    this.documentsTableBody = document.getElementById("documentsTableBody");
    this.refreshDocsBtn = document.getElementById("refreshDocsBtn");
    this.ragSearchInput = document.getElementById("ragSearchInput");
    this.ragSearchBtn = document.getElementById("ragSearchBtn");
    this.ragSearchResults = document.getElementById("ragSearchResults");

    // Modals
    this.phoneSimModal = document.getElementById("phoneSimModal");
    this.openPhoneSimBtn = document.getElementById("openPhoneSimBtn");
    this.acceptCallBtn = document.getElementById("acceptCallBtn");
    this.declineCallBtn = document.getElementById("declineCallBtn");

    this.chunkInspectorModal = document.getElementById("chunkInspectorModal");
    this.closeInspectorBtn = document.getElementById("closeInspectorBtn");
    this.inspectorDocTitle = document.getElementById("inspectorDocTitle");
    this.inspectorChunksList = document.getElementById("inspectorChunksList");

    this.addStudentModal = document.getElementById("addStudentModal");
    this.openAddStudentModalBtn = document.getElementById(
      "openAddStudentModalBtn",
    );
    this.closeAddStudentBtn = document.getElementById("closeAddStudentBtn");
    this.cancelAddStudentBtn = document.getElementById("cancelAddStudentBtn");
    this.addStudentForm = document.getElementById("addStudentForm");

    // Delete Document Modal
    this.deleteDocModal = document.getElementById("deleteDocModal");
    this.closeDeleteModalBtn = document.getElementById("closeDeleteModalBtn");
    this.cancelDeleteBtn = document.getElementById("cancelDeleteBtn");
    this.confirmDeleteBtn = document.getElementById("confirmDeleteBtn");
    this.deleteDocName = document.getElementById("deleteDocName");
    this.pendingDeleteDocId = null;
    this.pendingDeleteDocName = null;

    // Transcripts Elements (MongoDB)
    this.transcriptsTable = document.getElementById("transcriptsTable");
    this.transcriptsBody = document.getElementById("transcriptsBody");
    this.refreshTranscriptsBtn = document.getElementById("refreshTranscriptsBtn");
    this.transcriptModal = document.getElementById("transcriptModal");
    this.closeTranscriptModalBtn = document.getElementById("closeTranscriptModalBtn");
    this.closeTranscriptModalActionBtn = document.getElementById("closeTranscriptModalActionBtn");
    this.transcriptModalTitle = document.getElementById("transcriptModalTitle");
    this.transcriptModalMeta = document.getElementById("transcriptModalMeta");
    this.transcriptModalBody = document.getElementById("transcriptModalBody");
  }

  initEventListeners() {
    // Tabs (WAI-ARIA tablist: click + arrow-key navigation)
    this.tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        this.switchTab(btn.getAttribute("data-tab"));
      });
      btn.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        e.preventDefault();
        const tabs = Array.from(this.tabButtons);
        const idx = tabs.indexOf(btn);
        const next =
          e.key === "ArrowRight"
            ? tabs[(idx + 1) % tabs.length]
            : tabs[(idx - 1 + tabs.length) % tabs.length];
        next.focus();
        this.switchTab(next.getAttribute("data-tab"));
      });
    });

    // Theme toggle (persisted, respects OS preference on first visit)
    this.initTheme();

    // Language Change
    this.languageSelect.addEventListener("change", (e) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(
          JSON.stringify({ event: "start", language_code: e.target.value }),
        );
      }
    });

    // Mode switch
    this.modeAutoVadBtn.addEventListener("click", () =>
      this.setCallMode("auto_vad"),
    );
    this.modePttBtn.addEventListener("click", () => this.setCallMode("ptt"));

    // Call Toggle
    this.startCallBtn.addEventListener("click", () => this.toggleVoiceCall());
    if (this.micOrb) {
      this.micOrb.style.cursor = "pointer";
      this.micOrb.addEventListener("click", () => this.toggleVoiceCall());
    }

    // Push to talk (PTT)
    this.pushToTalkBtn.addEventListener("mousedown", () =>
      this.startPushToTalk(),
    );
    window.addEventListener("mouseup", () => this.stopPushToTalk());
    window.addEventListener("pointerup", () => this.stopPushToTalk());
    window.addEventListener("pointercancel", () => this.stopPushToTalk());
    window.addEventListener("blur", () => this.stopPushToTalk());
    this.pushToTalkBtn.addEventListener("touchstart", (e) => {
      e.preventDefault();
      this.startPushToTalk();
    });
    window.addEventListener("touchend", () => this.stopPushToTalk());

    // Interrupt Button
    this.interruptBtn.addEventListener("click", () => this.interruptAgent(true));

    // Instant Text Question Form
    if (this.sendTextQueryBtn && this.textQueryInput) {
      const submitQuery = () => {
        const val = this.textQueryInput.value.trim();
        if (!val) return;
        this.sendTextQuery(val);
        this.textQueryInput.value = "";
      };
      this.sendTextQueryBtn.addEventListener("click", submitQuery);
      this.textQueryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") submitQuery();
      });
    }

    // Clear Conversation Feed listener
    if (this.clearConversationBtn && this.conversationFeed) {
      this.clearConversationBtn.addEventListener("click", () => {
        this.conversationFeed.innerHTML = `
          <div class="msg-bubble assistant-bubble" style="display: flex; gap: 10px; align-items: flex-start;">
            <span style="background: var(--primary, #3b82f6); color: white; border-radius: 50%; width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; flex-shrink: 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">P</span>
            <div style="flex: 1; background: var(--surface); padding: 10px 14px; border-radius: 0 12px 12px 12px; border: 1px solid var(--border); line-height: 1.45; color: var(--ink);">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: 600; font-size: 0.78rem; color: var(--primary, #3b82f6);">Priya (DDU IT)</span>
                <span class="lang-tag" style="font-size: 0.68rem; background: var(--border); padding: 1px 6px; border-radius: 4px; color: var(--muted);">ગુજરાતી</span>
              </div>
              <span class="bubble-text">નમસ્તે, હું ડીડીયુ આઈટી ડિપાર્ટમેન્ટમાંથી પ્રિયા વાત કરું છું. હું તમારી શું મદદ કરી શકું? તમે ગુજરાતી, હિન્દી અથવા અંગ્રેજીમાં વાત કરી શકો છો.</span>
            </div>
          </div>
        `;
        this.currentAssistantBubble = null;
        this.currentAssistantTextSpan = null;
        this.currentThinkingIndicator = null;
      });
    }

    // Call History refresh
    if (this.refreshCallsBtn) {
      this.refreshCallsBtn.addEventListener("click", () =>
        this.loadCallHistory(),
      );
    }

    // RAG Dropzone
    this.pdfDropzone.addEventListener("click", () => this.pdfFileInput.click());
    this.pdfFileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) this.handleFileUpload(e.target.files[0]);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      this.pdfDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        this.pdfDropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      this.pdfDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        this.pdfDropzone.classList.remove("dragover");
      });
    });

    this.pdfDropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length > 0)
        this.handleFileUpload(e.dataTransfer.files[0]);
    });

    this.refreshDocsBtn.addEventListener("click", () => this.loadDocuments());

    // Documents Table Event Delegation (Inspect & Delete)
    if (this.documentsTableBody) {
      this.documentsTableBody.addEventListener("click", (e) => {
        const deleteBtn = e.target.closest(".btn-delete");
        if (deleteBtn) {
          e.preventDefault();
          const docId = deleteBtn.getAttribute("data-doc-id");
          const filename = deleteBtn.getAttribute("data-filename") || "document";
          this.openDeleteModal(docId, filename);
          return;
        }

        const inspectBtn = e.target.closest(".btn-inspect");
        if (inspectBtn) {
          e.preventDefault();
          const docId = inspectBtn.getAttribute("data-doc-id");
          const filename = inspectBtn.getAttribute("data-filename") || "document";
          this.inspectChunks(docId, filename);
          return;
        }
      });
    }

    // Delete Document Modal Listeners
    if (this.closeDeleteModalBtn) {
      this.closeDeleteModalBtn.addEventListener("click", () => this.closeDeleteModal());
    }
    if (this.cancelDeleteBtn) {
      this.cancelDeleteBtn.addEventListener("click", () => this.closeDeleteModal());
    }
    if (this.confirmDeleteBtn) {
      this.confirmDeleteBtn.addEventListener("click", () => this.confirmDeleteDocument());
    }
    if (this.deleteDocModal) {
      this.deleteDocModal.addEventListener("click", (e) => {
        if (e.target === this.deleteDocModal) this.closeDeleteModal();
      });
    }

    // RAG Search Tester
    this.ragSearchBtn.addEventListener("click", () => this.executeRagSearch());
    this.ragSearchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.executeRagSearch();
    });

    // Telephony Simulator Modal
    this.openPhoneSimBtn.addEventListener("click", () => {
      this.phoneSimModal.style.display = "flex";
    });
    this.declineCallBtn.addEventListener("click", () => {
      this.phoneSimModal.style.display = "none";
    });
    this.acceptCallBtn.addEventListener("click", () => {
      this.phoneSimModal.style.display = "none";
      this.switchTab("voice");
      if (!this.isCallActive) this.toggleVoiceCall();
    });

    // Chunk Inspector Modal
    if (this.closeInspectorBtn) {
      this.closeInspectorBtn.addEventListener("click", () => {
        if (this.chunkInspectorModal) this.chunkInspectorModal.style.display = "none";
      });
    }
    if (this.chunkInspectorModal) {
      this.chunkInspectorModal.addEventListener("click", (e) => {
        if (e.target === this.chunkInspectorModal) {
          this.chunkInspectorModal.style.display = "none";
        }
      });
    }

    // Add Student Modal
    this.openAddStudentModalBtn.addEventListener("click", () => {
      this.addStudentModal.style.display = "flex";
    });
    this.closeAddStudentBtn.addEventListener("click", () => {
      this.addStudentModal.style.display = "none";
    });
    this.cancelAddStudentBtn.addEventListener("click", () => {
      this.addStudentModal.style.display = "none";
    });
    this.addStudentForm.addEventListener("submit", (e) =>
      this.handleAddStudentSubmit(e),
    );

    // Transcripts Listeners (MongoDB)
    if (this.refreshTranscriptsBtn) {
      this.refreshTranscriptsBtn.addEventListener("click", () => this.loadRecentTranscripts());
    }
    if (this.closeTranscriptModalBtn) {
      this.closeTranscriptModalBtn.addEventListener("click", () => this.closeTranscriptModal());
    }
    if (this.closeTranscriptModalActionBtn) {
      this.closeTranscriptModalActionBtn.addEventListener("click", () => this.closeTranscriptModal());
    }
    if (this.transcriptModal) {
      this.transcriptModal.addEventListener("click", (e) => {
        if (e.target === this.transcriptModal) this.closeTranscriptModal();
      });
    }
  }

  initTheme() {
    const saved = localStorage.getItem("univoice-theme");
    const prefersDark =
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;
    this.setTheme(saved || (prefersDark ? "dark" : "light"));
    if (this.themeToggle) {
      this.themeToggle.addEventListener("click", () => {
        const current =
          document.documentElement.getAttribute("data-theme") === "dark"
            ? "light"
            : "dark";
        this.setTheme(current);
      });
    }
  }

  setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("univoice-theme", theme);
    if (this.themeToggle)
      this.themeToggle.innerText = theme === "dark" ? "Light" : "Dark";
  }

  setCallMode(mode) {
    this.callMode = mode;
    if (mode === "auto_vad") {
      this.modeAutoVadBtn.classList.add("active");
      this.modePttBtn.classList.remove("active");
      this.pushToTalkBtn.style.display = "none";
      if (this.isCallActive)
        this.updateStateText(
          "Hands-free on: just start speaking whenever you are ready.",
        );
    } else {
      this.modeAutoVadBtn.classList.remove("active");
      this.modePttBtn.classList.add("active");
      if (this.isCallActive) {
        this.pushToTalkBtn.style.display = "inline-flex";
        this.pushToTalkBtn.disabled = false;
        this.updateStateText(
          "Hold-to-speak on: press and hold the button while you talk.",
        );
      }
    }
  }

  switchTab(targetTab) {
    this.tabButtons.forEach((b) => {
      const isActive = b.getAttribute("data-tab") === targetTab;
      b.classList.toggle("active", isActive);
      b.setAttribute("aria-selected", isActive ? "true" : "false");
      if (isActive) b.removeAttribute("tabindex");
      else b.setAttribute("tabindex", "-1");
    });
    this.tabPanes.forEach((p) => p.classList.remove("active"));

    const activePane = document.getElementById(`tab-${targetTab}`);

    if (activePane) activePane.classList.add("active");

    if (targetTab === "database") this.loadDatabaseData();
    if (targetTab === "rag") this.loadDocuments();
    if (targetTab === "diagnostics") this.checkSystemStatus();
  }

  // =========================================================================
  // WebSocket Connection
  // =========================================================================

  initWebSocket() {
    const defaultHost = window.location.host;
    const backendHost =
      window.BACKEND_HOST ||
      (window.BACKEND_URL
        ? window.BACKEND_URL.replace(/^https?:\/\//, "")
        : defaultHost);
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${backendHost}/ws/call`;

    this.setWsBadge("connecting", "Connecting...");
    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      this.setWsBadge("connected", "Online & Ready");
      const isReconnect = !!this.hasConnectedOnce;
      this.hasConnectedOnce = true;
      this.ws.send(
        JSON.stringify({
          event: "start",
          language_code: this.languageSelect.value,
          is_reconnect: isReconnect,
        }),
      );
    };

    this.ws.onclose = () => {
      this.setWsBadge("offline", "Disconnected");
      setTimeout(() => this.initWebSocket(), 3000);
    };

    this.ws.onerror = (err) => {
      console.error("[ws] error:", err);
      this.setWsBadge("offline", "Connection Error");
    };

    this.ws.onmessage = async (event) => {
      if (event.data instanceof ArrayBuffer) {
        this.enqueueAudioChunk(event.data);
      } else {
        try {
          const msg = JSON.parse(event.data);
          this.handleServerEvent(msg);
        } catch (e) {
          console.error("JSON parse error:", e);
        }
      }
    };
  }

  setWsBadge(status, text) {
    if (!this.wsStatusBadge) return;
    this.wsStatusBadge.className = "status";
    const dot = this.wsStatusBadge.querySelector(".status-dot");
    if (status === "connected") {
      this.wsStatusBadge.style.color = "var(--ink)";
      this.wsStatusBadge.style.background = "var(--surface)";
      this.wsStatusBadge.style.borderColor = "var(--border)";
      if (dot) dot.style.background = "var(--green)";
    } else if (status === "connecting") {
      this.wsStatusBadge.style.color = "var(--muted)";
      this.wsStatusBadge.style.background = "var(--surface)";
      this.wsStatusBadge.style.borderColor = "var(--border)";
      if (dot) dot.style.background = "var(--faint)";
    } else {
      this.wsStatusBadge.style.color = "var(--red-tx)";
      this.wsStatusBadge.style.background = "var(--surface)";
      this.wsStatusBadge.style.borderColor = "var(--red-tx)";
      if (dot) dot.style.background = "var(--red-tx)";
    }
    const textSpan = this.wsStatusBadge.querySelector(".status-text");
    if (textSpan) textSpan.innerText = text;
  }

  handleServerEvent(msg) {
    if (msg.event === "user_transcript") {
      this.updateStateText(
        `Checking records for: "${msg.text.length > 60 ? msg.text.slice(0, 60) + "…" : msg.text}"`,
      );
      if (msg.language && this.languageSelect) {
        this.languageSelect.value = msg.language;
      }
      this.addUserMessageBubble(msg.text, msg.language);
      this.showThinkingIndicator();
    } else if (msg.event === "agent_filler") {
      this.updateStateText("Checking official university records…");
      this.micOrb.classList.add("agent-speaking");
      this.interruptBtn.style.display = "inline-flex";
    } else if (msg.event === "agent_thinking") {
      this.updateStateText(
        "Checking official records and preparing your answer…",
      );
      this.showThinkingIndicator();
    } else if (msg.event === "tool_executed") {
      this.addToolExecutionBadge(msg.tool, msg.args, msg.preview);
    } else if (msg.event === "agent_partial_text") {
      this.updateStateText("Assistant is answering — listen closely.");
      this.interruptBtn.style.display = "inline-flex";
      this.appendAssistantStreamingChunk(msg.text);
    } else if (msg.event === "agent_done") {
      if (msg.language && this.languageSelect) {
        this.languageSelect.value = msg.language;
      }
      this.finalizeAssistantBubble(msg.full_text, msg.language, msg.call_hangup);
      if (msg.call_hangup) {
        this.updateStateText("Call concluded. Disconnecting...");
      } else {
        this.updateStateText("Ready. Speak your next question whenever ready.");
      }
      this.loadCallHistory();
      this.loadRecentTranscripts();
    } else if (msg.event === "call_started_ack") {
      this.updateStateText(
        `Call ${msg.call_id || ""} started. Speak naturally — I am listening.`,
      );
    } else if (msg.event === "call_ended_ack") {
      this.updateStateText("Call logged. See Call History for details.");
      setTimeout(() => {
        this.loadCallHistory();
        this.loadRecentTranscripts();
      }, 1500);
    } else if (msg.event === "call_ended") {
      if (this.isCallActive) {
        this.toggleVoiceCall();
      }
      this.finalizeAssistantBubble(null, null, true);
      this.updateStateText("Call disconnected and logged.");
      setTimeout(() => {
        this.loadCallHistory();
        this.loadRecentTranscripts();
      }, 1500);
    } else if (msg.event === "interrupted") {
      this.interruptAgent(false);
      this.finalizeAssistantBubble(null, null, false);
      this.updateStateText("Listening — please speak your question.");
    } else if (msg.event === "empty_transcript") {
      this.removeThinkingIndicator();
      if (msg.had_filler && this.isPlayingAudio) {
        this.interruptAgent(false);
      }
      const dur = msg.duration ? `(${msg.duration.toFixed(1)}s)` : "";
      const rms = msg.rms !== undefined ? msg.rms : 0;
      let hint = "Didn't catch any words in that audio.";
      if (rms < 0.003) {
        hint =
          "Microphone input was completely silent (volume is zero). Please check that the correct microphone is selected and unmuted in your system/browser settings.";
      } else if (msg.duration && msg.duration < 0.8) {
        hint = `Audio was very short ${dur}. Please speak a full question or try the 'Hold to speak' button.`;
      } else {
        hint = `Audio was received ${dur}, but words were not recognized clearly. Try speaking a bit closer to the microphone or try 'Hold to speak'.`;
      }
      this.updateStateText(hint);
    } else if (msg.event === "error") {
      this.removeThinkingIndicator();
      this.updateStateText(
        msg.message || "Something went wrong while answering.",
      );
    }
  }

  // =========================================================================
  // Live Conversation Transcript Stream Helpers
  // =========================================================================

  getLanguageLabel(code) {
    const map = {
      "gu-IN": "ગુજરાતી",
      "gu": "ગુજરાતી",
      "hi-IN": "हिंदी",
      "hi": "हिंदी",
      "en-IN": "English",
      "en": "English",
    };
    return map[code] || (this.languageSelect ? this.languageSelect.options[this.languageSelect.selectedIndex]?.text : "ગુજરાતી");
  }

  addUserMessageBubble(text, lang) {
    if (!this.conversationFeed) return;
    this.removeThinkingIndicator();
    this.currentAssistantBubble = null;
    this.currentAssistantTextSpan = null;

    const langLabel = this.getLanguageLabel(lang);
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble user-bubble";
    bubble.style.cssText = "display: flex; gap: 10px; justify-content: flex-end; align-items: flex-start; margin-top: 4px;";
    bubble.innerHTML = `
      <div style="max-width: 82%; background: var(--primary, #3b82f6); color: white; padding: 10px 14px; border-radius: 12px 0 12px 12px; line-height: 1.45; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; gap: 8px;">
          <span style="font-weight: 600; font-size: 0.76rem; opacity: 0.9;">You</span>
          <span style="font-size: 0.68rem; background: rgba(255,255,255,0.22); padding: 1px 6px; border-radius: 4px;">${langLabel}</span>
        </div>
        <span style="font-size: 0.9rem;">${this.escapeHtml(text)}</span>
      </div>
      <span style="background: var(--ink-subtle, #64748b); color: white; border-radius: 50%; width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 700; flex-shrink: 0;">U</span>
    `;
    this.conversationFeed.appendChild(bubble);
    this.scrollConversationFeed();
  }

  showThinkingIndicator() {
    if (!this.conversationFeed || this.currentThinkingIndicator) return;
    const indicator = document.createElement("div");
    indicator.className = "msg-bubble thinking-bubble";
    indicator.style.cssText = "display: flex; gap: 10px; align-items: center; margin: 4px 0;";
    indicator.innerHTML = `
      <span style="background: var(--primary, #3b82f6); color: white; border-radius: 50%; width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; flex-shrink: 0;">P</span>
      <div style="background: var(--surface); padding: 8px 12px; border-radius: 0 10px 10px 10px; border: 1px solid var(--border); font-size: 0.8rem; color: var(--muted); display: inline-flex; align-items: center; gap: 6px;">
        <span>Priya is checking university records…</span>
      </div>
    `;
    this.conversationFeed.appendChild(indicator);
    this.currentThinkingIndicator = indicator;
    this.scrollConversationFeed();
  }

  removeThinkingIndicator() {
    if (this.currentThinkingIndicator && this.currentThinkingIndicator.parentNode) {
      this.currentThinkingIndicator.remove();
    }
    this.currentThinkingIndicator = null;
  }

  addToolExecutionBadge(toolName, args, preview) {
    if (!this.conversationFeed) return;
    const badge = document.createElement("div");
    badge.style.cssText = "display: flex; align-items: center; gap: 6px; padding: 4px 10px; background: rgba(59, 130, 246, 0.08); border: 1px dashed rgba(59, 130, 246, 0.3); border-radius: 6px; font-size: 0.76rem; color: var(--primary, #3b82f6); margin: 2px 0 2px 36px; max-width: fit-content;";
    const label = preview || `${toolName}()`;
    badge.innerHTML = `<span>🛠️</span> <span class="mono">${this.escapeHtml(label)}</span>`;
    this.conversationFeed.appendChild(badge);
    this.scrollConversationFeed();
  }

  appendAssistantStreamingChunk(sentence) {
    if (!this.conversationFeed) return;
    this.removeThinkingIndicator();

    if (!this.currentAssistantBubble) {
      const bubble = document.createElement("div");
      bubble.className = "msg-bubble assistant-bubble";
      bubble.style.cssText = "display: flex; gap: 10px; align-items: flex-start; margin-top: 4px;";
      bubble.innerHTML = `
        <span style="background: var(--primary, #3b82f6); color: white; border-radius: 50%; width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; flex-shrink: 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">P</span>
        <div style="flex: 1; background: var(--surface); padding: 10px 14px; border-radius: 0 12px 12px 12px; border: 1px solid var(--border); line-height: 1.45; color: var(--ink);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-weight: 600; font-size: 0.78rem; color: var(--primary, #3b82f6);">Priya (DDU IT)</span>
            <span class="assistant-lang-tag" style="font-size: 0.68rem; background: var(--border); padding: 1px 6px; border-radius: 4px; color: var(--muted);">${this.getLanguageLabel(this.languageSelect ? this.languageSelect.value : "gu-IN")}</span>
          </div>
          <span class="bubble-text" style="font-size: 0.9rem;"></span>
        </div>
      `;
      this.conversationFeed.appendChild(bubble);
      this.currentAssistantBubble = bubble;
      this.currentAssistantTextSpan = bubble.querySelector(".bubble-text");
    }

    if (this.currentAssistantTextSpan) {
      const existing = this.currentAssistantTextSpan.innerText.trim();
      this.currentAssistantTextSpan.innerText = existing ? `${existing} ${sentence}` : sentence;
    }
    this.scrollConversationFeed();
  }

  finalizeAssistantBubble(fullText, lang, isHangup) {
    this.removeThinkingIndicator();
    if (!this.currentAssistantBubble && fullText && fullText.trim()) {
      this.appendAssistantStreamingChunk(fullText);
    }
    if (this.currentAssistantBubble) {
      if (fullText && this.currentAssistantTextSpan) {
        this.currentAssistantTextSpan.innerText = fullText;
      }
      const tag = this.currentAssistantBubble.querySelector(".assistant-lang-tag");
      if (tag && lang) {
        tag.innerText = this.getLanguageLabel(lang);
      }
      if (isHangup) {
        const hangupBadge = document.createElement("span");
        hangupBadge.style.cssText = "display: inline-block; font-size: 0.7rem; background: rgba(239,68,68,0.1); color: #ef4444; padding: 2px 6px; border-radius: 4px; margin-top: 6px;";
        hangupBadge.innerText = "📞 Call Concluded";
        this.currentAssistantBubble.querySelector("div").appendChild(hangupBadge);
      }
    }
    this.currentAssistantBubble = null;
    this.currentAssistantTextSpan = null;
    this.scrollConversationFeed();
  }

  scrollConversationFeed() {
    if (this.conversationFeed) {
      this.conversationFeed.scrollTop = this.conversationFeed.scrollHeight;
    }
  }

  async loadCallHistory() {
    if (!this.callHistoryBody) return;
    try {
      const res = await fetch("/api/calls/history?limit=50");
      const data = await res.json();
      const calls = data.calls || [];

      if (calls.length === 0) {
        this.callHistoryBody.innerHTML = `
          <tr>
            <td colspan="7" class="muted">No calls yet. Press Start voice call to begin.</td>
          </tr>
        `;
        return;
      }

      this.callHistoryBody.innerHTML = calls
        .map((c) => {
          let queries = [];
          try {
            queries = JSON.parse(c.queries_json || "[]");
          } catch (e) {
            queries = [];
          }
          const first = queries.length ? queries[0].text : "";
          const extra =
            queries.length > 1 ? ` (+${queries.length - 1} more)` : "";
          const purpose = first ? this.escapeHtml(first) + extra : "—";
          const when = c.started_at
            ? new Date(c.started_at).toLocaleString()
            : "—";
          const ongoing = !c.ended_at;
          const intent = c.intent || "Admission & General";
          const lead = c.lead_status || "Prospective Applicant";
          const summary = c.summary || purpose;
          const sentimentIcon = c.sentiment === "Positive" ? "😊" : c.sentiment === "Frustrated" ? "⚠️" : "💬";

          return `
          <tr>
            <td><code class="mono">${this.escapeHtml(c.call_id || "")}</code></td>
            <td>${this.escapeHtml(c.caller_number || "Web")}</td>
            <td>${ongoing ? '<span class="tag">Ongoing</span>' : this.formatDuration(c.duration_seconds || 0)}</td>
            <td>
              <div style="display:flex; flex-direction:column; gap:4px;">
                <span class="tag" style="font-size:0.75rem;">${sentimentIcon} ${this.escapeHtml(intent)}</span>
                <span class="muted small">${this.escapeHtml(lead)}</span>
              </div>
            </td>
            <td class="query-cell" title="${this.escapeHtml(summary)}">
              <span style="font-size:0.85rem; line-height:1.3;">${this.escapeHtml(summary)}</span>
            </td>
            <td class="muted small">${when}</td>
            <td>
              <button class="btn btn-sm" style="font-size:0.75rem; padding: 4px 8px;" onclick="window.__appInstance.openTranscriptModal('${this.escapeHtml(c.call_id || "")}')">
                View Transcript
              </button>
            </td>
          </tr>
        `;
        })
        .join("");
    } catch (e) {
      console.error("Call history load error:", e);
    }
  }

  async loadRecentTranscripts() {
    if (!this.transcriptsBody) return;
    try {
      const res = await fetch("/api/transcripts/recent?limit=10");
      const data = await res.json();
      const transcripts = data.transcripts || [];

      if (transcripts.length === 0) {
        this.transcriptsBody.innerHTML = `
          <tr>
            <td colspan="7" class="muted">No call transcripts recorded in MongoDB yet. Make a call to see live transcripts.</td>
          </tr>
        `;
        return;
      }

      this.transcriptsBody.innerHTML = transcripts
        .map((t) => {
          const cid = t.call_id || "—";
          const caller = t.caller_number || "Web";
          const lang = t.language || "gu-IN";
          const dur = this.formatDuration(t.duration_seconds || 0);
          const turns = t.turn_count || (t.turns ? t.turns.length : 0);
          const when = t.created_at ? new Date(t.created_at).toLocaleString() : "—";

          return `
          <tr>
            <td><code class="mono">${this.escapeHtml(cid)}</code></td>
            <td>${this.escapeHtml(caller)}</td>
            <td><span class="tag" style="font-size:0.75rem;">${this.escapeHtml(lang)}</span></td>
            <td>${dur}</td>
            <td><span class="tag" style="font-size:0.75rem;">${turns} turns</span></td>
            <td class="muted small">${when}</td>
            <td>
              <button class="btn btn-sm btn-primary" style="font-size:0.75rem; padding: 4px 10px;" onclick="window.__appInstance.openTranscriptModal('${this.escapeHtml(cid)}')">
                View Dialogue
              </button>
            </td>
          </tr>
        `;
        })
        .join("");
    } catch (e) {
      console.error("Transcripts load error:", e);
      if (this.transcriptsBody) {
        this.transcriptsBody.innerHTML = `
          <tr>
            <td colspan="7" class="muted" style="color:var(--danger)">Error loading transcripts from server.</td>
          </tr>
        `;
      }
    }
  }

  async openTranscriptModal(callId) {
    if (!this.transcriptModal) return;
    this.transcriptModal.style.display = "flex";
    if (this.transcriptModalTitle) {
      this.transcriptModalTitle.textContent = `Call Transcript · ${callId}`;
    }
    if (this.transcriptModalMeta) {
      this.transcriptModalMeta.textContent = "Loading dialogue from MongoDB...";
    }
    if (this.transcriptModalBody) {
      this.transcriptModalBody.innerHTML = `<div class="muted small" style="text-align:center; padding: 20px;">Fetching dialogue turns...</div>`;
    }

    try {
      const res = await fetch(`/api/transcripts/${encodeURIComponent(callId)}`);
      if (!res.ok) throw new Error("Transcript not found");
      const data = await res.json();
      const doc = data.transcript || {};
      const turns = doc.turns || [];

      if (this.transcriptModalMeta) {
        const caller = doc.caller_number || "Web";
        const lang = doc.language || "gu-IN";
        const dur = this.formatDuration(doc.duration_seconds || 0);
        this.transcriptModalMeta.textContent = `Caller: ${caller} · Lang: ${lang} · Duration: ${dur} · Source: ${data.source || 'MongoDB'}`;
      }

      if (turns.length === 0) {
        this.transcriptModalBody.innerHTML = `
          <div class="muted small" style="text-align:center; padding: 20px;">No speech turns recorded for this call yet.</div>
        `;
        return;
      }

      this.transcriptModalBody.innerHTML = turns
        .map((t) => {
          const isUser = t.role === "user";
          const speakerName = isUser ? "Caller" : "Priya (DDU IT)";
          const bubbleBg = isUser ? "var(--bg-card, #1e2430)" : "var(--primary-subtle, rgba(46, 117, 89, 0.15))";
          const borderClr = isUser ? "var(--border, #2a3344)" : "var(--primary, #2e7559)";
          const badgeColor = isUser ? "#3b82f6" : "#10b981";

          return `
          <div style="display:flex; flex-direction:column; align-items:flex-start; margin-bottom: 8px;">
            <div style="display:flex; align-items:center; gap: 6px; margin-bottom: 3px;">
              <span style="font-size:0.7rem; font-weight:600; color:${badgeColor}; text-transform:uppercase;">${speakerName}</span>
            </div>
            <div style="background:${bubbleBg}; border:1px solid ${borderClr}; border-radius:8px; padding: 8px 12px; font-size:0.88rem; line-height:1.4; max-width:98%; word-break:break-word;">
              ${this.escapeHtml(t.text || "")}
            </div>
          </div>
        `;
        })
        .join("");
    } catch (e) {
      console.error("Fetch transcript error:", e);
      if (this.transcriptModalBody) {
        this.transcriptModalBody.innerHTML = `
          <div style="color:var(--danger); text-align:center; padding: 20px;">Could not load transcript for call ${this.escapeHtml(callId)}.</div>
        `;
      }
    }
  }

  closeTranscriptModal() {
    if (this.transcriptModal) {
      this.transcriptModal.style.display = "none";
    }
  }

  formatDuration(totalSeconds) {
    const s = Math.max(0, parseInt(totalSeconds, 10) || 0);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const mm = h > 0 ? String(m).padStart(2, "0") : String(m);
    return `${h > 0 ? h + ":" : ""}${mm}:${String(sec).padStart(2, "0")}`;
  }

  escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (ch) => {
      const map = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      };
      return map[ch];
    });
  }

  // =========================================================================
  // Small usability helpers: footer year, table search, keyboard access
  // =========================================================================

  initUxEnhancements() {
    const yearEl = document.getElementById("footerYear");
    if (yearEl) yearEl.textContent = String(new Date().getFullYear());

    // Keyboard access for dropzone
    if (this.pdfDropzone && this.pdfFileInput) {
      this.pdfDropzone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          this.pdfFileInput.click();
        }
      });
    }

    // Escape closes any open modal
    window.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      [
        this.phoneSimModal,
        this.chunkInspectorModal,
        this.addStudentModal,
      ].forEach((m) => {
        if (m) m.style.display = "none";
      });
    });

    // Click outside modal card closes it
    [
      this.phoneSimModal,
      this.chunkInspectorModal,
      this.addStudentModal,
    ].forEach((m) => {
      if (!m) return;
      m.addEventListener("click", (e) => {
        if (e.target === m) m.style.display = "none";
      });
    });

    // Staff table search (students + placements + admissions)
    const searchInput = document.getElementById("studentSearchInput");
    if (searchInput) {
      searchInput.addEventListener("input", () => {
        const q = searchInput.value.trim().toLowerCase();
        [
          "studentsTableBody",
          "placementsTableBody",
          "admissionsTableBody",
        ].forEach((tbodyId) => {
          const tbody = document.getElementById(tbodyId);
          if (!tbody) return;
          tbody.querySelectorAll("tr").forEach((row) => {
            if (row.querySelector(".loading-cell")) return;
            const text = row.innerText.toLowerCase();
            row.style.display = !q || text.includes(q) ? "" : "none";
          });
        });
      });
    }
  }

  // =========================================================================
  // WebAudio Recording & Auto-VAD
  // =========================================================================

  sendTextQuery(text) {
    if (!text || !text.trim()) return;
    const cleanText = text.trim();
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      // Ensure audio context is ready so synthesized response audio can play
      if (!this.audioContext) {
        try {
          this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {}
      }
      if (this.audioContext && this.audioContext.state === "suspended") {
        this.audioContext.resume();
      }
      this.updateStateText(`Asking: "${cleanText.length > 50 ? cleanText.slice(0, 50) + "…" : cleanText}"…`);
      this.ws.send(
        JSON.stringify({
          event: "text_query",
          text: cleanText,
          language_code: this.languageSelect.value,
        }),
      );
    } else {
      this.updateStateText("⚠️ Voice gateway is reconnecting, please try again in a moment…");
      this.initWebSocket();
    }
  }

  async toggleVoiceCall() {
    if (!this.isCallActive) {
      this.startCallBtn.disabled = true;
      this.startCallBtn.innerText = "Requesting mic…";
      this.updateStateText("Requesting microphone permission from your browser…");

      try {
        await this.initAudioContext();
        this.isCallActive = true;
        this.startCallBtn.disabled = false;
        this.callStartupGrace = Date.now() + 800; // Ignore mouse clicks & startup transients for 800ms
        this.startCallBtn.innerText = "End call";
        this.startCallBtn.classList.remove("btn-primary");

        // Open a call log entry for this voice call
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(
            JSON.stringify({
              event: "call_started",
              language_code: this.languageSelect.value,
              caller_number: "Web",
            }),
          );
        }

        if (this.callMode === "ptt") {
          this.pushToTalkBtn.style.display = "inline-flex";
          this.pushToTalkBtn.disabled = false;
          this.updateStateText(
            "Connected. Hold the button, speak clearly, then release.",
          );
        } else {
          this.pushToTalkBtn.style.display = "none";
          this.startAutoVadListening();
          this.updateStateText("Connected. Speak naturally — I am listening.");
        }

        // Start timer
        this.callStartTime = Date.now();
        this.callTimerInterval = setInterval(() => {
          const diff = Math.floor((Date.now() - this.callStartTime) / 1000);
          const mins = String(Math.floor(diff / 60)).padStart(2, "0");
          const secs = String(diff % 60).padStart(2, "0");
          if (this.callDuration)
            this.callDuration.innerText = `${mins}:${secs}`;
        }, 1000);
      } catch (err) {
        console.error("Mic access error:", err);
        this.isCallActive = false;
        this.startCallBtn.disabled = false;
        this.startCallBtn.innerText = "Start voice call";
        this.startCallBtn.classList.add("btn-primary");

        let userAdvice = "Microphone permission is required for live voice calls. Please allow microphone access in your browser address bar, or ask your question in the query box below.";
        if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
          userAdvice = "No microphone hardware found. Please connect a microphone, or ask your question via the text box below.";
        } else if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
          userAdvice = "Microphone access was blocked. Please click the tune/lock icon in your browser address bar to allow microphone access.";
        }
        this.updateStateText(`⚠️ ${userAdvice}`);
      }
    } else {
      // End call
      this.isCallActive = false;
      this.stopAutoVadListening();
      this.stopRecording();
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach((track) => track.stop());
      }
      this.startCallBtn.innerText = "Start voice call";
      this.startCallBtn.classList.add("btn-primary");
      this.pushToTalkBtn.style.display = "none";
      this.interruptBtn.style.display = "none";
      this.updateStateText("Call ended. Logging to Call History…");
      clearInterval(this.callTimerInterval);
      if (this.callDuration) this.callDuration.innerText = "00:00";
      // Close the call log entry for this voice call
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ event: "call_ended" }));
      } else {
        setTimeout(() => this.loadCallHistory(), 1500);
      }
    }
  }

  async initAudioContext() {
    if (!this.audioContext) {
      this.audioContext = new (
        window.AudioContext || window.webkitAudioContext
      )();
    }
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }
    // Attempt to register high-performance AudioWorklet for off-main-thread processing
    try {
      if (this.audioContext.audioWorklet) {
        await this.audioContext.audioWorklet.addModule("/static/audio-processor.js");
        this.workletSupported = true;
      }
    } catch (err) {
      console.warn("AudioWorklet module notice, fallback processor enabled:", err);
      this.workletSupported = false;
    }
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    this.audioInputNode = this.audioContext.createMediaStreamSource(
      this.mediaStream,
    );
    this.analyserNode = this.audioContext.createAnalyser();
    this.analyserNode.fftSize = 256;
    this.audioInputNode.connect(this.analyserNode);
  }

  startAutoVadListening() {
    if (!this.isCallActive) return;
    if (this.processorNode) {
      try {
        this.processorNode.disconnect();
      } catch (e) {}
      this.processorNode = null;
    }

    this.audioChunks = [];
    this.isSpeechDetected = false;
    this.consecutiveSpeechFrames = 0;
    this.consecutiveBargeInFrames = 0;

    const handleAudioFrame = (input, rms) => {
      if (
        !this.isCallActive ||
        this.callMode !== "auto_vad" ||
        Date.now() < this.callStartupGrace
      )
        return;

      // Slowly track background noise floor when user isn't speaking and agent isn't playing
      if (!this.isSpeechDetected && !this.isPlayingAudio) {
        this.noiseFloor = this.noiseFloor * 0.95 + rms * 0.05;
      }

      // Dynamic speech threshold anchored safely above room noise floor
      const dynamicSpeechThreshold = Math.max(
        this.minSpeechThreshold,
        this.noiseFloor * 2.5 + 0.012,
      );

      // --- BARGE-IN: If customer speaks while agent is speaking, stop agent immediately ---
      if (this.isPlayingAudio) {
        const playbackElapsed = Date.now() - (this.playbackStartTime || 0);
        const isInitialGreeting = (Date.now() - (this.callStartTime || 0)) < 12000;
        // Resilient threshold during speech playback to prevent laptop speaker echo from triggering false interruptions
        const bargeInThreshold = isInitialGreeting
          ? Math.max(0.18, dynamicSpeechThreshold * 4.0)
          : Math.max(0.12, dynamicSpeechThreshold * 2.8);
        const requiredFrames = isInitialGreeting ? 8 : Math.max(5, this.requiredBargeInFrames || 5);

        if (playbackElapsed > 500 && rms > bargeInThreshold) {
          this.consecutiveBargeInFrames++;
          if (this.consecutiveBargeInFrames >= requiredFrames) {
            console.log(
              `🛑 Caller interrupted while agent was speaking (RMS: ${rms.toFixed(3)} > ${bargeInThreshold.toFixed(3)}). Stopping agent playback.`,
            );
            this.consecutiveBargeInFrames = 0;
            this.interruptAgent(true);
            this.isSpeechDetected = true;
            this.consecutiveSpeechFrames = this.requiredSpeechFrames;
            this.micOrb.classList.add("active");
            this.updateStateText(
              "Listening — I stopped speaking, please go ahead.",
            );
            this.audioChunks = [new Float32Array(input)];
            if (this.vadSilenceTimer) {
              clearTimeout(this.vadSilenceTimer);
              this.vadSilenceTimer = null;
            }
          }
        } else {
          this.consecutiveBargeInFrames = 0;
        }
        return;
      }

      this.consecutiveBargeInFrames = 0;

      // --- SPEECH DETECTION ---
      if (rms > dynamicSpeechThreshold) {
        this.consecutiveSpeechFrames++;

        // Only confirm speech after sustained energy (rejects single-frame mouse clicks / coughs)
        if (this.consecutiveSpeechFrames >= this.requiredSpeechFrames) {
          if (!this.isSpeechDetected) {
            this.isSpeechDetected = true;
            this.micOrb.classList.add("active");
            this.updateStateText("Listening — I can hear you, keep speaking.");
          }
          this.audioChunks.push(new Float32Array(input));

          // Clear silence timeout while actively speaking
          if (this.vadSilenceTimer) {
            clearTimeout(this.vadSilenceTimer);
            this.vadSilenceTimer = null;
          }

          // Safety auto-commit if speaking continuously for > 8.0s
          const currentLen = this.audioChunks.reduce(
            (acc, c) => acc + c.length,
            0,
          );
          const actualRate =
            (this.audioContext && this.audioContext.sampleRate) || 16000;
          if (currentLen > actualRate * 8.0) {
            this.commitAutoVadSpeech();
          }
        }
      } else {
        // Below speech threshold
        this.consecutiveSpeechFrames = 0;

        if (this.isSpeechDetected) {
          // Still buffer during natural speech pauses
          this.audioChunks.push(new Float32Array(input));

          if (!this.vadSilenceTimer) {
            this.vadSilenceTimer = setTimeout(() => {
              this.commitAutoVadSpeech();
            }, this.vadSilenceDuration);
          }
        }
      }
    };

    if (this.workletSupported) {
      try {
        this.processorNode = new AudioWorkletNode(
          this.audioContext,
          "voice-agent-audio-processor",
        );
        this.processorNode.port.onmessage = (e) => {
          if (e.data && e.data.event === "audio_data") {
            handleAudioFrame(e.data.buffer, e.data.rms);
          }
        };
        this.audioInputNode.connect(this.processorNode);
        return;
      } catch (err) {
        console.warn("AudioWorkletNode instantiation failed, using fallback:", err);
      }
    }

    // Fallback ScriptProcessor
    const bufferSize = 4096;
    this.processorNode = this.audioContext.createScriptProcessor(
      bufferSize,
      1,
      1,
    );
    window.__vadProcessor = this.processorNode;
    this.processorNode.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      const output = e.outputBuffer.getChannelData(0);
      output.fill(0);
      let sum = 0;
      for (let i = 0; i < input.length; i++) {
        sum += input[i] * input[i];
      }
      const rms = Math.sqrt(sum / input.length);
      handleAudioFrame(input, rms);
    };

    this.audioInputNode.connect(this.processorNode);
    this.processorNode.connect(this.audioContext.destination);
  }

  stopAutoVadListening() {
    if (this.vadSilenceTimer) {
      clearTimeout(this.vadSilenceTimer);
      this.vadSilenceTimer = null;
    }
    if (this.processorNode) {
      this.processorNode.disconnect();
      this.audioInputNode.disconnect(this.processorNode);
      this.processorNode = null;
    }
  }

  commitAutoVadSpeech() {
    this.isSpeechDetected = false;
    this.consecutiveSpeechFrames = 0;
    this.consecutiveBargeInFrames = 0;
    this.micOrb.classList.remove("active");
    this.updateStateText("Processing what you said…");

    const totalLength = this.audioChunks.reduce(
      (acc, chunk) => acc + chunk.length,
      0,
    );
    const actualSampleRate =
      (this.audioContext && this.audioContext.sampleRate) || 16000;
    const minSamples = Math.round(actualSampleRate * 0.45);
    if (totalLength < minSamples) {
      this.audioChunks = [];
      this.updateStateText("Listening. Speak whenever you are ready.");
      return;
    }

    const mergedBuffer = new Float32Array(totalLength);
    let offset = 0;
    let sumSquares = 0;
    for (const chunk of this.audioChunks) {
      mergedBuffer.set(chunk, offset);
      for (let i = 0; i < chunk.length; i++) {
        sumSquares += chunk[i] * chunk[i];
      }
      offset += chunk.length;
    }
    this.audioChunks = [];

    // Verify overall RMS of recorded audio. If below speech threshold, discard as ambient noise
    const overallRms = Math.sqrt(sumSquares / totalLength);
    if (overallRms < this.minSpeechThreshold * 0.40) {
      console.log(`🔇 Discarded ambient noise buffer (RMS: ${overallRms.toFixed(4)})`);
      this.updateStateText("Listening. Speak whenever you are ready.");
      return;
    }

    // Cleanly resample to 16000Hz expected by Sarvam STT
    const resampled = this.resampleTo16k(mergedBuffer, actualSampleRate);
    const wavBlob = this.encodeWAV(resampled, 16000);
    wavBlob.arrayBuffer().then((arrayBuf) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(arrayBuf);
      }
    });
  }

  // --- Push to talk ---
  startPushToTalk() {
    if (!this.isCallActive || this.isRecording) return;
    this.interruptAgent();
    this.isRecording = true;
    this.audioChunks = [];
    this.micOrb.classList.add("active");
    this.updateStateText("Listening. Speak now, then release the button.");

    const bufferSize = 4096;
    this.processorNode = this.audioContext.createScriptProcessor(
      bufferSize,
      1,
      1,
    );
    this.processorNode.onaudioprocess = (e) => {
      if (!this.isRecording) return;
      const input = e.inputBuffer.getChannelData(0);
      this.audioChunks.push(new Float32Array(input));
    };

    this.audioInputNode.connect(this.processorNode);
    this.processorNode.connect(this.audioContext.destination);
  }

  stopPushToTalk() {
    if (!this.isRecording) return;
    this.isRecording = false;
    this.micOrb.classList.remove("active");
    this.updateStateText("Writing down what you said…");

    if (this.processorNode) {
      this.processorNode.disconnect();
      this.audioInputNode.disconnect(this.processorNode);
      this.processorNode = null;
    }

    const totalLength = this.audioChunks.reduce(
      (acc, chunk) => acc + chunk.length,
      0,
    );
    const actualSampleRate =
      (this.audioContext && this.audioContext.sampleRate) || 16000;
    if (totalLength < Math.round(actualSampleRate * 0.2)) {
      this.updateStateText(
        "That was very short. Hold the button and speak clearly.",
      );
      this.audioChunks = [];
      return;
    }

    const mergedBuffer = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.audioChunks) {
      mergedBuffer.set(chunk, offset);
      offset += chunk.length;
    }
    this.audioChunks = [];

    // Cleanly resample to 16000Hz expected by Sarvam STT
    const resampled = this.resampleTo16k(mergedBuffer, actualSampleRate);
    const wavBlob = this.encodeWAV(resampled, 16000);
    wavBlob.arrayBuffer().then((arrayBuf) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(arrayBuf);
      }
    });
  }

  stopRecording() {
    this.isRecording = false;
    this.micOrb.classList.remove("active");
  }

  interruptAgent(broadcast = true) {
    this.playbackQueue = [];
    this.isPlayingAudio = false;
    this.nextAudioStartTime = 0;
    if (this.activeAudioSources && this.activeAudioSources.length > 0) {
      for (const src of this.activeAudioSources) {
        try {
          src.stop();
          src.disconnect();
        } catch (e) {}
      }
      this.activeAudioSources = [];
    }
    this.micOrb.classList.remove("agent-speaking");
    this.interruptBtn.style.display = "none";
    if (broadcast && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ event: "interrupt" }));
    }
    this.updateStateText("Stopped. Ask your next question whenever ready.");
  }

  resampleTo16k(audioBuffer, origSampleRate) {
    if (!origSampleRate || origSampleRate === 16000) return audioBuffer;
    const ratio = origSampleRate / 16000;
    const newLength = Math.round(audioBuffer.length / ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const srcIndex = i * ratio;
      const indexFloor = Math.floor(srcIndex);
      const indexCeil = Math.min(indexFloor + 1, audioBuffer.length - 1);
      const weight = srcIndex - indexFloor;
      result[i] =
        (1 - weight) * audioBuffer[indexFloor] +
        weight * audioBuffer[indexCeil];
    }
    return result;
  }

  encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (view, offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, "data");
    view.setUint32(40, samples.length * 2, true);

    let index = 44;
    for (let i = 0; i < samples.length; i++) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      index += 2;
    }

    return new Blob([view], { type: "audio/wav" });
  }

  // =========================================================================
  // Audio Playback Queue with Gapless Scheduling
  // =========================================================================

  enqueueAudioChunk(arrayBuffer) {
    this.playbackQueue.push(arrayBuffer);
    this.scheduleNextAudioChunks();
  }

  async scheduleNextAudioChunks() {
    if (this.playbackQueue.length === 0) return;

    if (!this.isPlayingAudio) {
      this.playbackStartTime = Date.now();
    }
    this.isPlayingAudio = true;
    this.micOrb.classList.add("agent-speaking");
    this.interruptBtn.style.display = "inline-flex";

    if (!this.audioContext) {
      this.audioContext = new (
        window.AudioContext || window.webkitAudioContext
      )();
    }
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }

    while (this.playbackQueue.length > 0) {
      const chunk = this.playbackQueue.shift();
      try {
        const audioBuffer = await this.audioContext.decodeAudioData(
          chunk.slice(0),
        );
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);

        // Gapless seamless scheduling: align start time with exact end of prior buffer
        const now = this.audioContext.currentTime;
        const startTime = Math.max(now + 0.01, this.nextAudioStartTime);
        source.start(startTime);
        this.nextAudioStartTime = startTime + audioBuffer.duration;

        this.activeAudioSources.push(source);

        source.onended = () => {
          this.activeAudioSources = this.activeAudioSources.filter(
            (s) => s !== source,
          );
          if (
            this.activeAudioSources.length === 0 &&
            this.playbackQueue.length === 0
          ) {
            this.isPlayingAudio = false;
            this.nextAudioStartTime = 0;
            this.micOrb.classList.remove("agent-speaking");
            this.interruptBtn.style.display = "none";
          }
        };
      } catch (e) {
        console.error("Playback decode error:", e);
      }
    }
  }

  // =========================================================================
  // Canvas Visualizer
  // =========================================================================

  startCanvasAnimation() {
    if (!this.canvas || !this.canvasCtx) return;

    let phase = 0;
    const timeData = new Uint8Array(128);

    const draw = () => {
      this.animationFrameId = requestAnimationFrame(draw);
      const width = this.canvas.width;
      const height = this.canvas.height;
      this.canvasCtx.clearRect(0, 0, width, height);

      // Measure real live mic volume from analyserNode
      let liveMicEnergy = 0;
      if (this.analyserNode && this.isCallActive) {
        this.analyserNode.getByteTimeDomainData(timeData);
        let sum = 0;
        for (let i = 0; i < timeData.length; i++) {
          const val = (timeData[i] - 128) / 128;
          sum += val * val;
        }
        liveMicEnergy = Math.sqrt(sum / timeData.length);
      }

      let amplitude = 10;
      let frequency = 0.02;

      if (this.isPlayingAudio) {
        amplitude = 35;
        frequency = 0.035;
      } else if (liveMicEnergy > 0.0025) {
        // Wave dynamically jumps when real sound enters mic!
        amplitude = Math.min(65, 14 + liveMicEnergy * 450);
        frequency = 0.038;
      } else if (this.isSpeechDetected || this.isRecording) {
        amplitude = 35;
        frequency = 0.035;
      }

      const dark =
        document.documentElement.getAttribute("data-theme") === "dark";
      const base = dark ? "244, 244, 242" : "17, 17, 17";
      const blue = dark ? "122, 165, 255" : "29, 78, 216";
      const green = dark ? "95, 206, 138" : "21, 128, 61";

      for (let i = 0; i < 3; i++) {
        this.canvasCtx.beginPath();
        this.canvasCtx.lineWidth = 1.5;

        if (i === 0) this.canvasCtx.strokeStyle = `rgba(${base}, 0.30)`;
        else if (i === 1)
          this.canvasCtx.strokeStyle = `rgba(${blue}, 0.38)`;
        else this.canvasCtx.strokeStyle = `rgba(${green}, 0.32)`;

        for (let x = 0; x < width; x++) {
          const y =
            height / 2 +
            Math.sin(x * frequency + phase + i * 1.2) *
              amplitude *
              Math.sin((x / width) * Math.PI);
          if (x === 0) this.canvasCtx.moveTo(x, y);
          else this.canvasCtx.lineTo(x, y);
        }
        this.canvasCtx.stroke();
      }

      phase += 0.04;
    };

    draw();
  }

  updateStateText(text) {
    if (this.agentStateText) this.agentStateText.innerText = text;
  }

  // =========================================================================
  // PDF Knowledge Base (RAG)
  // =========================================================================

  async handleFileUpload(file) {
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
      this.showUploadFeedback("error", "Please upload a valid PDF document.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    this.uploadProgressContainer.style.display = "block";
    this.uploadProgressBar.style.width = "40%";
    this.uploadProgressText.innerText = `Extracting & Chunking ${file.name}...`;

    try {
      const response = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });

      this.uploadProgressBar.style.width = "100%";
      const data = await response.json();

      if (response.ok && data.success) {
        this.showUploadFeedback(
          "success",
          `Saved "${file.name}" — ${data.data.total_chunks} paragraphs indexed and ready to answer.`,
        );
        this.loadDocuments();
        this.loadDatabaseData();
      } else {
        this.showUploadFeedback("error", data.detail || "Failed to index PDF.");
      }
    } catch (err) {
      console.error("Upload error:", err);
      this.showUploadFeedback("error", "Network error during upload.");
    } finally {
      setTimeout(() => {
        this.uploadProgressContainer.style.display = "none";
        this.uploadProgressBar.style.width = "0%";
      }, 2000);
    }
  }

  showUploadFeedback(type, message) {
    this.uploadFeedback.className = `upload-feedback ${type}`;
    this.uploadFeedback.innerText = message;
    this.uploadFeedback.style.display = "block";
    setTimeout(() => {
      this.uploadFeedback.style.display = "none";
    }, 5000);
  }

  async loadDocuments() {
    if (!this.documentsTableBody) return;
    try {
      const res = await fetch("/api/documents");
      const data = await res.json();
      const docs = data.documents || [];

      if (docs.length === 0) {
        this.documentsTableBody.innerHTML = `
          <tr>
            <td colspan="5" class="loading-cell">No PDF documents indexed yet. Upload a university prospectus above!</td>
          </tr>
        `;
        return;
      }

      this.documentsTableBody.innerHTML = docs
        .map(
          (d) => `
        <tr>
          <td><strong>${this.escapeHtml(d.filename || "")}</strong><br/><small class="muted">${this.escapeHtml(d.category || "Official document")}</small></td>
          <td>${this.escapeHtml(d.upload_time || "—")}</td>
          <td>${d.max_page || 1} pages</td>
          <td><span class="badge">${d.total_chunks || 0} chunks</span></td>
          <td>
            <div style="display: flex; gap: 6px; align-items: center;">
              <button class="btn-inspect btn-sm" data-doc-id="${this.escapeHtml(d.doc_id || "")}" data-filename="${this.escapeHtml(d.filename || "")}" type="button" style="padding: 4px 10px; font-size: 0.8rem; border-radius: 6px; border: 1px solid var(--border); background: var(--surface); cursor: pointer;">Inspect</button>
              <button class="btn-delete btn-sm" data-doc-id="${this.escapeHtml(d.doc_id || "")}" data-filename="${this.escapeHtml(d.filename || "")}" type="button" style="padding: 4px 10px; font-size: 0.8rem; border-radius: 6px; border: 1px solid #fca5a5; background: #fee2e2; color: #b91c1c; cursor: pointer; font-weight: 500;">Delete</button>
            </div>
          </td>
        </tr>
      `,
        )
        .join("");
    } catch (e) {
      console.error("Error loading docs:", e);
    }
  }

  openDeleteModal(docId, filename) {
    if (!docId) return;
    this.pendingDeleteDocId = docId;
    this.pendingDeleteDocName = filename || docId;
    if (this.deleteDocName) {
      this.deleteDocName.innerText = `"${this.pendingDeleteDocName}"`;
    }
    if (this.deleteDocModal) {
      this.deleteDocModal.style.display = "flex";
    }
  }

  closeDeleteModal() {
    this.pendingDeleteDocId = null;
    this.pendingDeleteDocName = null;
    if (this.deleteDocModal) {
      this.deleteDocModal.style.display = "none";
    }
  }

  async confirmDeleteDocument() {
    if (!this.pendingDeleteDocId) return;
    const docId = this.pendingDeleteDocId;
    const filename = this.pendingDeleteDocName || docId;

    if (this.confirmDeleteBtn) {
      this.confirmDeleteBtn.disabled = true;
      this.confirmDeleteBtn.innerText = "Deleting…";
    }

    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.success) {
        this.closeDeleteModal();
        this.showUploadFeedback(`Deleted "${filename}" from vector memory.`, "success");
        await this.loadDocuments();
        await this.loadDatabaseData();
        await this.checkSystemStatus();
      } else {
        alert(data.detail || data.message || "Failed to delete document from vector store.");
      }
    } catch (e) {
      console.error("Delete doc error:", e);
      alert("Error deleting document: " + e.message);
    } finally {
      if (this.confirmDeleteBtn) {
        this.confirmDeleteBtn.disabled = false;
        this.confirmDeleteBtn.innerText = "Delete permanently";
      }
    }
  }

  deleteDoc(docId, filename) {
    this.openDeleteModal(docId, filename);
  }

  async inspectChunks(docId, filename) {
    if (!this.chunkInspectorModal) return;
    this.chunkInspectorModal.style.display = "flex";
    if (this.inspectorDocTitle) {
      this.inspectorDocTitle.innerText = filename || `Document ${docId}`;
    }
    const subtitle = document.getElementById("inspectorDocSubtitle");
    if (subtitle) {
      subtitle.innerText = `Viewing all indexed chunks for ${filename || docId}`;
    }
    if (this.inspectorChunksList) {
      this.inspectorChunksList.innerHTML = `<div class="muted" style="padding: 24px; text-align: center;">Loading indexed paragraphs & chunks…</div>`;
    }

    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(docId)}/chunks`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const chunks = data.chunks || [];

      if (!this.inspectorChunksList) return;

      if (chunks.length === 0) {
        this.inspectorChunksList.innerHTML = `<div class="muted" style="padding: 24px; text-align: center;">No chunks found for this document.</div>`;
        return;
      }

      this.inspectorChunksList.innerHTML = `
        <div style="margin-bottom: 12px; font-size: 0.85rem; color: var(--muted); display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 8px;">
          <span>Total Chunks: <strong style="color: var(--ink);">${chunks.length}</strong></span>
          <span class="mono" style="font-size: 0.75rem; background: var(--surface); padding: 2px 8px; border-radius: 4px; border: 1px solid var(--border);">${this.escapeHtml(docId)}</span>
        </div>
        <div style="display: flex; flex-direction: column; gap: 10px; max-height: 60vh; overflow-y: auto; padding-right: 4px;">
          ${chunks
            .map(
              (c, idx) => `
            <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 0.84rem; line-height: 1.5; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.76rem; color: var(--muted); border-bottom: 1px dashed var(--border); padding-bottom: 4px;">
                <span style="font-weight: 600; color: var(--primary, #3b82f6);">Chunk #${c.chunk_index || idx + 1} &bull; Page ${c.page || 1}</span>
                <span class="mono" style="font-size: 0.72rem; opacity: 0.8;">${this.escapeHtml(c.chunk_id || "")}</span>
              </div>
              <div style="color: var(--ink); white-space: pre-wrap; word-break: break-word;">${this.escapeHtml(c.text || "")}</div>
            </div>
          `
            )
            .join("")}
        </div>
      `;
    } catch (err) {
      console.error("[inspectChunks] error:", err);
      if (this.inspectorChunksList) {
        this.inspectorChunksList.innerHTML = `<div style="color: var(--red-tx); padding: 24px; text-align: center;">Failed to load document chunks: ${this.escapeHtml(err.message)}</div>`;
      }
    }
  }

  async executeRagSearch() {
    const query = this.ragSearchInput.value.trim();
    if (!query) return;

    this.ragSearchResults.innerHTML = `<div class="empty-state-text">Searching ChromaDB embeddings...</div>`;

    try {
      const res = await fetch("/api/documents/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, n_results: 3 }),
      });
      const data = await res.json();
      const results = data.results || [];

      if (results.length === 0) {
        this.ragSearchResults.innerHTML = `<div class="empty-state-text">No matching chunks found. Upload more PDFs or try different keywords.</div>`;
        return;
      }

      this.ragSearchResults.innerHTML = results
        .map(
          (r, i) => `
        <div class="chunk-card">
          <div class="chunk-header">
            <span>Match #${i + 1} (${r.metadata.filename || "Doc"} - Page ${r.metadata.page || 1})</span>
            <span class="score-badge">${Math.round(r.similarity_score * 100)}% Match</span>
          </div>
          <p>${r.text}</p>
        </div>
      `,
        )
        .join("");
    } catch (e) {
      console.error("RAG search error:", e);
    }
  }

  // =========================================================================
  // Database Explorer & CRUD
  // =========================================================================

  async loadDatabaseData() {
    try {
      const dashRes = await fetch("/api/data/dashboard");
      const dashData = await dashRes.json();
      document.getElementById("metricStudents").innerText =
        dashData.total_students || 0;
      document.getElementById("metricPrograms").innerText =
        dashData.total_programs || 0;
      document.getElementById("metricPlacements").innerText =
        dashData.placement_records || 0;
      document.getElementById("metricChunks").innerText =
        dashData.total_chunks || 0;

      // Students
      const stuRes = await fetch("/api/data/students");
      const stuData = await stuRes.json();
      const studentsTableBody = document.getElementById("studentsTableBody");
      if (studentsTableBody && stuData.students) {
        studentsTableBody.innerHTML = stuData.students
          .map((s) => {
            const marksStr = (s.marks || [])
              .map(
                (m) =>
                  `${m.subject}: ${m.marks_obtained}/100 (<span class="grade-badge">${m.grade}</span>)`,
              )
              .join("<br/>");
            return `
            <tr>
              <td><strong>${s.student_id}</strong></td>
              <td>${s.name}</td>
              <td>${s.department_name}</td>
              <td>Sem ${s.semester}</td>
              <td><small>${marksStr || "No marks recorded"}</small></td>
            </tr>
          `;
          })
          .join("");
      }

      // Placements
      const plcRes = await fetch("/api/data/placements");
      const plcData = await plcRes.json();
      const plcTableBody = document.getElementById("placementsTableBody");
      if (plcTableBody && plcData.placements) {
        plcTableBody.innerHTML = plcData.placements
          .map(
            (p) => `
          <tr>
            <td><strong>${p.department_name}</strong></td>
            <td>₹${p.highest_package_lpa} LPA</td>
            <td>₹${p.average_package_lpa} LPA</td>
            <td><span class="badge">${p.placement_rate_pct}%</span></td>
          </tr>
        `,
          )
          .join("");
      }

      // Admissions
      const admRes = await fetch("/api/data/admissions");
      const admData = await admRes.json();
      const admTableBody = document.getElementById("admissionsTableBody");
      if (admTableBody && admData.admissions) {
        admTableBody.innerHTML = admData.admissions
          .map(
            (a) => `
          <tr>
            <td><strong>${a.program}</strong></td>
            <td>${a.fee_per_year}</td>
            <td>${a.last_date_to_apply}</td>
          </tr>
        `,
          )
          .join("");
      }
    } catch (e) {
      console.error("DB load error:", e);
    }
  }

  async handleAddStudentSubmit(e) {
    e.preventDefault();
    const studentId = document.getElementById("newStuId").value.trim();
    const name = document.getElementById("newStuName").value.trim();
    const dept = document.getElementById("newStuDept").value;
    const sem = parseInt(document.getElementById("newStuSem").value, 10);
    const phone = document.getElementById("newStuPhone").value.trim();
    const subject =
      document.getElementById("newStuSubject").value.trim() ||
      "Advanced Computing";
    const marks =
      parseInt(document.getElementById("newStuMarks").value, 10) || 90;

    const payload = {
      student_id: studentId,
      name: name,
      department_id: dept,
      semester: sem,
      parent_phone: phone,
      marks: [
        {
          subject: subject,
          marks_obtained: marks,
          max_marks: 100,
          grade: marks >= 90 ? "A+" : "A",
        },
      ],
      attendance: [
        { subject: subject, total_classes: 40, classes_attended: 38 },
      ],
    };

    try {
      const res = await fetch("/api/data/students", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        alert(
          `Saved ${name} (${studentId}). You can now ask about ${name} by voice or text.`,
        );
        this.addStudentModal.style.display = "none";
        this.addStudentForm.reset();
        this.loadDatabaseData();
      } else {
        alert(data.detail || "Failed to add student.");
      }
    } catch (err) {
      console.error("Add student error:", err);
      alert("Network error.");
    }
  }

  // =========================================================================
  // System Diagnostics
  // =========================================================================

  async checkSystemStatus() {
    try {
      const res = await fetch("/api/system/status");
      const data = await res.json();

      // LLM
      const llmPill = document.getElementById("diagLlmStatus");
      const llmModel = document.getElementById("diagLlmModel");
      const llmProvider = document.getElementById("diagLlmProvider");
      if (llmPill) {
        const ok = data.llm.status === "ready";
        llmPill.innerText = data.llm.status.toUpperCase();
        llmPill.className = "pill";
        llmPill.style.background = ok ? "#EDF3EC" : "#FDEBEC";
        llmPill.style.color = ok ? "#346538" : "#9F2F2D";
      }
      if (llmModel) llmModel.innerText = data.llm.model;
      if (llmProvider && data.llm.provider)
        llmProvider.innerText = data.llm.provider;

      // Sarvam
      const sarvamPill = document.getElementById("diagSarvamStatus");
      if (sarvamPill) {
        const ok = !!data.sarvam_ai.configured;
        sarvamPill.innerText = ok ? "CONFIGURED" : "MISSING KEY";
        sarvamPill.className = "pill";
        sarvamPill.style.background = ok ? "#EDF3EC" : "#FDEBEC";
        sarvamPill.style.color = ok ? "#346538" : "#9F2F2D";
      }

      // DB
      const dbMode = document.getElementById("diagDbMode");
      if (dbMode) dbMode.innerText = data.database.mode;

      // RAG
      const ragChunks = document.getElementById("diagRagChunks");
      if (ragChunks)
        ragChunks.innerText = `${data.vector_store.total_chunks} chunks (${data.vector_store.indexed_documents} docs)`;
    } catch (e) {
      console.error("Status check error:", e);
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.voiceApp = new VoiceAgentApp();
});
