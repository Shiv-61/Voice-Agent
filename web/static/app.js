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

    // Auto-VAD threshold & timer state
    this.vadThreshold = 0.015;
    this.vadSilenceTimer = null;
    this.vadSilenceDuration = 1200; // ms silence before commit
    this.isSpeechDetected = false;

    // Audio Playback Queue
    this.playbackQueue = [];
    this.isPlayingAudio = false;
    this.currentAudioSource = null;

    // Canvas animation
    this.canvas = document.getElementById("waveformCanvas");
    this.canvasCtx = this.canvas ? this.canvas.getContext("2d") : null;
    this.animationFrameId = null;

    this.initElements();
    this.initEventListeners();
    this.initWebSocket();
    this.loadDocuments();
    this.loadDatabaseData();
    this.checkSystemStatus();

    this.startCanvasAnimation();
    setInterval(() => this.checkSystemStatus(), 10000);
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

    this.micOrb = document.getElementById("micOrb");
    this.agentStateText = document.getElementById("agentStateText");
    this.messagesContainer = document.getElementById("messagesContainer");
    this.clearChatBtn = document.getElementById("clearChatBtn");
    this.languageSelect = document.getElementById("languageSelect");
    this.wsStatusBadge = document.getElementById("wsStatusBadge");
    this.callDuration = document.getElementById("callDuration");

    // Dual Chat Input
    this.chatInputForm = document.getElementById("chatInputForm");
    this.chatTextInput = document.getElementById("chatTextInput");

    // Suggested Chips
    this.chipButtons = document.querySelectorAll(".chip-btn");

    // RAG Elements
    this.pdfDropzone = document.getElementById("pdfDropzone");
    this.pdfFileInput = document.getElementById("pdfFileInput");
    this.uploadProgressContainer = document.getElementById("uploadProgressContainer");
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
    this.openAddStudentModalBtn = document.getElementById("openAddStudentModalBtn");
    this.closeAddStudentBtn = document.getElementById("closeAddStudentBtn");
    this.cancelAddStudentBtn = document.getElementById("cancelAddStudentBtn");
    this.addStudentForm = document.getElementById("addStudentForm");
  }

  initEventListeners() {
    // Tabs
    this.tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        this.switchTab(btn.getAttribute("data-tab"));
      });
    });

    // Language Change
    this.languageSelect.addEventListener("change", (e) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ event: "start", language_code: e.target.value }));
      }
    });

    // Mode switch
    this.modeAutoVadBtn.addEventListener("click", () => this.setCallMode("auto_vad"));
    this.modePttBtn.addEventListener("click", () => this.setCallMode("ptt"));

    // Call Toggle
    this.startCallBtn.addEventListener("click", () => this.toggleVoiceCall());

    // Push to talk (PTT)
    this.pushToTalkBtn.addEventListener("mousedown", () => this.startPushToTalk());
    window.addEventListener("mouseup", () => this.stopPushToTalk());
    this.pushToTalkBtn.addEventListener("touchstart", (e) => {
      e.preventDefault();
      this.startPushToTalk();
    });
    window.addEventListener("touchend", () => this.stopPushToTalk());

    // Interrupt Button
    this.interruptBtn.addEventListener("click", () => this.interruptAgent());

    // Dual Chat Text Input Form
    this.chatInputForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const text = this.chatTextInput.value.trim();
      if (!text) return;
      this.chatTextInput.value = "";
      this.sendTextMessage(text);
    });

    // Suggested Chips click
    this.chipButtons.forEach((chip) => {
      chip.addEventListener("click", () => {
        const query = chip.getAttribute("data-query");
        this.sendTextMessage(query);
      });
    });

    // Clear chat
    this.clearChatBtn.addEventListener("click", () => {
      this.messagesContainer.innerHTML = `
        <div class="message-bubble agent-bubble greeting">
          <div class="bubble-header">
            <span class="speaker-name">🎓 UniVoice Assistant</span>
            <span class="speaker-time">Just now</span>
          </div>
          <div class="bubble-content">
            Chat cleared. Ready for your admission & academic questions!
          </div>
        </div>
      `;
    });

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
      if (e.dataTransfer.files.length > 0) this.handleFileUpload(e.dataTransfer.files[0]);
    });

    this.refreshDocsBtn.addEventListener("click", () => this.loadDocuments());

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
    this.closeInspectorBtn.addEventListener("click", () => {
      this.chunkInspectorModal.style.display = "none";
    });

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
    this.addStudentForm.addEventListener("submit", (e) => this.handleAddStudentSubmit(e));
  }

  setCallMode(mode) {
    this.callMode = mode;
    if (mode === "auto_vad") {
      this.modeAutoVadBtn.classList.add("active");
      this.modePttBtn.classList.remove("active");
      this.pushToTalkBtn.style.display = "none";
      if (this.isCallActive) this.updateStateText("🎙️ Hands-Free Mode: Start speaking whenever ready.");
    } else {
      this.modeAutoVadBtn.classList.remove("active");
      this.modePttBtn.classList.add("active");
      if (this.isCallActive) {
        this.pushToTalkBtn.style.display = "inline-flex";
        this.pushToTalkBtn.disabled = false;
        this.updateStateText("🔘 Push-to-Talk Mode: Hold the button to speak.");
      }
    }
  }

  switchTab(targetTab) {
    this.tabButtons.forEach((b) => b.classList.remove("active"));
    this.tabPanes.forEach((p) => p.classList.remove("active"));

    const activeBtn = document.querySelector(`.tab-btn[data-tab="${targetTab}"]`);
    const activePane = document.getElementById(`tab-${targetTab}`);

    if (activeBtn) activeBtn.classList.add("active");
    if (activePane) activePane.classList.add("active");

    if (targetTab === "database") this.loadDatabaseData();
    if (targetTab === "rag") this.loadDocuments();
    if (targetTab === "diagnostics") this.checkSystemStatus();
  }

  // =========================================================================
  // WebSocket Connection
  // =========================================================================

  initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/call`;

    this.setWsBadge("connecting", "Connecting...");
    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      this.setWsBadge("connected", "Online & Ready");
      this.ws.send(JSON.stringify({
        event: "start",
        language_code: this.languageSelect.value,
      }));
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
    this.wsStatusBadge.className = "status-indicator";
    if (status === "connected") {
      this.wsStatusBadge.style.color = "#059669";
      this.wsStatusBadge.style.background = "#edf8f2";
      this.wsStatusBadge.style.borderColor = "rgba(91, 185, 140, 0.25)";
    } else if (status === "connecting") {
      this.wsStatusBadge.style.color = "#d97706";
      this.wsStatusBadge.style.background = "#fef9ee";
      this.wsStatusBadge.style.borderColor = "rgba(245, 158, 11, 0.25)";
    } else {
      this.wsStatusBadge.style.color = "#dc2626";
      this.wsStatusBadge.style.background = "#fef2f2";
      this.wsStatusBadge.style.borderColor = "rgba(248, 113, 113, 0.25)";
    }
    const textSpan = this.wsStatusBadge.querySelector(".status-text");
    if (textSpan) textSpan.innerText = text;
  }

  handleServerEvent(msg) {
    if (msg.event === "user_transcript") {
      this.currentAgentBubbleContent = null;
      this.addMessageBubble("user", msg.text, msg.language);
      this.updateStateText(`Processing: "${msg.text}"`);
    } else if (msg.event === "agent_thinking") {
      this.currentAgentBubbleContent = null;
      this.updateStateText("Reasoning with NVIDIA Nemotron 3 Ultra...");
    } else if (msg.event === "tool_executed") {
      // Render glowing tool execution pill in chat
      this.addToolExecutionPill(msg.tool, msg.args, msg.preview);
    } else if (msg.event === "agent_partial_text") {
      this.appendAgentPartialText(msg.text, msg.language);
      this.updateStateText(`Agent speaking: "${msg.text.slice(0, 40)}..."`);
      this.interruptBtn.style.display = "inline-flex";
    } else if (msg.event === "agent_done") {
      this.finalizeAgentText(msg.full_text, msg.language);
      this.updateStateText("Ready. Listening for next question...");
    } else if (msg.event === "empty_transcript") {
      this.updateStateText("Didn't catch that. Please speak again.");
    } else if (msg.event === "error") {
      this.addErrorBubble(msg.message || "An error occurred during processing.");
      this.updateStateText("System Notice");
    }
  }

  appendAgentPartialText(text, lang = "") {
    if (!this.messagesContainer) return;
    if (!this.currentAgentBubbleContent) {
      const bubble = document.createElement("div");
      bubble.className = "message-bubble agent-bubble";
      const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      bubble.innerHTML = `
        <div class="bubble-header">
          <span class="speaker-name">🎓 UniVoice Assistant</span>
          <span class="speaker-time">${timeStr}</span>
        </div>
        <div class="bubble-content"></div>
      `;
      this.messagesContainer.appendChild(bubble);
      this.currentAgentBubbleContent = bubble.querySelector(".bubble-content");
    }
    if (this.currentAgentBubbleContent) {
      const existing = this.currentAgentBubbleContent.innerText;
      this.currentAgentBubbleContent.innerText = existing ? `${existing} ${text}` : text;
      this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
  }

  finalizeAgentText(fullText, lang = "") {
    if (this.currentAgentBubbleContent) {
      if (fullText) this.currentAgentBubbleContent.innerText = fullText;
      this.currentAgentBubbleContent = null;
    } else if (fullText) {
      this.addMessageBubble("agent", fullText, lang);
    }
  }

  addErrorBubble(message) {
    if (!this.messagesContainer) return;
    const bubble = document.createElement("div");
    bubble.className = "message-bubble error-bubble";
    const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    bubble.innerHTML = `
      <div class="bubble-header">
        <span class="speaker-name">⚠️ System Notice</span>
        <span class="speaker-time">${timeStr}</span>
      </div>
      <div class="bubble-content">${message}</div>
    `;
    this.messagesContainer.appendChild(bubble);
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  sendTextMessage(text) {
    if (!text.trim()) return;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.interruptAgent(); // Stop any previous speech
      this.ws.send(JSON.stringify({
        event: "text_query",
        text: text.trim(),
        language_code: this.languageSelect.value,
      }));
    }
  }

  addToolExecutionPill(toolName, args, preview) {
    if (!this.messagesContainer) return;
    const pill = document.createElement("div");
    pill.className = "tool-pill";
    const icon = toolName === "search_university_docs" ? "📄 RAG Vector Query" : "⚡ SQL Tool";
    pill.innerHTML = `<span>${icon}:</span> <code>${preview || toolName}</code>`;
    this.messagesContainer.appendChild(pill);
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }

  // =========================================================================
  // WebAudio Recording & Auto-VAD
  // =========================================================================

  async toggleVoiceCall() {
    if (!this.isCallActive) {
      try {
        await this.initAudioContext();
        this.isCallActive = true;
        this.startCallBtn.innerHTML = `
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
          <span>End Voice Call</span>
        `;
        this.startCallBtn.classList.replace("btn-primary", "btn-danger");

        if (this.callMode === "ptt") {
          this.pushToTalkBtn.style.display = "inline-flex";
          this.pushToTalkBtn.disabled = false;
          this.updateStateText("🔘 Hold button to speak.");
        } else {
          this.pushToTalkBtn.style.display = "none";
          this.startAutoVadListening();
          this.updateStateText("🎙️ Hands-Free Connected. Speak into microphone anytime.");
        }

        // Start timer
        this.callStartTime = Date.now();
        this.callTimerInterval = setInterval(() => {
          const diff = Math.floor((Date.now() - this.callStartTime) / 1000);
          const mins = String(Math.floor(diff / 60)).padStart(2, "0");
          const secs = String(diff % 60).padStart(2, "0");
          if (this.callDuration) this.callDuration.innerText = `${mins}:${secs}`;
        }, 1000);

      } catch (err) {
        console.error("Mic access denied:", err);
        alert("Microphone permission is required for live voice calls.");
      }
    } else {
      // End call
      this.isCallActive = false;
      this.stopAutoVadListening();
      this.stopRecording();
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach((track) => track.stop());
      }
      this.startCallBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
        </svg>
        <span>Start Voice Call</span>
      `;
      this.startCallBtn.classList.replace("btn-danger", "btn-primary");
      this.pushToTalkBtn.style.display = "none";
      this.interruptBtn.style.display = "none";
      this.updateStateText("Call ended.");
      clearInterval(this.callTimerInterval);
      if (this.callDuration) this.callDuration.innerText = "00:00";
    }
  }

  async initAudioContext() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    }
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }
    this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.audioInputNode = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.analyserNode = this.audioContext.createAnalyser();
    this.analyserNode.fftSize = 256;
    this.audioInputNode.connect(this.analyserNode);
  }

  // --- Auto-VAD Listening ---
  startAutoVadListening() {
    if (!this.isCallActive || this.processorNode) return;

    const bufferSize = 4096;
    this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
    this.audioChunks = [];
    this.isSpeechDetected = false;

    this.processorNode.onaudioprocess = (e) => {
      if (!this.isCallActive || this.callMode !== "auto_vad" || this.isPlayingAudio) return;

      const input = e.inputBuffer.getChannelData(0);
      // Calculate RMS energy
      let sum = 0;
      for (let i = 0; i < input.length; i++) {
        sum += input[i] * input[i];
      }
      const rms = Math.sqrt(sum / input.length);

      if (rms > this.vadThreshold) {
        // Speech detected
        if (!this.isSpeechDetected) {
          this.isSpeechDetected = true;
          this.micOrb.classList.add("active");
          this.updateStateText("🎙️ Listening... Speaking detected.");
        }
        this.audioChunks.push(new Float32Array(input));

        // Clear silence timeout
        if (this.vadSilenceTimer) {
          clearTimeout(this.vadSilenceTimer);
          this.vadSilenceTimer = null;
        }
      } else if (this.isSpeechDetected) {
        // Still buffer during small pauses
        this.audioChunks.push(new Float32Array(input));

        if (!this.vadSilenceTimer) {
          this.vadSilenceTimer = setTimeout(() => {
            this.commitAutoVadSpeech();
          }, this.vadSilenceDuration);
        }
      }
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
    this.micOrb.classList.remove("active");
    this.updateStateText("Transcribing voice with Sarvam AI...");

    const totalLength = this.audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
    if (totalLength < 3000) {
      this.audioChunks = [];
      this.updateStateText("🎙️ Listening... Speak whenever ready.");
      return;
    }

    const mergedBuffer = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.audioChunks) {
      mergedBuffer.set(chunk, offset);
      offset += chunk.length;
    }
    this.audioChunks = [];

    const wavBlob = this.encodeWAV(mergedBuffer, 16000);
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
    this.updateStateText("🎙️ Listening... Speak now.");

    const bufferSize = 4096;
    this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
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
    this.updateStateText("Transcribing voice with Sarvam AI...");

    if (this.processorNode) {
      this.processorNode.disconnect();
      this.audioInputNode.disconnect(this.processorNode);
      this.processorNode = null;
    }

    const totalLength = this.audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
    if (totalLength < 2000) {
      this.updateStateText("Too short. Please hold button and speak clearly.");
      return;
    }

    const mergedBuffer = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.audioChunks) {
      mergedBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    const wavBlob = this.encodeWAV(mergedBuffer, 16000);
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

  interruptAgent() {
    this.playbackQueue = [];
    this.isPlayingAudio = false;
    if (this.currentAudioSource) {
      try {
        this.currentAudioSource.stop();
      } catch (e) {}
      this.currentAudioSource = null;
    }
    this.micOrb.classList.remove("agent-speaking");
    this.interruptBtn.style.display = "none";
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ event: "interrupt" }));
    }
    this.updateStateText("Interrupted. Ready for next query.");
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
  // Audio Playback Queue
  // =========================================================================

  enqueueAudioChunk(arrayBuffer) {
    this.playbackQueue.push(arrayBuffer);
    if (!this.isPlayingAudio) {
      this.playNextAudioChunk();
    }
  }

  async playNextAudioChunk() {
    if (this.playbackQueue.length === 0) {
      this.isPlayingAudio = false;
      this.micOrb.classList.remove("agent-speaking");
      this.interruptBtn.style.display = "none";
      return;
    }

    this.isPlayingAudio = true;
    this.micOrb.classList.add("agent-speaking");
    this.interruptBtn.style.display = "inline-flex";

    const chunk = this.playbackQueue.shift();
    try {
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }
      const audioBuffer = await this.audioContext.decodeAudioData(chunk.slice(0));
      this.currentAudioSource = this.audioContext.createBufferSource();
      this.currentAudioSource.buffer = audioBuffer;
      this.currentAudioSource.connect(this.audioContext.destination);

      this.currentAudioSource.onended = () => {
        this.currentAudioSource = null;
        this.playNextAudioChunk();
      };
      this.currentAudioSource.start();
    } catch (e) {
      console.error("Playback decode error:", e);
      this.playNextAudioChunk();
    }
  }

  // =========================================================================
  // Canvas Visualizer
  // =========================================================================

  startCanvasAnimation() {
    if (!this.canvas || !this.canvasCtx) return;

    let phase = 0;
    const draw = () => {
      this.animationFrameId = requestAnimationFrame(draw);
      const width = this.canvas.width;
      const height = this.canvas.height;
      this.canvasCtx.clearRect(0, 0, width, height);

      let amplitude = 12;
      let frequency = 0.02;

      if (this.isSpeechDetected || this.isRecording) {
        amplitude = 45;
        frequency = 0.04;
      } else if (this.isPlayingAudio) {
        amplitude = 35;
        frequency = 0.035;
      }

      for (let i = 0; i < 3; i++) {
        this.canvasCtx.beginPath();
        this.canvasCtx.lineWidth = 2.5 - i * 0.5;

        if (i === 0) this.canvasCtx.strokeStyle = "rgba(124, 92, 252, 0.3)";
        else if (i === 1) this.canvasCtx.strokeStyle = "rgba(245, 158, 107, 0.35)";
        else this.canvasCtx.strokeStyle = "rgba(91, 185, 140, 0.4)";

        for (let x = 0; x < width; x++) {
          const y = height / 2 + Math.sin(x * frequency + phase + i * 1.2) * amplitude * Math.sin((x / width) * Math.PI);
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

  addMessageBubble(role, text, lang = "") {
    if (!this.messagesContainer || !text.trim()) return;

    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${role === "user" ? "user-bubble" : "agent-bubble"}`;

    const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const speaker = role === "user" ? `🧑 Caller (${lang || "User"})` : "🎓 UniVoice Agent";

    bubble.innerHTML = `
      <div class="bubble-header">
        <span class="speaker-name">${speaker}</span>
        <span class="speaker-time">${timeStr}</span>
      </div>
      <div class="bubble-content">${text}</div>
    `;

    this.messagesContainer.appendChild(bubble);
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
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
        this.showUploadFeedback("success", `✓ Successfully indexed "${file.name}" (${data.data.total_chunks} chunks).`);
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

      this.documentsTableBody.innerHTML = docs.map((d) => `
        <tr>
          <td><strong>📄 ${d.filename}</strong></td>
          <td>${d.upload_time}</td>
          <td>${d.max_page} pages</td>
          <td><span class="badge badge-accent">${d.total_chunks} chunks</span></td>
          <td>
            <button class="btn-inspect" onclick="window.voiceApp.inspectChunks('${d.doc_id}', '${d.filename}')">Inspect</button>
            <button class="btn-delete" onclick="window.voiceApp.deleteDoc('${d.doc_id}')">Delete</button>
          </td>
        </tr>
      `).join("");
    } catch (e) {
      console.error("Error loading docs:", e);
    }
  }

  async inspectChunks(docId, filename) {
    this.inspectorDocTitle.innerText = `📄 ${filename}`;
    this.inspectorChunksList.innerHTML = `<div class="loading-cell">Loading chunk vectors...</div>`;
    this.chunkInspectorModal.style.display = "flex";

    try {
      const res = await fetch(`/api/documents/${docId}/chunks`);
      const data = await res.json();
      const chunks = data.chunks || [];

      if (chunks.length === 0) {
        this.inspectorChunksList.innerHTML = `<div class="empty-state-text">No chunks found for this document.</div>`;
        return;
      }

      this.inspectorChunksList.innerHTML = chunks.map((c) => `
        <div class="chunk-card">
          <div class="chunk-header">
            <span>Page ${c.page} &bull; Chunk #${c.chunk_index}</span>
            <code style="font-size: 0.72rem; color: var(--text-dim);">${c.chunk_id}</code>
          </div>
          <p style="font-size: 0.85rem; line-height: 1.5; color: var(--text-main);">${c.text}</p>
        </div>
      `).join("");
    } catch (e) {
      console.error("Inspect chunks error:", e);
      this.inspectorChunksList.innerHTML = `<div class="upload-feedback error">Failed to load chunks.</div>`;
    }
  }

  async deleteDoc(docId) {
    if (!confirm("Are you sure you want to delete this document from the vector store?")) return;
    try {
      const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
      if (res.ok) {
        this.loadDocuments();
        this.loadDatabaseData();
      }
    } catch (e) {
      console.error("Delete doc error:", e);
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

      this.ragSearchResults.innerHTML = results.map((r, i) => `
        <div class="chunk-card">
          <div class="chunk-header">
            <span>Match #${i + 1} (${r.metadata.filename || "Doc"} - Page ${r.metadata.page || 1})</span>
            <span class="score-badge">${Math.round(r.similarity_score * 100)}% Match</span>
          </div>
          <p>${r.text}</p>
        </div>
      `).join("");
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
      document.getElementById("metricStudents").innerText = dashData.total_students || 0;
      document.getElementById("metricPrograms").innerText = dashData.total_programs || 0;
      document.getElementById("metricPlacements").innerText = dashData.placement_records || 0;
      document.getElementById("metricChunks").innerText = dashData.total_chunks || 0;

      // Students
      const stuRes = await fetch("/api/data/students");
      const stuData = await stuRes.json();
      const studentsTableBody = document.getElementById("studentsTableBody");
      if (studentsTableBody && stuData.students) {
        studentsTableBody.innerHTML = stuData.students.map((s) => {
          const marksStr = (s.marks || []).map((m) => `${m.subject}: ${m.marks_obtained}/100 (<span class="grade-badge">${m.grade}</span>)`).join("<br/>");
          return `
            <tr>
              <td><strong>${s.student_id}</strong></td>
              <td>${s.name}</td>
              <td>${s.department_name}</td>
              <td>Sem ${s.semester}</td>
              <td><small>${marksStr || "No marks recorded"}</small></td>
            </tr>
          `;
        }).join("");
      }

      // Placements
      const plcRes = await fetch("/api/data/placements");
      const plcData = await plcRes.json();
      const plcTableBody = document.getElementById("placementsTableBody");
      if (plcTableBody && plcData.placements) {
        plcTableBody.innerHTML = plcData.placements.map((p) => `
          <tr>
            <td><strong>${p.department_name}</strong></td>
            <td>₹${p.highest_package_lpa} LPA</td>
            <td>₹${p.average_package_lpa} LPA</td>
            <td><span class="badge badge-accent">${p.placement_rate_pct}%</span></td>
          </tr>
        `).join("");
      }

      // Admissions
      const admRes = await fetch("/api/data/admissions");
      const admData = await admRes.json();
      const admTableBody = document.getElementById("admissionsTableBody");
      if (admTableBody && admData.admissions) {
        admTableBody.innerHTML = admData.admissions.map((a) => `
          <tr>
            <td><strong>${a.program}</strong></td>
            <td>${a.fee_per_year}</td>
            <td>${a.last_date_to_apply}</td>
          </tr>
        `).join("");
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
    const subject = document.getElementById("newStuSubject").value.trim() || "Advanced Computing";
    const marks = parseInt(document.getElementById("newStuMarks").value, 10) || 90;

    const payload = {
      student_id: studentId,
      name: name,
      department_id: dept,
      semester: sem,
      parent_phone: phone,
      marks: [{ subject: subject, marks_obtained: marks, max_marks: 100, grade: marks >= 90 ? "A+" : "A" }],
      attendance: [{ subject: subject, total_classes: 40, classes_attended: 38 }],
    };

    try {
      const res = await fetch("/api/data/students", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        alert(`✓ Added ${name} (${studentId})! You can now ask the voice agent about ${name}.`);
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
        llmPill.innerText = data.llm.status.toUpperCase();
        llmPill.className = `status-pill ${data.llm.status === "ready" ? "pill-green" : "pill-red"}`;
      }
      if (llmModel) llmModel.innerText = data.llm.model;
      if (llmProvider && data.llm.provider) llmProvider.innerText = data.llm.provider;

      // Sarvam
      const sarvamPill = document.getElementById("diagSarvamStatus");
      if (sarvamPill) {
        sarvamPill.innerText = data.sarvam_ai.configured ? "CONFIGURED" : "MISSING KEY";
        sarvamPill.className = `status-pill ${data.sarvam_ai.configured ? "pill-green" : "pill-red"}`;
      }

      // DB
      const dbMode = document.getElementById("diagDbMode");
      if (dbMode) dbMode.innerText = data.database.mode;

      // RAG
      const ragChunks = document.getElementById("diagRagChunks");
      if (ragChunks) ragChunks.innerText = `${data.vector_store.total_chunks} chunks (${data.vector_store.indexed_documents} docs)`;

    } catch (e) {
      console.error("Status check error:", e);
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.voiceApp = new VoiceAgentApp();
});
