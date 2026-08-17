/**
 * University Voice Agent & Document Intelligence Web Application
 * Handles WebAudio mic streaming, WebSocket communication, Canvas Waveform Visualizer,
 * PDF RAG uploads, and Database inspection.
 */

class VoiceAgentApp {
  constructor() {
    this.ws = null;
    this.audioContext = null;
    this.mediaStream = null;
    this.audioInputNode = null;
    this.analyserNode = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.isCallActive = false;
    this.callStartTime = null;
    this.callTimerInterval = null;

    // Audio Playback Queue
    this.playbackQueue = [];
    this.isPlayingAudio = false;

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

    // Start background canvas wave animation
    this.startCanvasAnimation();

    // Periodic diagnostics check
    setInterval(() => this.checkSystemStatus(), 10000);
  }

  initElements() {
    // Tabs
    this.tabButtons = document.querySelectorAll(".tab-btn");
    this.tabPanes = document.querySelectorAll(".tab-pane");

    // Controls
    this.startCallBtn = document.getElementById("startCallBtn");
    this.pushToTalkBtn = document.getElementById("pushToTalkBtn");
    this.micOrb = document.getElementById("micOrb");
    this.agentStateText = document.getElementById("agentStateText");
    this.messagesContainer = document.getElementById("messagesContainer");
    this.clearChatBtn = document.getElementById("clearChatBtn");
    this.languageSelect = document.getElementById("languageSelect");
    this.wsStatusBadge = document.getElementById("wsStatusBadge");
    this.callDuration = document.getElementById("callDuration");

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

    // Suggested Chips
    this.chipButtons = document.querySelectorAll(".chip-btn");
  }

  initEventListeners() {
    // Tab switching
    this.tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        this.switchTab(targetTab);
      });
    });

    // Language Change
    this.languageSelect.addEventListener("change", (e) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ event: "start", language_code: e.target.value }));
      }
    });

    // Start/End Voice Call Button
    this.startCallBtn.addEventListener("click", () => this.toggleVoiceCall());

    // Push to talk button
    this.pushToTalkBtn.addEventListener("mousedown", () => this.startPushToTalk());
    window.addEventListener("mouseup", () => this.stopPushToTalk());
    this.pushToTalkBtn.addEventListener("touchstart", (e) => {
      e.preventDefault();
      this.startPushToTalk();
    });
    window.addEventListener("touchend", () => this.stopPushToTalk());

    // Suggested Chips click
    this.chipButtons.forEach((chip) => {
      chip.addEventListener("click", () => {
        const query = chip.getAttribute("data-query");
        this.addMessageBubble("user", query);
        this.updateStateText(`Processing query: "${query}"`);
        // Synthesize direct query request if Ollama available
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.simulateTextInput(query);
        }
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

    // RAG Dropzone events
    this.pdfDropzone.addEventListener("click", () => this.pdfFileInput.click());
    this.pdfFileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        this.handleFileUpload(e.target.files[0]);
      }
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
      if (e.dataTransfer.files.length > 0) {
        this.handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    this.refreshDocsBtn.addEventListener("click", () => this.loadDocuments());

    // RAG Search Tester
    this.ragSearchBtn.addEventListener("click", () => this.executeRagSearch());
    this.ragSearchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.executeRagSearch();
    });
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
        // Binary WAV audio chunk from Sarvam TTS
        this.enqueueAudioChunk(event.data);
      } else {
        // JSON event
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
      this.wsStatusBadge.style.color = "var(--accent-emerald)";
      this.wsStatusBadge.style.borderColor = "rgba(16, 185, 129, 0.3)";
    } else if (status === "connecting") {
      this.wsStatusBadge.style.color = "var(--accent-amber)";
      this.wsStatusBadge.style.borderColor = "rgba(245, 158, 11, 0.3)";
    } else {
      this.wsStatusBadge.style.color = "var(--accent-rose)";
      this.wsStatusBadge.style.borderColor = "rgba(244, 63, 94, 0.3)";
    }
    const textSpan = this.wsStatusBadge.querySelector(".status-text");
    if (textSpan) textSpan.innerText = text;
  }

  handleServerEvent(msg) {
    if (msg.event === "user_transcript") {
      this.addMessageBubble("user", msg.text, msg.language);
      this.updateStateText(`Transcribed: "${msg.text}"`);
    } else if (msg.event === "agent_thinking") {
      this.updateStateText("Reasoning with Qwen 2.5 + Querying Tools...");
    } else if (msg.event === "agent_partial_text") {
      this.updateStateText(`Agent speaking: "${msg.text}"`);
    } else if (msg.event === "agent_done") {
      this.addMessageBubble("agent", msg.full_text, msg.language);
      this.updateStateText("Ready. Listening for next query...");
    } else if (msg.event === "empty_transcript") {
      this.updateStateText("Didn't catch that. Please speak again.");
    }
  }

  // =========================================================================
  // Audio Recording & Streaming (WebAudio API)
  // =========================================================================

  async toggleVoiceCall() {
    if (!this.isCallActive) {
      // Start call
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
        this.pushToTalkBtn.disabled = false;
        this.updateStateText("Call connected. Hold Push-to-Talk or speak into microphone.");

        // Start call timer
        this.callStartTime = Date.now();
        this.callTimerInterval = setInterval(() => {
          const diff = Math.floor((Date.now() - this.callStartTime) / 1000);
          const mins = String(Math.floor(diff / 60)).padStart(2, "0");
          const secs = String(diff % 60).padStart(2, "0");
          if (this.callDuration) this.callDuration.innerText = `${mins}:${secs}`;
        }, 1000);

      } catch (err) {
        console.error("Mic access denied:", err);
        alert("Microphone permission is required for voice calling.");
      }
    } else {
      // End call
      this.isCallActive = false;
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
      this.pushToTalkBtn.disabled = true;
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

  startPushToTalk() {
    if (!this.isCallActive || this.isRecording) return;
    this.isRecording = true;
    this.audioChunks = [];
    this.micOrb.classList.add("active");
    this.updateStateText("🎙️ Listening... Speak now.");

    // Create script processor to capture PCM samples
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
    }

    // Merge Float32 chunks into single WAV file
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
    if (this.processorNode) {
      this.processorNode.disconnect();
    }
  }

  encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (view, offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    /* RIFF identifier */
    writeString(view, 0, "RIFF");
    /* file length */
    view.setUint32(4, 36 + samples.length * 2, true);
    /* RIFF type */
    writeString(view, 8, "WAVE");
    /* format chunk identifier */
    writeString(view, 12, "fmt ");
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, 1, true);
    /* channel count */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, "data");
    /* data chunk length */
    view.setUint32(40, samples.length * 2, true);

    // Write PCM 16-bit
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
      return;
    }

    this.isPlayingAudio = true;
    this.micOrb.classList.add("agent-speaking");

    const chunk = this.playbackQueue.shift();
    try {
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      const audioBuffer = await this.audioContext.decodeAudioData(chunk.slice(0));
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.audioContext.destination);

      source.onended = () => {
        this.playNextAudioChunk();
      };
      source.start();
    } catch (e) {
      console.error("Playback decode error:", e);
      this.playNextAudioChunk();
    }
  }

  // =========================================================================
  // Canvas Animated Waveform Visualizer
  // =========================================================================

  startCanvasAnimation() {
    if (!this.canvas || !this.canvasCtx) return;

    let phase = 0;
    const draw = () => {
      this.animationFrameId = requestAnimationFrame(draw);
      const width = this.canvas.width;
      const height = this.canvas.height;
      this.canvasCtx.clearRect(0, 0, width, height);

      let amplitude = 15;
      let frequency = 0.02;

      if (this.isRecording) {
        amplitude = 45;
        frequency = 0.04;
      } else if (this.isPlayingAudio) {
        amplitude = 35;
        frequency = 0.035;
      }

      // Draw glowing sine waves
      for (let i = 0; i < 3; i++) {
        this.canvasCtx.beginPath();
        this.canvasCtx.lineWidth = 2.5 - i * 0.5;

        if (i === 0) this.canvasCtx.strokeStyle = "rgba(99, 102, 241, 0.4)";
        else if (i === 1) this.canvasCtx.strokeStyle = "rgba(6, 182, 212, 0.5)";
        else this.canvasCtx.strokeStyle = "rgba(168, 85, 247, 0.6)";

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

  // =========================================================================
  // UI Helpers & Messages
  // =========================================================================

  updateStateText(text) {
    if (this.agentStateText) this.agentStateText.innerText = text;
  }

  addMessageBubble(role, text, lang = "") {
    if (!this.messagesContainer || !text.trim()) return;

    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${role === "user" ? "user-bubble" : "agent-bubble"}`;

    const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const speaker = role === "user" ? `🧑 You (${lang || "Caller"})` : "🎓 UniVoice Agent";

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
            <button class="btn-delete" onclick="window.voiceApp.deleteDoc('${d.doc_id}')">Delete</button>
          </td>
        </tr>
      `).join("");
    } catch (e) {
      console.error("Error loading docs:", e);
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
  // Database Explorer
  // =========================================================================

  async loadDatabaseData() {
    try {
      // Dashboard KPIs
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
              <td><small>${marksStr || "No marks"}</small></td>
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

  // =========================================================================
  // System Diagnostics Check
  // =========================================================================

  async checkSystemStatus() {
    try {
      const res = await fetch("/api/system/status");
      const data = await res.json();

      // LLM
      const llmPill = document.getElementById("diagLlmStatus");
      const llmModel = document.getElementById("diagLlmModel");
      if (llmPill) {
        llmPill.innerText = data.llm.status.toUpperCase();
        llmPill.className = `status-pill ${data.llm.status === "ready" ? "pill-green" : "pill-red"}`;
      }
      if (llmModel) llmModel.innerText = data.llm.model;

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

// Global initialization
window.addEventListener("DOMContentLoaded", () => {
  window.voiceApp = new VoiceAgentApp();
});
