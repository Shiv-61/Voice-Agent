/**
 * AudioWorkletProcessor for Voice Agent.
 * Runs on the dedicated audio rendering thread for zero-latency, glitch-free audio capture.
 * Transmits 4096-sample audio buffers and instantaneous RMS energy metrics to the main thread.
 */

class VoiceAgentAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 4096;
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];

    // Calculate instantaneous RMS energy of the current 128-sample render quantum
    let sum = 0;
    for (let i = 0; i < channelData.length; i++) {
      sum += channelData[i] * channelData[i];
    }
    const rms = Math.sqrt(sum / channelData.length);

    // Accumulate samples into buffer and transmit once full
    for (let i = 0; i < channelData.length; i++) {
      this.buffer[this.bufferIndex++] = channelData[i];
      if (this.bufferIndex >= this.bufferSize) {
        this.port.postMessage({
          event: "audio_data",
          buffer: this.buffer.slice(0),
          rms: rms,
        });
        this.bufferIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor("voice-agent-audio-processor", VoiceAgentAudioProcessor);
