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
    this.sumSquares = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];

    // Accumulate samples and sum of squares into buffer
    for (let i = 0; i < channelData.length; i++) {
      const s = channelData[i];
      this.buffer[this.bufferIndex++] = s;
      this.sumSquares += s * s;

      if (this.bufferIndex >= this.bufferSize) {
        const bufferRms = Math.sqrt(this.sumSquares / this.bufferSize);
        this.port.postMessage({
          event: "audio_data",
          buffer: this.buffer.slice(0),
          rms: bufferRms,
        });
        this.bufferIndex = 0;
        this.sumSquares = 0;
      }
    }

    return true;
  }
}

registerProcessor("voice-agent-audio-processor", VoiceAgentAudioProcessor);
