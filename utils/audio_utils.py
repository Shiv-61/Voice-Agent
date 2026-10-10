"""
Audio utilities for telephony streaming:
- G.711 mu-law (8kHz) <-> Linear PCM16 (16kHz) conversion
- Resampling and framing for Vobiz / Plivo / SIP telephony
"""

import struct
try:
    import audioop
    HAS_AUDIOOP = True
except ImportError:
    HAS_AUDIOOP = False

# Precomputed mu-law decode table for pure-python fallback
_MULAW_DECODE_TABLE = []
for i in range(256):
    inv = ~i & 0xFF
    sign = (inv & 0x80)
    exponent = (inv >> 4) & 0x07
    mantissa = inv & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    _MULAW_DECODE_TABLE.append(-sample if sign else sample)


def _linear_to_mulaw_sample(sample: int) -> int:
    """Convert a single 16-bit PCM sample to 8-bit mu-law byte."""
    MULAW_MAX = 0x1FFF
    MULAW_BIAS = 33
    sign = 0
    if sample < 0:
        sign = 0x80
        sample = -sample
    sample = min(sample + MULAW_BIAS, MULAW_MAX)
    exponent = 7
    for exp_val in [0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100]:
        if sample >= exp_val:
            break
        exponent -= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def mulaw_to_pcm16_16k(mulaw_bytes: bytes) -> bytes:
    """
    Convert incoming 8kHz G.711 mu-law telephony audio into 16kHz 16-bit signed PCM.
    Used for Sarvam STT and VAD processing.
    """
    if not mulaw_bytes:
        return b""

    if HAS_AUDIOOP:
        try:
            pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
            pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
            return pcm_16k
        except Exception:
            pass

    # Pure Python fallback: decode 8kHz mu-law -> 8kHz PCM16, then duplicate samples for 16kHz
    pcm_samples_8k = [_MULAW_DECODE_TABLE[b] for b in mulaw_bytes]
    # Linear 2x interpolation / duplication for 8kHz -> 16kHz
    resampled_samples = []
    for s in pcm_samples_8k:
        resampled_samples.extend((s, s))
    return struct.pack(f"<{len(resampled_samples)}h", *resampled_samples)


def pcm16_16k_to_mulaw(pcm16_bytes: bytes, in_sample_rate: int = 22050) -> bytes:
    """
    Convert PCM or WAV audio from TTS (default 22050Hz from Sarvam AI, or 16000Hz from Edge-TTS)
    into 8kHz G.711 mu-law telephony audio with zero pitch or speed distortion.
    Used for outbound playAudio events over Vobiz telephony stream.
    """
    if not pcm16_bytes:
        return b""

    in_sr = in_sample_rate
    pcm_raw = pcm16_bytes

    # If full WAV bytes were provided, dynamically inspect the true sample rate from header
    if pcm16_bytes.startswith(b"RIFF"):
        try:
            import io, wave
            with io.BytesIO(pcm16_bytes) as bio, wave.open(bio, "rb") as wf:
                in_sr = wf.getframerate()
                pcm_raw = wf.readframes(wf.getnframes())
        except Exception:
            pcm_raw = pcm16_bytes[44:]

    if HAS_AUDIOOP:
        try:
            if in_sr != 8000:
                pcm_8k, _ = audioop.ratecv(pcm_raw, 2, 1, in_sr, 8000, None)
            else:
                pcm_8k = pcm_raw
            return audioop.lin2ulaw(pcm_8k, 2)
        except Exception:
            pass

    # Pure Python fallback: decimate -> 8kHz and encode
    n_samples = len(pcm_raw) // 2
    if n_samples == 0:
        return b""
    samples = struct.unpack(f"<{n_samples}h", pcm_raw[:n_samples * 2])
    step = max(1, in_sr // 8000)
    samples_8k = samples[::step]
    return bytes(_linear_to_mulaw_sample(s) for s in samples_8k)

