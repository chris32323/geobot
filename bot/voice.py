"""AI voice-over (male). Default: Piper (free, runs offline on the GitHub runner). Optional: OpenAI TTS."""
import os, re, subprocess, wave
from pathlib import Path
import numpy as np
import requests
from .config import (TTS_ENGINE, PIPER_VOICE, MODELS, OPENAI_VOICE, OPENAI_TTS_MODEL, VOICE_PITCH, VOICE_SPEED,
                     VOICE_EXPRESSION, VOICE_PAUSE, KOKORO_VOICE, KOKORO_SPEED)

SR = 44100


def _wav_to_array(path, tempo=1.0, pitch=1.0):
    """Load any wav -> mono float32 at 44.1 kHz. pitch<1 = deeper (same speed), tempo>1 = faster."""
    out = str(path) + ".44k.wav"
    filters = []
    if abs(pitch - 1.0) > 1e-3:
        filters += [f"asetrate={int(SR * pitch)}", f"aresample={SR}", f"atempo={1 / pitch:.4f}"]
    if abs(tempo - 1.0) > 1e-3:
        filters.append(f"atempo={tempo:.4f}")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR)]
    if filters:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-af", f"aresample={SR}," + ",".join(filters), "-ac", "1", "-ar", str(SR)]
    subprocess.run(cmd + [out], check=True)
    with wave.open(out, "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return a


class Voice:
    def __init__(self, workdir):
        self.dir = Path(workdir); self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.engine = TTS_ENGINE
        self._piper = None
        self._kokoro = None
        if self.engine == "kokoro":
            try:
                from kokoro_onnx import Kokoro
                self._kokoro = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
                print(f"  voice: Kokoro, {KOKORO_VOICE}, speed {KOKORO_SPEED}")
            except Exception as e:
                print(f"WARNING: Kokoro could not start ({e}) - falling back to Piper")
                self.engine = "piper"
        if self.engine == "piper":
            from piper import PiperVoice
            model = MODELS / f"{PIPER_VOICE}.onnx"
            if not model.exists():
                raise FileNotFoundError(f"Piper voice model missing: {model} (run scripts/fetch_assets.sh)")
            self._piper = PiperVoice.load(str(model))
            print(f"  voice: Piper, {PIPER_VOICE}")

    def _piper_wav(self, text, path):
        v = self._piper
        with wave.open(str(path), "wb") as wf:
            if hasattr(v, "synthesize_wav"):                  # piper-tts >= 1.3
                try:
                    from piper import SynthesisConfig
                    cfg = SynthesisConfig(length_scale=VOICE_SPEED, noise_scale=VOICE_EXPRESSION)
                except (ImportError, TypeError):
                    cfg = None
                if cfg is not None:
                    v.synthesize_wav(text, wf, syn_config=cfg)
                else:
                    v.synthesize_wav(text, wf)
            else:                                              # piper-tts 1.2.x
                v.synthesize(text, wf, length_scale=VOICE_SPEED, noise_scale=VOICE_EXPRESSION)

    def _kokoro_wav(self, text, path):
        lang = "en-gb" if KOKORO_VOICE.startswith("b") else "en-us"
        samples, sr = self._kokoro.create(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang=lang)
        pcm = (np.clip(np.asarray(samples, np.float32), -1, 1) * 32000).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(int(sr))
            wf.writeframes(pcm.tobytes())

    def _openai_wav(self, text, path):
        r = requests.post("https://api.openai.com/v1/audio/speech",
                          headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                          json=dict(model=OPENAI_TTS_MODEL, voice=OPENAI_VOICE, input=text, response_format="wav",
                                    instructions="Energetic, friendly male game-show host. Short and punchy."),
                          timeout=120)
        r.raise_for_status()
        Path(path).write_bytes(r.content)

    def _one(self, text):
        self.n += 1
        raw = self.dir / f"line{self.n:02d}.wav"
        if self.engine == "openai":
            self._openai_wav(text, raw)
        elif self.engine == "kokoro":
            self._kokoro_wav(text, raw)
        else:
            self._piper_wav(text, raw)
        a = _wav_to_array(raw, pitch=VOICE_PITCH)
        nz = np.flatnonzero(np.abs(a) > 0.01)                # trim leading/trailing silence
        return a[max(nz[0] - 800, 0): nz[-1] + 1600] if len(nz) else a

    def say(self, text, max_len=None):
        """Speak a line calmly: each sentence separately, with a short pause in between. Never sped up."""
        parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
        gap = np.zeros(int(VOICE_PAUSE * SR), np.float32)
        out = []
        for i, p in enumerate(parts):
            if i: out.append(gap)
            out.append(self._one(p))
        return np.concatenate(out) if out else np.zeros(1, np.float32)
