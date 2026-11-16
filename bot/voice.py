"""AI voice-over (male). Default: Piper (free, runs offline on the GitHub runner). Optional: OpenAI TTS."""
import os, subprocess, wave
from pathlib import Path
import numpy as np
import requests
from .config import TTS_ENGINE, PIPER_VOICE, MODELS, OPENAI_VOICE, OPENAI_TTS_MODEL

SR = 44100


def _wav_to_array(path):
    """Load any wav, return mono float32 at 44.1 kHz (resampled with ffmpeg)."""
    out = str(path) + ".44k.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), out], check=True)
    with wave.open(out, "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return a


class Voice:
    def __init__(self, workdir):
        self.dir = Path(workdir); self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.engine = TTS_ENGINE
        self._piper = None
        if self.engine == "piper":
            from piper import PiperVoice
            model = MODELS / f"{PIPER_VOICE}.onnx"
            if not model.exists():
                raise FileNotFoundError(f"Piper voice model missing: {model} (run scripts/fetch_assets.sh)")
            self._piper = PiperVoice.load(str(model))

    def _piper_wav(self, text, path):
        v = self._piper
        with wave.open(str(path), "wb") as wf:
            if hasattr(v, "synthesize_wav"):                  # piper-tts >= 1.3
                try:
                    from piper import SynthesisConfig
                    v.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=0.9))
                except ImportError:
                    v.synthesize_wav(text, wf)
            else:                                              # piper-tts 1.2.x
                v.synthesize(text, wf, length_scale=0.9)

    def _openai_wav(self, text, path):
        r = requests.post("https://api.openai.com/v1/audio/speech",
                          headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                          json=dict(model=OPENAI_TTS_MODEL, voice=OPENAI_VOICE, input=text, response_format="wav",
                                    instructions="Energetic, friendly male game-show host. Short and punchy."),
                          timeout=120)
        r.raise_for_status()
        Path(path).write_bytes(r.content)

    def say(self, text, max_len=None):
        """Synthesize one line. Returns mono float32 array at 44.1 kHz (sped up slightly if it must fit max_len)."""
        self.n += 1
        raw = self.dir / f"line{self.n:02d}.wav"
        if self.engine == "openai":
            self._openai_wav(text, raw)
        else:
            self._piper_wav(text, raw)
        a = _wav_to_array(raw)
        if max_len and len(a) / SR > max_len:
            tempo = min(len(a) / SR / max_len, 1.35)
            fast = self.dir / f"line{self.n:02d}_fast.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-filter:a", f"atempo={tempo:.3f}",
                            "-ac", "1", "-ar", str(SR), str(fast)], check=True)
            a = _wav_to_array(fast)
        return a
