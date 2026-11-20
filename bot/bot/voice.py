"""Voice-over with ElevenLabs only (Eleven v4, Voice Library voice "Nick - Relatable & Engaging Friend").

Every sentence is recorded once and kept in data/voice/ (cache). After that the voice costs nothing and
ElevenLabs is not even contacted. There is NO other AI voice and NO silent fallback: if a sentence is missing
and ElevenLabs cannot make it, the run stops with a clear message instead of posting a reel with a wrong voice.
(Kokoro / Piper, free offline voices, are only used if TTS_ENGINE is explicitly set to kokoro or piper.)
"""
import hashlib, json as _json, os, re, subprocess, time, wave
from pathlib import Path
import numpy as np
import requests
from .safety import redact
from .config import (TTS_ENGINE, PIPER_VOICE, MODELS, VOICE_PITCH, VOICE_SPEED, VOICE_EXPRESSION, VOICE_PAUSE,
                     KOKORO_VOICE, KOKORO_SPEED, ELEVEN_MODEL, ELEVEN_VOICE_ID, ELEVEN_VOICE_NAME,
                     ELEVEN_STABILITY, ELEVEN_SIMILARITY, ELEVEN_FORMAT)

SR = 44100
ELEVEN_API = "https://api.elevenlabs.io"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "voice"
CACHE_VERSION = 1           # bump to re-record every line


class VoiceError(RuntimeError):
    """A sentence is missing in the cache and ElevenLabs cannot record it - stop instead of using another voice."""


def sentences(text):
    """The bot speaks (and caches) sentence by sentence."""
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]


def _wav_to_array(path, pitch=1.0):
    """Load any audio file (wav/mp3/flac) -> mono float32 at 44.1 kHz. pitch<1 = deeper (same speed)."""
    out = str(path) + ".44k.wav"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path)]
    if abs(pitch - 1.0) > 1e-3:
        cmd += ["-af", f"aresample={SR},asetrate={int(SR * pitch)},aresample={SR},atempo={1 / pitch:.4f}"]
    subprocess.run(cmd + ["-ac", "1", "-ar", str(SR), out], check=True)
    with wave.open(out, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


class Voice:
    def __init__(self, workdir):
        self.dir = Path(workdir); self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0; self.hits = 0; self.misses = 0; self.chars = 0
        self._piper = None; self._kokoro = None
        self._ready = False; self.eleven_id = ELEVEN_VOICE_ID or None
        engine = (TTS_ENGINE or "elevenlabs").strip().lower()
        if engine not in ("elevenlabs", "kokoro", "piper"):
            print(f"  voice: TTS_ENGINE='{engine}' is not supported any more - using ElevenLabs")
            engine = "elevenlabs"
        self.engine = engine
        if engine == "elevenlabs":
            print(f"  voice: ElevenLabs, {ELEVEN_VOICE_ID or ELEVEN_VOICE_NAME} ({ELEVEN_MODEL}, stability "
                  f"{ELEVEN_STABILITY:g}, similarity {ELEVEN_SIMILARITY:g}), cache {CACHE_DIR.name}/")
        elif engine == "kokoro":
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
            print(f"  voice: Kokoro, {KOKORO_VOICE}, speed {KOKORO_SPEED}")
        else:
            from piper import PiperVoice
            model = MODELS / f"{PIPER_VOICE}.onnx"
            if not model.exists():
                raise FileNotFoundError(f"Piper voice model missing: {model} (run scripts/fetch_assets.sh)")
            self._piper = PiperVoice.load(str(model))
            print(f"  voice: Piper, {PIPER_VOICE}")

    # ------------------------------------------------------------------ ElevenLabs
    def _xi(self):
        key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        if not key:
            raise VoiceError("the secret ELEVENLABS_API_KEY is missing (GitHub: Settings -> Secrets and variables "
                             "-> Actions -> New repository secret)")
        return {"xi-api-key": key}

    def _find_voice(self):
        """Voice ID from the ELEVEN_VOICE_ID variable, otherwise by name in your ElevenLabs voices."""
        if self.eleven_id:
            return self.eleven_id
        name = ELEVEN_VOICE_NAME.lower()
        r = requests.get(f"{ELEVEN_API}/v2/voices", headers=self._xi(), timeout=60,
                         params=dict(search=ELEVEN_VOICE_NAME.split(" - ")[0], page_size=100))
        if r.status_code != 200:
            raise VoiceError(f"could not list your ElevenLabs voices (HTTP {r.status_code}: {r.text[:200]}) - "
                             f"set the variable ELEVEN_VOICE_ID or give the API key 'Voices: Read'")
        for v in r.json().get("voices", []):
            if (v.get("name") or "").strip().lower() in (name, name.split(" - ")[0]):
                print(f"  voice: found '{v.get('name')}' in your voices ({v['voice_id']})")
                return v["voice_id"]
        raise VoiceError(f"voice '{ELEVEN_VOICE_NAME}' is not in your ElevenLabs voices - add it to My Voices "
                         f"and/or set the variable ELEVEN_VOICE_ID")

    def _eleven_mp3(self, text, path):
        body = {"text": text, "model_id": ELEVEN_MODEL,
                "voice_settings": {"stability": ELEVEN_STABILITY, "similarity_boost": ELEVEN_SIMILARITY}}
        for attempt in range(4):
            try:
                r = requests.post(f"{ELEVEN_API}/v1/text-to-speech/{self.eleven_id}", headers=self._xi(),
                                  params={"output_format": ELEVEN_FORMAT}, json=body, timeout=120)
            except requests.RequestException as e:
                if attempt < 3:
                    time.sleep(4 * (attempt + 1)); continue
                raise VoiceError(f"ElevenLabs not reachable: {redact(str(e))[:200]}")
            if r.status_code == 200 and r.content:
                Path(path).write_bytes(r.content)
                self.chars += len(text)
                return
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:   # busy / too many at once
                time.sleep(4 * (attempt + 1)); continue
            msg = redact(f"HTTP {r.status_code}: {r.text[:300]}"); low = msg.lower()
            hint = ""
            if r.status_code == 401 or "invalid_api_key" in low:
                hint = " -> check the secret ELEVENLABS_API_KEY and its permission 'Text to Speech: Access'"
            elif "free" in low or "subscription" in low or r.status_code == 402:
                hint = " -> Voice Library voices need a paid ElevenLabs plan (Starter) for the API"
            elif "quota" in low or "credit" in low:
                hint = " -> no ElevenLabs credits left this month"
            elif "voice" in low:
                hint = " -> check ELEVEN_VOICE_ID / that Nick is in My Voices"
            raise VoiceError(f"ElevenLabs refused the request ({msg}){hint}")

    def _make_ready(self):
        if not self._ready:
            self.eleven_id = self._find_voice()
            self._ready = True
            print(f"  voice: ElevenLabs ready ({self.eleven_id}, {ELEVEN_MODEL})")

    # ------------------------------------------------------------------ offline voices (only if chosen)
    def _kokoro_wav(self, text, path):
        lang = "en-gb" if KOKORO_VOICE.startswith("b") else "en-us"
        samples, sr = self._kokoro.create(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang=lang)
        pcm = (np.clip(np.asarray(samples, np.float32), -1, 1) * 32000).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(int(sr)); wf.writeframes(pcm.tobytes())

    def _piper_wav(self, text, path):
        v = self._piper
        with wave.open(str(path), "wb") as wf:
            if hasattr(v, "synthesize_wav"):
                try:
                    from piper import SynthesisConfig
                    v.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=VOICE_SPEED,
                                                                          noise_scale=VOICE_EXPRESSION))
                except (ImportError, TypeError):
                    v.synthesize_wav(text, wf)
            else:
                v.synthesize(text, wf, length_scale=VOICE_SPEED, noise_scale=VOICE_EXPRESSION)

    # ------------------------------------------------------------------ one sentence
    def _take(self, text):
        """One take of one sentence -> float32 array at 44.1 kHz (silence trimmed, loudness evened out)."""
        self.n += 1
        if self.engine == "elevenlabs":
            raw = self.dir / f"line{self.n:02d}.mp3"
            self._eleven_mp3(text, raw)
        else:
            raw = self.dir / f"line{self.n:02d}.wav"
            (self._kokoro_wav if self.engine == "kokoro" else self._piper_wav)(text, raw)
        a = _wav_to_array(raw, pitch=VOICE_PITCH)
        nz = np.flatnonzero(np.abs(a) > 0.004)               # trim silence below -48 dBFS
        a = a[max(nz[0] - 800, 0): nz[-1] + 1600] if len(nz) else a
        if self.engine == "elevenlabs":
            from .audiofx import polish_line
            a = polish_line(a)                               # 70 Hz high-pass, 3:1 compression, -20 dBFS RMS
        return a

    # ---- cache: every ElevenLabs sentence is recorded once and kept in the repo (data/voice/)
    @staticmethod
    def cache_path(text):
        parts = [text, "elevenlabs", ELEVEN_MODEL, ELEVEN_VOICE_ID or ELEVEN_VOICE_NAME, ELEVEN_STABILITY,
                 ELEVEN_SIMILARITY, ELEVEN_FORMAT, VOICE_PITCH, CACHE_VERSION]
        key = _json.dumps(parts, ensure_ascii=False)
        return CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".flac")

    def _one(self, text):
        if self.engine != "elevenlabs":
            return self._take(text)
        path = self.cache_path(text)
        if path.exists():
            self.hits += 1
            print(f"  voice cache: hit  '{text}'")
            return _wav_to_array(path)
        self._make_ready()
        from .voiceqa import best_take
        a = best_take(lambda: self._take(text), text)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "cache_tmp.wav"
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:a", "flac", str(path)], check=True)
        self.misses += 1
        print(f"  voice cache: NEW  '{text}' (saved)")
        return a

    def say(self, text, max_len=None):
        """Speak a line calmly: each sentence separately, with a short pause in between. Never sped up."""
        gap = np.zeros(int(VOICE_PAUSE * SR), np.float32)
        out = []
        for i, p in enumerate(sentences(text)):
            if i: out.append(gap)
            out.append(self._one(p))
        return np.concatenate(out) if out else np.zeros(1, np.float32)


def prune_cache(texts):
    """Delete every recording in data/voice/ that does not belong to the current ElevenLabs voice and lines
    (e.g. the old OpenAI recordings). Returns the number of deleted files."""
    keep = {Voice.cache_path(s).name for t in texts for s in sentences(t)}
    removed = 0
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.flac"):
            if f.name not in keep:
                f.unlink(); removed += 1
    return removed
