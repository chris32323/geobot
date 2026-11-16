"""Central settings. Everything can be overridden with environment variables (GitHub Secrets / Variables)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
NE_DIR = Path(os.getenv("NE_DIR", DATA / "naturalearth"))
MODELS = Path(os.getenv("MODELS_DIR", ROOT / "models"))

FONT_BOLD = str(ASSETS / "fonts" / "Poppins-Bold.ttf")
FONT_MEDIUM = str(ASSETS / "fonts" / "Poppins-Medium.ttf")

# ---- video
W, H, FPS = 1080, 1920, 30
ROUNDS_PER_REEL = int(os.getenv("ROUNDS_PER_REEL", "2"))
RADIUS_KM = int(os.getenv("RADIUS_KM", "2000"))
TITLE_EDITION = os.getenv("TITLE_EDITION", "World Edition")
HFOV_DEG = 82

# ---- voice (male by default)
TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")             # piper | openai
PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-ryan-high")   # male US voice
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "onyx")            # male voice
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
REQUIRE_VOICE = os.getenv("REQUIRE_VOICE", "1") == "1"
VOICE_PITCH = float(os.getenv("VOICE_PITCH", "0.88"))      # < 1 = deeper voice (0.80 very deep, 1.0 original)

# ---- Mapillary
MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN", "")
MIN_PANO_WIDTH = int(os.getenv("MIN_PANO_WIDTH", "3000"))
REQUIRE_ROAD = os.getenv("REQUIRE_ROAD", "1") == "1"         # only outdoor street panoramas (road + sky detected)

# ---- Instagram (Instagram API with Instagram Login)
IG_API = os.getenv("IG_API", "https://graph.instagram.com/v26.0")
IG_USER_ID = os.getenv("IG_USER_ID", "").strip().strip('"').strip()
IG_ACCESS_TOKEN = "".join(os.getenv("IG_ACCESS_TOKEN", "").split()).strip('"\'')

HASHTAGS = os.getenv("HASHTAGS", "#geoguessr #geography #travel #guessthelocation #maps #quiz #streetview")
