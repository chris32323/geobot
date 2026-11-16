"""Find and download a real 360° panorama from Mapillary (images are CC BY-SA 4.0)."""
import io, random, time
import numpy as np
import requests
from PIL import Image
from .config import MAPILLARY_TOKEN, MIN_PANO_WIDTH

API = "https://graph.mapillary.com/images"
FIELDS = "id,thumb_original_url,is_pano,width,height,captured_at,computed_geometry,creator"


class NoPanorama(Exception):
    pass


def _search(lon, lat, half):
    bbox = f"{lon - half:.5f},{lat - half:.5f},{lon + half:.5f},{lat + half:.5f}"
    for attempt in range(3):
        try:
            r = requests.get(API, params=dict(access_token=MAPILLARY_TOKEN, fields=FIELDS, bbox=bbox,
                                              is_pano="true", limit=100), timeout=60)
            if r.status_code == 200:
                return r.json().get("data", [])
            print(f"  mapillary HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            print(f"  mapillary error: {e}")
        time.sleep(3 * (attempt + 1))
    return []


def _good(img_meta):
    w, h = img_meta.get("width") or 0, img_meta.get("height") or 0
    return (img_meta.get("is_pano") and img_meta.get("thumb_original_url")
            and w >= MIN_PANO_WIDTH and h > 0 and abs(w / h - 2.0) < 0.03)


def fetch_panorama(lon, lat, rng=None):
    """Return (equirectangular RGB array, metadata dict). Raises NoPanorama."""
    if not MAPILLARY_TOKEN:
        raise RuntimeError("MAPILLARY_TOKEN is not set")
    rng = rng or random.Random()
    for half in (0.01, 0.025, 0.05):           # ~1 km, ~2.5 km, ~5 km boxes
        cands = [m for m in _search(lon, lat, half) if _good(m)]
        print(f"  bbox ±{half}°: {len(cands)} usable panoramas")
        rng.shuffle(cands)
        # prefer recent, high-resolution captures
        cands.sort(key=lambda m: (-(m.get("width") or 0) // 1000, -(m.get("captured_at") or 0)))
        for m in cands[:6]:
            try:
                r = requests.get(m["thumb_original_url"], timeout=120)
                r.raise_for_status()
                im = Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception as e:
                print(f"  download failed ({e}), next")
                continue
            if im.width > 8192:
                im = im.resize((8192, 4096), Image.LANCZOS)
            arr = np.array(im)
            if arr.std() < 12:                     # blank / broken image
                continue
            geo = (m.get("computed_geometry") or {}).get("coordinates") or [lon, lat]
            meta = dict(id=m["id"], lon=geo[0], lat=geo[1], captured_at=m.get("captured_at"),
                        creator=(m.get("creator") or {}).get("username"),
                        url=f"https://www.mapillary.com/app/?pKey={m['id']}")
            return arr, meta
    raise NoPanorama(f"no usable 360° panorama near {lat:.4f},{lon:.4f}")
