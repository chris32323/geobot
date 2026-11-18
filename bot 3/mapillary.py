"""Find and download a real 360° panorama from Mapillary (images are CC BY-SA 4.0)."""
import gzip, io, math, random, struct, time
import numpy as np
import requests
from PIL import Image
from .config import MAPILLARY_TOKEN, MIN_PANO_WIDTH, REQUIRE_ROAD
from .level import level_panorama
from .safety import redact
from . import vision

Image.MAX_IMAGE_PIXELS = 120_000_000      # refuse absurdly large images (decompression-bomb guard)

API = "https://graph.mapillary.com/images"
FIELDS = "id,thumb_original_url,is_pano,width,height,captured_at,computed_geometry,creator"


class NoPanorama(Exception):
    pass


class MapillaryTrouble(NoPanorama):
    """Mapillary's API failed (server errors) - the place itself may be fine, so it is NOT blocked."""


MAX_HALF = 0.045            # Mapillary allows at most 0.010 square degrees per box (0.09 x 0.09 here)
TILES = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
TILE_Z = 14                 # the zoom level whose "image" layer lists every single image
_state = {"in_a_row": 0, "search_failures": 0, "tiles": {}, "details": {}}


# ------------------------------------------------------------------ way 1: normal search (fast)
def _bbox_search(lon, lat, half):
    """One search request for all panoramas in the box. None = Mapillary's search failed."""
    bbox = f"{lon - half:.5f},{lat - half:.5f},{lon + half:.5f},{lat + half:.5f}"
    for limit in (100, 25):
        try:
            r = requests.get(API, params=dict(access_token=MAPILLARY_TOKEN, fields=FIELDS, bbox=bbox,
                                              is_pano="true", limit=limit), timeout=60)
            if r.status_code == 200:
                return r.json().get("data", [])
            print(redact(f"  mapillary search HTTP {r.status_code} (limit {limit}): {r.text[:120]}"))
        except requests.RequestException as e:
            print(redact(f"  mapillary search error: {e}"))
        time.sleep(1)
    return None


# ------------------------------------------------------------------ way 2: map tiles (robust)
# Mapillary's coverage tiles are pre-built files on a different server. Their "image" layer lists every
# image with position and is_pano, so they still work when the search is overloaded in dense places.
def _varint(b, i):
    r = sh = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << sh
        if c < 0x80:
            return r, i
        sh += 7


def _fields(b):
    """Minimal protobuf reader: yields (field number, wire type, value)."""
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v = b[i:i + 8]; i += 8
        elif wt == 2:
            ln, i = _varint(b, i); v = b[i:i + ln]; i += ln
        elif wt == 5:
            v = b[i:i + 4]; i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        yield f, wt, v


def _packed(v, wt):
    if wt != 2:
        return [v]
    out, i = [], 0
    while i < len(v):
        x, i = _varint(v, i); out.append(x)
    return out


def _zz(n):
    return (n >> 1) ^ -(n & 1)


def _value(b):
    for f, wt, v in _fields(b):
        if f == 1: return v.decode("utf-8", "replace")
        if f == 2: return struct.unpack("<f", v)[0]
        if f == 3: return struct.unpack("<d", v)[0]
        if f == 4: return v - (1 << 64) if v >= (1 << 63) else v
        if f == 5: return v
        if f == 6: return _zz(v)
        if f == 7: return bool(v)
    return None


def decode_points(tile, layer_name="image"):
    """Mapbox vector tile -> [(x, y, extent, properties)] for the point features of one layer."""
    out = []
    for f, wt, layer in _fields(tile):
        if f != 3:
            continue
        name, keys, vals, feats, extent = None, [], [], [], 4096
        for lf, lwt, lv in _fields(layer):
            if lf == 1: name = lv.decode("utf-8", "replace")
            elif lf == 2: feats.append(lv)
            elif lf == 3: keys.append(lv.decode("utf-8", "replace"))
            elif lf == 4: vals.append(_value(lv))
            elif lf == 5: extent = lv
        if name != layer_name:
            continue
        for fb in feats:
            fid, tags, geom, gtype = None, [], [], 0
            for ff, fwt, fv in _fields(fb):
                if ff == 1: fid = fv
                elif ff == 2: tags += _packed(fv, fwt)
                elif ff == 3: gtype = fv
                elif ff == 4: geom += _packed(fv, fwt)
            if gtype != 1 or len(geom) < 3 or (geom[0] & 7) != 1:
                continue
            props = {keys[tags[k]]: vals[tags[k + 1]] for k in range(0, len(tags) - 1, 2)
                     if tags[k] < len(keys) and tags[k + 1] < len(vals)}
            if props.get("id") is None:
                props["id"] = fid
            x = y = 0
            for k in range(min(geom[0] >> 3, (len(geom) - 1) // 2)):
                x += _zz(geom[1 + 2 * k]); y += _zz(geom[2 + 2 * k])
                out.append((x, y, extent, props))
    return out


def _tile_xy(lon, lat, z):
    n = 2 ** z
    lat = max(min(lat, 85.0), -85.0)
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def _tile_lonlat(z, tx, ty, px, py, extent):
    n = 2 ** z
    lon = (tx + px / extent) / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + py / extent) / n))))
    return lon, lat


def _tile(tx, ty):
    """360° images in one tile as [(lon, lat, props)], [] if the tile is empty, None if it failed."""
    key = (TILE_Z, tx, ty)
    if key in _state["tiles"]:
        return _state["tiles"][key]
    for attempt in range(2):
        try:
            r = requests.get(TILES.format(z=TILE_Z, x=tx, y=ty), params=dict(access_token=MAPILLARY_TOKEN),
                             timeout=60)
            if r.status_code in (204, 404):
                _state["tiles"][key] = []; return []
            if r.status_code == 200:
                data = r.content
                if data[:2] == b"\x1f\x8b":
                    data = gzip.decompress(data)
                pts = [(*_tile_lonlat(TILE_Z, tx, ty, px, py, ext), props)
                       for px, py, ext, props in decode_points(data, "image")
                       if props.get("is_pano") not in (False, 0)]
                _state["tiles"][key] = pts
                return pts
            print(redact(f"  mapillary tile HTTP {r.status_code}: {r.text[:120]}"))
        except Exception as e:
            print(redact(f"  mapillary tile error: {type(e).__name__}: {e}"))
        time.sleep(1)
    return None


def _details(image_id):
    """Full metadata for one image (single-image requests are cheap and reliable).
    Returns dict, False if the image is gone, None on an API error."""
    if image_id in _state["details"]:
        return _state["details"][image_id]
    for attempt in range(2):
        try:
            r = requests.get(f"https://graph.mapillary.com/{image_id}",
                             params=dict(access_token=MAPILLARY_TOKEN, fields=FIELDS), timeout=30)
            if r.status_code == 200:
                _state["details"][image_id] = r.json(); return _state["details"][image_id]
            if r.status_code in (400, 404):
                return False
            print(redact(f"  mapillary image HTTP {r.status_code}: {r.text[:120]}"))
        except requests.RequestException as e:
            print(redact(f"  mapillary image error: {e}"))
        time.sleep(1)
    return None


def _tile_search(lon, lat, half, recent=False, want=15, max_lookups=30):
    x0, y0 = _tile_xy(lon - half, lat + half, TILE_Z)
    x1, y1 = _tile_xy(lon + half, lat - half, TILE_Z)
    cx, cy = _tile_xy(lon, lat, TILE_Z)
    tiles = sorted(((tx, ty) for tx in range(int(x0), int(x1) + 1) for ty in range(int(y0), int(y1) + 1)),
                   key=lambda t: (t[0] + 0.5 - cx) ** 2 + (t[1] + 0.5 - cy) ** 2)[:4]
    pts, worked = [], False
    for tx, ty in tiles:
        got = _tile(tx, ty)
        if got is None:
            continue
        worked = True
        pts += [p for p in got if abs(p[0] - lon) <= half and abs(p[1] - lat) <= half]
    if not worked:
        return None
    if recent:
        pts.sort(key=lambda p: -float(p[2].get("captured_at") or 0))
    else:
        pts.sort(key=lambda p: (p[0] - lon) ** 2 + (p[1] - lat) ** 2)
    good, errors, looked = [], 0, 0
    for plon, plat, props in pts[:max_lookups]:
        d = _details(str(props.get("id")))
        looked += 1
        if d is None:
            errors += 1; continue
        if d and _good(d):
            good.append(d)
            if len(good) >= want:
                break
    print(f"  map tiles: {len(pts)} panoramas nearby, {len(good)} usable after checking {looked}")
    if pts and not good and errors == looked:
        return None
    return good


def _search(lon, lat, half, recent=False):
    """Panoramas in the box, or None if Mapillary failed (not the same as 'nothing there').
    Normal search first; after 2 failed searches the rest of the run uses the map tiles directly."""
    half = min(half, MAX_HALF)
    found = None
    if _state["search_failures"] < 2:
        found = _bbox_search(lon, lat, half)
        if found is None:
            _state["search_failures"] += 1
            if _state["search_failures"] == 2:
                print("  Mapillary search is overloaded - using map tiles for the rest of this run")
    if found is None:
        found = _tile_search(lon, lat, half, recent=recent)
    if found is None:
        _state["in_a_row"] += 1
        if _state["in_a_row"] >= 6:
            raise RuntimeError("Mapillary keeps failing (search and map tiles, 6 times in a row) - "
                               "stopping early, the next scheduled run will try again")
        return None
    _state["in_a_row"] = 0
    return found


def _good(img_meta):
    w, h = img_meta.get("width") or 0, img_meta.get("height") or 0
    return (img_meta.get("is_pano") and img_meta.get("thumb_original_url")
            and w >= MIN_PANO_WIDTH and h > 0 and abs(w / h - 2.0) < 0.03)


def _detected_classes(image_id):
    """Mapillary's automatic segmentation labels for an image, e.g. 'construction--flat--road', 'nature--sky'."""
    try:
        r = requests.get(f"https://graph.mapillary.com/{image_id}/detections",
                         params=dict(access_token=MAPILLARY_TOKEN, fields="value"), timeout=60)
        if r.status_code != 200:
            return None
        return {d.get("value", "") for d in r.json().get("data", [])}
    except requests.RequestException:
        return None


def _is_outdoor(classes, need_road):
    """True = suitable outdoor scene, False = clearly not (indoors / no road), None = unknown (no detections yet)."""
    if not classes:
        return None
    sky = "nature--sky" in classes
    if not need_road:
        return sky
    road = any(c.startswith(("construction--flat--road", "marking--", "construction--flat--crosswalk")) for c in classes)
    return road and sky


# search box half-sizes in degrees: sights must be close so the landmark is actually in the picture
SEARCH = {"landmark": (0.002, 0.004, 0.008), "border": (0.002, 0.004, 0.008),
          "landscape": (0.004, 0.012, 0.03), "street": (0.004, 0.012, 0.03)}   # cities are dense: small boxes


def _sky_visible(arr):
    """Own check on the pixels: the upper part of an outdoor panorama is mostly sky, indoors it is a ceiling.
    Counts deep blue (clear desert / mountain sky), light blue, and bright grey/white (overcast) as sky."""
    h = arr.shape[0]
    top = arr[int(h * 0.02): int(h * 0.22)].astype(np.float32)
    r, g, b = top[..., 0], top[..., 1], top[..., 2]
    bright = (r + g + b) / 3
    spread = np.max(top, axis=-1) - np.min(top, axis=-1)
    blue = (b > r + 20) & (b >= g - 5) & (b > 80)
    overcast = (bright > 165) & (spread < 45)
    sky_like = blue | overcast | (bright > 205)
    return float(sky_like.mean()) > 0.40


def fetch_panorama(lon, lat, rng=None, kind="street", place=None):
    """Return (equirectangular RGB array, metadata dict). Raises NoPanorama."""
    if not MAPILLARY_TOKEN:
        raise RuntimeError("MAPILLARY_TOKEN is not set")
    rng = rng or random.Random()
    state = {"checks": 0, "fallback": None}

    def consider(got):
        """Optional AI check. Returns the panorama to use, or None to keep looking."""
        if not got:
            return None
        if not vision.ENABLED:
            return got
        if state["checks"] >= vision.MAX_CHECKS:
            return state["fallback"] or got
        state["checks"] += 1
        res = vision.check(got[0], place or "this place", kind)
        if res is None:
            return got                                   # API problem: never block the post
        got[1]["ai_check"] = res
        print(f"  {got[1]['id']}: AI check {res}")
        if not res["usable"] or res["spoiler"]:
            return None
        if kind != "street" and res["sight_visible"] is False:
            state["fallback"] = state["fallback"] or got  # usable, but the sight isn't in view: keep as plan B
            return None
        return got

    need_road = REQUIRE_ROAD and kind == "street"
    api_failed = 0; searches = 0
    for half in SEARCH.get(kind, SEARCH["street"]):
        searches += 1
        found = _search(lon, lat, half, recent=(kind == "street"))
        if found is None:
            api_failed += 1; continue
        cands = [m for m in found if _good(m)]
        print(f"  bbox ±{half}°: {len(cands)} usable panoramas")
        rng.shuffle(cands)
        # sights: closest first; streets: recent, high-resolution captures first
        if kind == "street":
            cands.sort(key=lambda m: (-(m.get("width") or 0) // 1000, -(m.get("captured_at") or 0)))
        else:
            def dist(m):
                c = (m.get("computed_geometry") or {}).get("coordinates") or [lon + 1, lat + 1]
                return (c[0] - lon) ** 2 + (c[1] - lat) ** 2
            cands.sort(key=dist)
        maybes = []
        for m in cands[:15]:
            ok = None
            if REQUIRE_ROAD:
                ok = _is_outdoor(_detected_classes(m["id"]), need_road)
                if ok is False:
                    print(f"  {m['id']}: {'no road/sky' if need_road else 'no sky'} detected (indoors?), skipped")
                    continue
                if ok is None:
                    print(f"  {m['id']}: no Mapillary detections yet, will check the pixels instead")
                    maybes.append(m); continue
            got = consider(_download(m, lon, lat, sky_known=ok is True))
            if got: return got
        for m in maybes[:4]:                  # no detections yet: rely on our own sky check only
            got = consider(_download(m, lon, lat))
            if got: return got
    if state["fallback"]:
        print("  no view with the sight clearly visible - using the best usable panorama")
        return state["fallback"]
    if api_failed == searches:
        raise MapillaryTrouble(f"Mapillary API errors near {lat:.4f},{lon:.4f}")
    raise NoPanorama(f"no usable outdoor 360° panorama near {lat:.4f},{lon:.4f}")


def _download(m, lon, lat, sky_known=False):
    try:
        r = requests.get(m["thumb_original_url"], timeout=120)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(redact(f"  download failed ({e}), next"))
        return None
    if im.width > 8192:
        im = im.resize((8192, 4096), Image.LANCZOS)
    arr = np.array(im)
    if arr.std() < 12:                     # blank / broken image
        print(f"  {m['id']}: image is blank or broken, skipped")
        return None
    if REQUIRE_ROAD and not sky_known and not _sky_visible(arr):   # Mapillary already saw sky -> trust it
        print(f"  {m['id']}: no open sky at the top (probably indoors), skipped")
        return None
    arr, lv = level_panorama(arr)          # straighten a crooked camera, or skip if far too tilted
    print(f"  {m['id']}: horizon check {lv}")
    if arr is None:
        return None
    geo = (m.get("computed_geometry") or {}).get("coordinates") or [lon, lat]
    meta = dict(id=m["id"], lon=geo[0], lat=geo[1], captured_at=m.get("captured_at"),
                creator=(m.get("creator") or {}).get("username"),
                url=f"https://www.mapillary.com/app/?pKey={m['id']}", level=lv)
    print(f"  downloaded panorama {m['id']}")
    return arr, meta
