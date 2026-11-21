"""Entry point.

  python -m bot.main render  --out out            # pick locations, fetch real panoramas, voice, render reel
  python -m bot.main render  --out out --demo     # offline test with synthetic panoramas (no tokens needed)
  python -m bot.main publish --out out --video-url URL
  python -m bot.main refresh-token                # prints a refreshed Instagram token
"""
import argparse, json, random, sys, datetime as dt
from pathlib import Path
import numpy as np
from . import config

HISTORY = config.DATA / "history.json"
LOCATIONS = config.DATA / "locations.json"

def intro_for(slot, n):
    if slot == 0:
        return ("ROUND 1", "WHERE ARE WE?", config.TITLE_EDITION)
    if n >= 3 and slot == n - 1:
        return ("FINAL ROUND", "LAST CHANCE!", "Only the pros get this one")
    return (f"ROUND {slot + 1}", "WHERE ARE WE NOW?", "It gets harder now")

SAY_INTRO = ["Where in the world are we?", "Next round. Where are we now?", "Last round. Where are we?"]
SAY_PAN = "Can you guess it?"          # as soon as the 360° pan starts
SAY_GUESS = "Lock in your guess."      # when the 2D map appears
SAY_PAN1 = ["Take a good look around.", "Okay, this one is a little tricky.", "Take your time on this one."]
SAY_RIGHT = "Did you get it right?"


def reveal_line(loc):
    """'We're in London.' -> 'And we're in London.' (sounds more natural as a reveal)."""
    r = (loc.get("reveal") or f"This is {loc.get('say') or loc['name']}.").strip()
    return "And " + r[0].lower() + r[1:] if r.lower().startswith(("we're", "we are")) else r
GENERIC_HINTS = ["Watch the road.", "Look at the signs.", "Check the plants.", "Look at the houses.", "Which side do they drive on?"]

# 50 / 50: round 1 is always a sight (famous landmark or famous landscape),
# round 2 is "everything else": beautiful scenery (deserts, mountains, forests, coasts), towns, streets, borders.
ROUND1_WEIGHTS = {"landmark": 0.5, "landscape": 0.5}
ROUND2_WEIGHTS = {"scenery": 0.55, "street": 0.35, "border": 0.10}


def load_json(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


NO_COVERAGE_DAYS = 21          # a place without a usable panorama is skipped for 3 weeks, then tried again


def _no_coverage(history):
    """{location_id: 'YYYY-MM-DD'}. Old versions kept a permanent list -> start fresh."""
    nc = history.get("no_coverage")
    if not isinstance(nc, dict):
        nc = {}; history["no_coverage"] = nc
    return nc


def blocked(history):
    from datetime import date, timedelta
    limit = (date.today() - timedelta(days=NO_COVERAGE_DAYS)).isoformat()
    return {i for i, d in _no_coverage(history).items() if d > limit}


def available_locations(history):
    """(fresh, reuse): fresh = not shown in the last 60 posts; reuse = shown before, the longest ago first.
    Places without a usable panorama (paused for 21 days) are left out of both."""
    locs = load_json(LOCATIONS, [])
    dead = blocked(history)
    last_seen = {}
    for i, post in enumerate(history.get("posted", [])):
        for lid in post.get("location_ids", []):
            last_seen[lid] = i
    recent = set()
    for post in history.get("posted", [])[-60:]:
        recent.update(post.get("location_ids", []))
    alive = [l for l in locs if l["id"] not in dead]
    fresh = [l for l in alive if l["id"] not in recent]
    reuse = sorted((l for l in alive if l["id"] in recent), key=lambda l: last_seen.get(l["id"], -1))
    return fresh, reuse


def weighted_order(rng, locs, weights, rank=None):
    """Random order where the TYPE is drawn by its weight first, then a random place of that type.
    So the mix follows the weights no matter how many places of each type exist. Weight 0 = dropped."""
    groups = {}
    for l in locs:
        if weights.get(l.get("type"), 0) > 0:
            groups.setdefault(l["type"], []).append(l)
    for g in groups.values():
        rng.shuffle(g)
        if rank:                                     # within a type: preferred difficulty first (popped from the end)
            g.sort(key=rank, reverse=True)
    out = []
    while groups:
        types = sorted(groups)
        t = rng.choices(types, [weights[x] for x in types])[0]
        out.append(groups[t].pop())
        if not groups[t]:
            del groups[t]
    return out


def candidates_for(slot, rng, pool):
    if slot == 0:   # a sight; within each type easy/medium first, hard only when those are used up
        return (weighted_order(rng, pool, ROUND1_WEIGHTS, rank=lambda l: l.get("difficulty") == "hard")
                + weighted_order(rng, pool, ROUND2_WEIGHTS))
    # round 2: scenery / town / border; medium/hard first
    return (weighted_order(rng, pool, ROUND2_WEIGHTS, rank=lambda l: l.get("difficulty") == "easy")
            + weighted_order(rng, pool, ROUND1_WEIGHTS))


def build_rounds(n, rng, history, demo=False):
    rounds = []; fresh, reuse = available_locations(history); tried = set()
    if demo:
        from PIL import Image
        demo_panos = sorted(Path(config.ROOT / "demo").glob("*.jpg"))
    for slot in range(n):
        # 1) places not shown recently, 2) places shown the longest time ago, 3) round 1 may then use any type
        order = candidates_for(slot, rng, fresh) + candidates_for(slot, rng, reuse[:max(len(reuse) // 2, 10)])
        if slot == 0:
            order += weighted_order(rng, fresh + reuse, {"street": 1, "landscape": 1, "landmark": 1, "border": 1})
        for loc in order:
            if loc["id"] in tried: continue
            tried.add(loc["id"])
            kind = loc.get("type", "street")
            print(f"round {slot + 1}: trying {loc['name']}, {loc['country']} ({kind}, {loc['difficulty']})")
            if demo:
                pano = np.array(Image.open(demo_panos[slot % len(demo_panos)]).convert("RGB"))[:, ::-1].copy()
                meta = dict(id="demo", creator="demo", lon=loc["lon"], lat=loc["lat"], url="")
            else:
                from .mapillary import fetch_panorama, NoPanorama, MapillaryTrouble
                try:
                    pano, meta = fetch_panorama(loc["lon"], loc["lat"], rng, kind=kind,
                                                place=f"{loc['name']}, {loc['country']}")
                except MapillaryTrouble as e:
                    print(f"  {e} - trying another place (not blocked)"); continue
                except NoPanorama as e:
                    print(f"  {e} - skipping this place for {NO_COVERAGE_DAYS} days")
                    from datetime import date
                    _no_coverage(history)[loc["id"]] = date.today().isoformat(); continue
            fact = (loc.get("fact") or "").strip()
            reveal = reveal_line(loc)
            if slot == 0: reveal += " Did you get it right?"
            rounds.append(dict(
                pano=pano, lonlat=(meta["lon"], meta["lat"]) if kind in ("street", "scenery") else (loc["lon"], loc["lat"]),
                difficulty=loc.get("difficulty", "medium"),
                place=loc["name"] if loc["country"] in loc["name"] else f"{loc['name']}, {loc['country']}",
                fact=fact, loc_id=loc["id"], intro=intro_for(slot, n),
                lines=dict(intro=SAY_INTRO[min(slot, 2)], pan1=SAY_PAN, guess=SAY_GUESS, reveal=reveal),
                credit="Imagery: Mapillary · CC BY-SA 4.0",
                meta=meta))
            break
        else:
            raise RuntimeError("ran out of locations with usable panoramas")
    return rounds


def make_caption(rounds):
    from .safety import clean_username
    creators = sorted({clean_username(r["meta"].get("creator")) for r in rounds
                       if r["meta"].get("creator") and r["meta"]["creator"] != "demo"} - {""})
    links = " ".join(r["meta"]["url"] for r in rounds if r["meta"].get("url"))
    return (f"Where are we? 🌍 {len(rounds)} rounds – guess before the timer runs out!\n\n"
            f"How many did you get right? Comment your score 👇 New round every day.\n\n"
            f"Street-level imagery: Mapillary contributors{(' (' + ', '.join(creators) + ')') if creators else ''}, "
            f"CC BY-SA 4.0. {links}\nMap data: Natural Earth.\n\n{config.HASHTAGS}")


BOT_VERSION = "2026-11-21c (new lines, difficulty stars, 50/50 places)"


def cmd_render(args):
    print(f"geobot version {BOT_VERSION}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else int(dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d"))
    rng = random.Random(seed)
    history = load_json(HISTORY, {"posted": [], "no_coverage": {}})
    rounds = build_rounds(config.ROUNDS_PER_REEL, rng, history, demo=args.demo)
    tts = None
    try:
        if args.fake_voice:
            from .render import SR
            tts = lambda text, mx: (0.15 * np.sin(2 * np.pi * 180 * np.arange(int(len(text) * 0.065 * SR)) / SR)).astype(np.float32)
        else:
            from .voice import Voice
            v = Voice(out / "voice"); tts = v.say
    except Exception as e:
        if config.REQUIRE_VOICE and not args.demo:
            raise
        print(f"WARNING: voice disabled ({e})")
    from .worldmap import WorldMap
    from .render import Reel
    print("rendering world map ..."); world = WorldMap()
    reel = Reel(world, rounds, tts=tts)
    print("rendering video ..."); reel.render(out / "reel.mp4", str(out))
    (out / "caption.txt").write_text(make_caption(rounds), encoding="utf-8")
    meta = dict(date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), seed=seed,
                location_ids=[r["loc_id"] for r in rounds], places=[r["place"] for r in rounds],
                images=[r["meta"] for r in rounds])
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    HISTORY.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")   # no_coverage updates
    print("done:", out / "reel.mp4")


def cmd_publish(args):
    from .instagram import publish_reel
    out = Path(args.out)
    caption = (out / "caption.txt").read_text(encoding="utf-8")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    mid, link = publish_reel(args.video_url, caption)
    print("published:", mid, link or "")
    history = load_json(HISTORY, {"posted": [], "no_coverage": {}})
    history["posted"].append(dict(date=meta["date"], media_id=mid, permalink=link,
                                  location_ids=meta["location_ids"], places=meta["places"]))
    HISTORY.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_refresh(_):
    from .instagram import refresh_token
    tok, exp = refresh_token()
    print(tok)


def cmd_design(args):
    """One-time: let OpenAI review the real frames and improve the design values (saved to data/design.json)."""
    print(f"geobot version {BOT_VERSION}")
    rng = random.Random(int(dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")) + 7)
    history = load_json(HISTORY, {"posted": [], "no_coverage": {}})     # read only, not saved
    rounds = build_rounds(config.ROUNDS_PER_REEL, rng, history, demo=args.demo)
    from .worldmap import WorldMap
    from . import designer
    print("rendering world map ..."); world = WorldMap()
    designer.run(world, rounds, args.out)


def cmd_voices(args):
    """One-time: record every sentence the bot can ever say into data/voice/ (then the voice is free)."""
    print(f"geobot version {BOT_VERSION}")
    from . import voiceqa
    from .render import Reel
    from .voice import Voice, prune_cache
    v = Voice(Path(args.out) / "voice")
    if v.engine != "elevenlabs":
        sys.exit("ElevenLabs is not the active voice (TTS_ENGINE) - nothing to record")
    fixed = SAY_INTRO + [SAY_PAN, SAY_GUESS, SAY_RIGHT, Reel.END_LINE]
    locs = json.loads(LOCATIONS.read_text())
    texts = fixed + [reveal_line(l) for l in locs]
    for text in texts:
        v.say(text)
    removed = prune_cache(texts)
    print(f"voice cache ready (ElevenLabs): {v.hits} already there, {v.misses} new (~{v.chars} credits), "
          f"{removed} old recordings deleted (e.g. OpenAI). Commit happens in the next step.")


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render"); r.add_argument("--out", default="out"); r.add_argument("--demo", action="store_true")
    r.add_argument("--fake-voice", action="store_true"); r.add_argument("--seed", type=int)
    p = sub.add_parser("publish"); p.add_argument("--out", default="out"); p.add_argument("--video-url", required=True)
    sub.add_parser("refresh-token")
    vo = sub.add_parser("voices"); vo.add_argument("--out", default="out")
    de = sub.add_parser("design"); de.add_argument("--out", default="out"); de.add_argument("--demo", action="store_true")
    a = ap.parse_args(argv)
    {"render": cmd_render, "publish": cmd_publish, "refresh-token": cmd_refresh, "voices": cmd_voices,
     "design": cmd_design}[a.cmd](a)


if __name__ == "__main__":
    main()
