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

INTROS = [("ROUND 1", "WHERE ARE WE?", "Find it before the timer runs out"),
          ("ROUND 2", "STILL IN?", "It gets harder now"),
          ("FINAL ROUND", "LAST CHANCE!", "Only the pros get this one")]
SAY_INTRO = ["Where are we?", "Still in?", "Final round!"]
SAY_PAN1 = ["Look around. Every detail counts.", "This one is tricky.", "Last one. Make it count."]
SAY_GUESS = ["Lock in your guess. Three seconds!", "Where are we? Make your guess!", "Final answer? Three seconds!"]
GENERIC_HINTS = ["Check the road markings.", "Look at the signs and the houses.", "Notice the plants and the sky.",
                 "Which side of the road do they drive on?"]


def load_json(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def pick_locations(n, rng, history):
    locs = load_json(LOCATIONS, [])
    used = set()
    for post in history.get("posted", [])[-60:]:
        used.update(post.get("location_ids", []))
    dead = set(history.get("no_coverage", []))
    pool = [l for l in locs if l["id"] not in used and l["id"] not in dead] or [l for l in locs if l["id"] not in dead]
    order = {"easy": 0, "medium": 1, "hard": 2}
    rng.shuffle(pool)
    pool.sort(key=lambda l: order.get(l.get("difficulty", "medium"), 1))   # easier rounds first
    # keep a mix: take candidates from easy->hard buckets, keep spares for missing coverage
    return pool


def build_rounds(n, rng, history, demo=False):
    rounds = []; candidates = pick_locations(n, rng, history)
    if demo:
        from PIL import Image
        demo_panos = sorted(Path(config.ROOT / "demo").glob("*.jpg"))
    spots = np.linspace(0, len(candidates) - 1, n * 4).astype(int) if candidates else []
    tried = set()
    for slot in range(n):
        # choose from the matching difficulty band, fall back to anything left
        band = [candidates[i] for i in spots[slot * 4:(slot + 1) * 4]] + candidates
        for loc in band:
            if loc["id"] in tried: continue
            tried.add(loc["id"])
            print(f"round {slot + 1}: trying {loc['name']}, {loc['country']} ({loc['difficulty']})")
            if demo:
                pano = np.array(Image.open(demo_panos[slot % len(demo_panos)]).convert("RGB"))[:, ::-1].copy()
                meta = dict(id="demo", creator="demo", lon=loc["lon"], lat=loc["lat"], url="")
            else:
                from .mapillary import fetch_panorama, NoPanorama
                try:
                    pano, meta = fetch_panorama(loc["lon"], loc["lat"], rng)
                except NoPanorama as e:
                    print(f"  {e}"); history.setdefault("no_coverage", []).append(loc["id"]); continue
            place = f"{loc['name']}, {loc['country']}"
            hint = loc.get("hint") or rng.choice(GENERIC_HINTS)
            reveal = (f"It's {loc.get('say', place)}! Was your guess close?" if slot % 2 == 0
                      else f"This was {loc.get('say', place)}. Did you get it?")
            rounds.append(dict(
                pano=pano, lonlat=(meta["lon"], meta["lat"]), place=place, loc_id=loc["id"],
                intro=INTROS[2] if (n >= 3 and slot == n - 1) else INTROS[min(slot, 1)],
                lines=dict(intro=SAY_INTRO[min(slot, 2)], pan1=SAY_PAN1[min(slot, 2)], pan2=hint,
                           guess=SAY_GUESS[min(slot, 2)], reveal=reveal),
                credit=f"Imagery: Mapillary{' / @' + meta['creator'] if meta.get('creator') else ''} · CC BY-SA 4.0",
                meta=meta))
            break
        else:
            raise RuntimeError("ran out of locations with usable panoramas")
    return rounds


def make_caption(rounds):
    creators = sorted({r["meta"].get("creator") for r in rounds if r["meta"].get("creator") and r["meta"]["creator"] != "demo"})
    links = " ".join(r["meta"]["url"] for r in rounds if r["meta"].get("url"))
    return (f"Where are we? 🌍 {len(rounds)} rounds – guess before the timer runs out!\n\n"
            f"How many did you get right? Comment your score 👇 New round every day.\n\n"
            f"Street-level imagery: Mapillary contributors{(' (' + ', '.join(creators) + ')') if creators else ''}, "
            f"CC BY-SA 4.0. {links}\nMap data: Natural Earth.\n\n{config.HASHTAGS}")


def cmd_render(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else int(dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d"))
    rng = random.Random(seed)
    history = load_json(HISTORY, {"posted": [], "no_coverage": []})
    rounds = build_rounds(config.ROUNDS_PER_REEL, rng, history, demo=args.demo)
    tts = None
    try:
        if args.fake_voice:
            from .render import SR
            tts = lambda text, mx: (0.15 * np.sin(2 * np.pi * 180 * np.arange(int(min(len(text) * 0.065, mx) * SR)) / SR)).astype(np.float32)
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
    history = load_json(HISTORY, {"posted": [], "no_coverage": []})
    history["posted"].append(dict(date=meta["date"], media_id=mid, permalink=link,
                                  location_ids=meta["location_ids"], places=meta["places"]))
    HISTORY.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_refresh(_):
    from .instagram import refresh_token
    tok, exp = refresh_token()
    print(tok)


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render"); r.add_argument("--out", default="out"); r.add_argument("--demo", action="store_true")
    r.add_argument("--fake-voice", action="store_true"); r.add_argument("--seed", type=int)
    p = sub.add_parser("publish"); p.add_argument("--out", default="out"); p.add_argument("--video-url", required=True)
    sub.add_parser("refresh-token")
    a = ap.parse_args(argv)
    {"render": cmd_render, "publish": cmd_publish, "refresh-token": cmd_refresh}[a.cmd](a)


if __name__ == "__main__":
    main()
