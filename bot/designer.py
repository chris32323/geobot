"""One-time design review by OpenAI.

Renders 5 real still frames (title, 360° pan, map + timer, answer card, end card), sends them together with
the current design values to an OpenAI vision model and lets it improve the values (colours, sizes, positions,
transparency) - never code. Every value is clamped to safe limits and overlaps are fixed automatically.
Two rounds: improve, then look at the result and refine. The result is saved to data/design.json and used
for every video from then on. Delete that file to go back to the original design.
"""
import base64, io, json, os, time
from pathlib import Path
import requests
from PIL import Image

from . import design as D
from .safety import redact

DESIGN_MODEL = os.getenv("DESIGN_MODEL") or os.getenv("VISION_MODEL") or "gpt-5-mini"
ROUNDS = int(os.getenv("DESIGN_ROUNDS") or 2)

BRIEF = """You are a senior motion and UI designer for short-form vertical video (Instagram Reels, TikTok, 1080x1920).
You review a daily geography quiz reel ("GeoGuessr style"): a real 360° street panorama is shown, viewers guess
the place, then a world map with a pin and a 2000 km circle reveals it, followed by an answer card and an end card.

The 5 images are, in order: 1) title screen "WHERE ARE WE?", 2) 360° pan with countdown timer,
3) world map with timer and "LOCK IN YOUR GUESS", 4) map with pin + answer card, 5) end card.
The panorama photo and the world map itself cannot be changed - only the overlay design.

Goals: premium, modern, clean, very readable on a phone in one second, strong contrast, consistent style,
pleasant colours that work on any photo (sky, desert, snow, city at night). Respect the Instagram safe zones:
the bottom ~250 px are partly covered by the caption and the right ~140 px by the like/comment buttons.

You may ONLY change these design values (current values and allowed ranges are given). Colours are hex.
Reply with JSON only:
{"theme": {only the keys you change: new value}, "changes": ["short reason per change"], "verdict": "one sentence"}
If the design is already very good, return an empty "theme" and say so in "verdict".
"""


def _jpeg_b64(img, w=540):
    im = img.convert("RGB"); im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=88)
    return base64.b64encode(b.getvalue()).decode()


def key_frames(world, rounds):
    """The 5 typical screens of a reel, rendered with the current design (no voice needed)."""
    from .render import Reel
    reel = Reel(world, rounds, tts=None)
    R0 = reel.rounds[0]; tl = R0["tl"]; s0 = reel.starts[0]
    times = [s0 + 0.3, s0 + tl["pan0"] + 1.6, s0 + tl["guess0"] + 1.2, s0 + tl["card"] + 1.6, reel.rounds_end + 2.2]
    return [reel.frame(t) for t in times]


def contact_sheet(frames, path, w=432):
    small = [f.resize((w, int(f.height * w / f.width)), Image.LANCZOS) for f in frames]
    sheet = Image.new("RGB", (w * len(small) + 12 * (len(small) - 1), small[0].height), (20, 20, 24))
    for i, f in enumerate(small):
        sheet.paste(f, (i * (w + 12), 0))
    sheet.save(path, quality=90)


def ask(frames, theme, note=""):
    key = os.getenv("OPENAI_API_KEY", "").strip()
    limits = {k: (v if isinstance(v, tuple) else ("hex" if v == "hex" else "true/false")) for k, v in D.LIMITS.items()}
    text = (BRIEF + f"\nCurrent design values:\n{json.dumps(theme, indent=1)}\n"
            f"Allowed ranges (min, max):\n{json.dumps(limits)}\n" + note)
    content = [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(f)}", "detail": "high"}}
        for f in frames]
    for attempt in range(3):
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", timeout=300,
                              headers={"Authorization": f"Bearer {key}"},
                              json=dict(model=DESIGN_MODEL, messages=[{"role": "user", "content": content}],
                                        response_format={"type": "json_object"}, max_completion_tokens=8000))
            if r.status_code == 200:
                msg = r.json()["choices"][0]["message"].get("content") or "{}"
                usage = r.json().get("usage", {})
                print(f"  OpenAI ({DESIGN_MODEL}): {usage.get('prompt_tokens', '?')} in / "
                      f"{usage.get('completion_tokens', '?')} out tokens")
                return json.loads(msg)
            print(redact(f"  OpenAI HTTP {r.status_code}: {r.text[:300]}"))
            if r.status_code in (400, 401, 403, 404):
                break
        except Exception as e:
            print(redact(f"  OpenAI error: {e}"))
        time.sleep(5)
    return None


def run(world, rounds, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing - the design run needs OpenAI")
    theme = D.load_theme()
    D.apply_theme(theme)
    frames = key_frames(world, rounds)
    contact_sheet(frames, out / "design_before.jpg")
    report = [f"design run with {DESIGN_MODEL}"]
    for rnd in range(1, ROUNDS + 1):
        note = ("" if rnd == 1 else
                "\nThis is round 2: the images already show your previous changes. Check them critically "
                "(readability, balance, contrast, overlaps) and refine. Return an empty theme if it is now good.")
        print(f"design round {rnd}: asking OpenAI ...")
        ans = ask(frames, theme, note)
        if not ans:
            report.append(f"round {rnd}: no answer from OpenAI - keeping the current design"); break
        changes = ans.get("theme") or {}
        report.append(f"round {rnd} verdict: {ans.get('verdict', '')}")
        report += [f"  - {c}" for c in ans.get("changes", [])]
        if not changes:
            report.append(f"round {rnd}: no further changes"); break
        new, notes = D.clean_theme({**theme, **changes})
        report += [f"  (safety) {n}" for n in notes]
        diff = {k: (theme[k], new[k]) for k in new if new[k] != theme.get(k)}
        report += [f"  {k}: {a} -> {b}" for k, (a, b) in diff.items()]
        theme = new
        D.apply_theme(theme)
        frames = key_frames(world, rounds)
    contact_sheet(frames, out / "design_after.jpg")
    saved = {k: v for k, v in theme.items() if v != D.DEFAULT_THEME.get(k)}
    D.THEME_FILE.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report.append(f"saved {len(saved)} changed values to data/design.json")
    (out / "design_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
