"""Render the replay to an MP4: every frame is replay.html drawn at time t, screenshotted.

    python render.py timeline.json out.mp4 [--fps 30] [--until SECONDS]
"""
import argparse
import json
import os
import subprocess

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--until", type=float)
    args = ap.parse_args()
    tl = json.load(open(args.timeline))
    end = min(tl["end"], args.until) if args.until else tl["end"]
    n = int(end * args.fps)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(args.fps),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                           args.out], stdin=subprocess.PIPE)
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        page = br.new_page(viewport={"width": 1920, "height": 1080})
        page.goto("file://" + os.path.join(HERE, "replay.html"))
        page.evaluate("tl => load(tl)", tl)
        for f in range(n):
            page.evaluate(f"render({f / args.fps})")
            ff.stdin.write(page.screenshot(type="png"))
            if f % (args.fps * 10) == 0:
                print(f"  {f / args.fps:.0f}s / {end:.0f}s", flush=True)
        br.close()
    ff.stdin.close()
    ff.wait()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
