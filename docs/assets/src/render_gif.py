"""Render an HTML animation to a looping GIF at twice its logical size.

The page must define window.render(t), which draws the frame at t seconds.
Usage: python docs/assets/src/render_gif.py docs/assets/src/star.html docs/assets/star.gif 640 160 4.0 25
Needs ffmpeg on PATH. Set PLAYWRIGHT_CHROMIUM_EXECUTABLE to use an installed Chrome.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

src, out = sys.argv[1], sys.argv[2]
width, height, seconds, fps = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
frames = round(seconds * fps)

with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE"))
    page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
    page.goto(Path(src).resolve().as_uri(), wait_until="networkidle")
    page.evaluate("document.fonts.ready")
    if not page.evaluate("document.fonts.check('800 28px Pretendard')"):
        sys.exit("Pretendard did not load; check the network.")
    for i in range(frames):
        page.evaluate(f"render({i / fps})")
        page.screenshot(path=f"{tmp}/f{i:04d}.png")
    browser.close()
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", f"{tmp}/f%04d.png",
        "-vf", "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];"
               "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
        "-loop", "0", out,
    ], check=True)
print(out)
