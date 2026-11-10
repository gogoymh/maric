"""Render an HTML file to PNG at twice its logical size, after its web fonts have loaded.

Usage: python docs/assets/src/render.py docs/assets/src/hero.html docs/assets/maric-hero.png 1280 460
Set PLAYWRIGHT_CHROMIUM_EXECUTABLE to use an installed Chrome instead of Playwright's Chromium.
"""
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

src, out, width, height = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE"))
    page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
    page.goto(Path(src).resolve().as_uri(), wait_until="networkidle")
    page.evaluate("document.fonts.ready")
    family = page.evaluate("getComputedStyle(document.querySelector('h1')).fontFamily")
    loaded = page.evaluate("document.fonts.check('700 44px Pretendard')")
    if not loaded:
        sys.exit(f"Pretendard did not load (font-family: {family}); check the network.")
    page.screenshot(path=out, full_page=False)
    browser.close()
print(out)
