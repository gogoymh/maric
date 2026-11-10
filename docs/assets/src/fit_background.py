"""Shrink the objects of the hero background toward the bottom-right corner, keeping the canvas size.

The generated image places its objects from 44% of the width, which is under the title. This keeps
the canvas and scales the part that holds the objects to SCALE, anchored bottom-right; the empty
left and top bands (smooth gradient only) are stretched to fill the space that opens up.

Usage: python docs/assets/src/fit_background.py <generated image> docs/assets/src/hero-background.jpg
"""
import sys

from PIL import Image

SCALE = 0.88        # size of the objects relative to the generated image
LEFT_BAND = 0.35    # fraction of the width, from the left, that holds no object
TOP_BAND = 0.25     # fraction of the height, from the top, that holds no object

src, out = sys.argv[1], sys.argv[2]
image = Image.open(src).convert("RGB")
width, height = image.size
x_cut, y_cut = round(width * LEFT_BAND), round(height * TOP_BAND)
new_x = width - round((width - x_cut) * SCALE)
new_y = height - round((height - y_cut) * SCALE)

canvas = Image.new("RGB", (width, height))
for (x0, x1, nx0, nx1) in ((0, x_cut, 0, new_x), (x_cut, width, new_x, width)):
    for (y0, y1, ny0, ny1) in ((0, y_cut, 0, new_y), (y_cut, height, new_y, height)):
        part = image.crop((x0, y0, x1, y1)).resize((nx1 - nx0, ny1 - ny0), Image.LANCZOS)
        canvas.paste(part, (nx0, ny0))
canvas.save(out, quality=92)
print(out, canvas.size)
