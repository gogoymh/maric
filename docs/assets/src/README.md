# README images

- `maric-hero.png` is `hero.html` rendered at twice its logical size (2560 x 920). The page sets the
  title in [Pretendard](https://github.com/orioncactus/pretendard) (SIL Open Font License, loaded from
  jsDelivr) over `hero-background.jpg`. Render it with:

  ```bash
  PLAYWRIGHT_CHROMIUM_EXECUTABLE="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    uv run --no-project --with playwright python docs/assets/src/render.py docs/assets/src/hero.html docs/assets/maric-hero.png 1280 460
  ```

  `render.py` waits for the fonts and stops if Pretendard did not load.
- `hero-background.jpg` is an AI-generated image supplied by the authors (2089 x 753), passed through
  `fit_background.py`. The script keeps the canvas size and scales the objects to 88%, anchored
  bottom-right, so that they clear the title; the empty left and top bands are stretched to fill the rest.

  ```bash
  python docs/assets/src/fit_background.py <generated image> docs/assets/src/hero-background.jpg
  ```

- `star.gif` is `star.html` rendered at twice its logical size (1280 x 320, 4 s, 25 frames per second).
  The page draws each frame with `render(t)`; `render_gif.py` captures the frames and builds the GIF with ffmpeg:

  ```bash
  PLAYWRIGHT_CHROMIUM_EXECUTABLE="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    uv run --no-project --with playwright python docs/assets/src/render_gif.py docs/assets/src/star.html docs/assets/star.gif 640 160 4.0 25
  ```

- `maric-figure1.png` and `maric-figure2.png` are Figures 1 and 2 of the paper, rendered at 1,600 px
  from the figure files in the arXiv source of v2
  (`pdftoppm -png -singlefile -scale-to-x 1600 -scale-to-y -1 fig1.pdf maric-figure1`).

The hero carries no result numbers; results appear only in the README's Results section.
