# QorFinder logo

A free-standing key whose bow is a magnifying lens, tilted to a magnifier's angle so it reads first as search,
then as a key. It is distilled from 许 (Qor), which means "to permit": the dot of 讠 became the lens, the
vertical of 午 the shaft, and the two strokes of 午 the bits.

- Permission: a key, and 许 itself.
- Finder: the bow is a lens, drawn with brush contrast (thin at the top left, full at the bottom right) and a
  glint on the glass.
- Private: a key also says the documents stay locked on the lawyer's own laptop.
- Drawn by hand as: a circle, a diagonal line, and two short ticks.

## Files

| File | Use |
|---|---|
| `qorfinder-logo-icon.svg` | the mark, for light backgrounds |
| `qorfinder-logo-icon.png` | the mark as a 600 px PNG with a transparent background, used on the poster |
| `qorfinder-logo-icon-mono.svg` | one colour (ink), for print and stamps |
| `qorfinder-logo-icon-reversed.svg` | the rose mark on an ink tile, for dark backgrounds and the app icon |
| `qorfinder-logo-horizontal.svg` | mark + wordmark, for headers |
| `qorfinder-logo-vertical.svg` | mark above the wordmark, for covers and splash screens |
| `gen_logo.py` | the source: writes every SVG here and `ui/public/favicon.svg` (the app's favicon and header mark) |
| `logo_sheet.html` | the review sheet, source of `logo_sheet.png` |
| `fonts/` | Fraunces, for the wordmark (SIL Open Font License, `OFL-Fraunces.txt`) |

Rebuild, from the repo root (needs `fonttools`, and Google Chrome for the PNGs):

    python3 logo/gen_logo.py
    cd logo && google-chrome --headless=new --hide-scrollbars --default-background-color=00000000 \
      --screenshot=$PWD/qorfinder-logo-icon.png --window-size=600,600 file://$PWD/qorfinder-logo-icon.svg
    google-chrome --headless=new --hide-scrollbars --screenshot=$PWD/logo_sheet.png \
      --window-size=1200,720 file://$PWD/logo_sheet.html

The wordmark is Fraunces converted to outlines, so the SVGs need no installed font.

## Colours

One palette for the logo, the app and the poster:

- Oxblood `#7A1E2C`, the colour of leather-bound law reports: the mark, "Qor", and the app's accent.
- Ink `#2A1F1D`, a warm near-black: "Finder", body text and dark tiles.
- Rose `#D98A8F`: the mark and the accent on dark backgrounds (the favicon switches in dark mode).

## Usage

- Clear space: at least a quarter of the mark's width on every side.
- Minimum size: 16 px on screen (the glint shows from 32 px); 10 mm in print.
- Keep the tilt; do not straighten, stretch, recolour outside the palette, or add shadows.
