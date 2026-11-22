"""Source of the QorFinder logo (run: python3 logo/gen_logo.py): a free-standing key whose bow is a magnifying
lens, distilled from 许 ("to permit"). Writes the SVGs here and the app's favicon, ui/public/favicon.svg."""
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

HERE = Path(__file__).resolve().parent
FAVICON = HERE.parent / "ui" / "public" / "favicon.svg"
OXBLOOD, INK, ROSE = "#7A1E2C", "#2A1F1D", "#D98A8F"   # mark and "Qor"; "Finder" and dark tiles; mark on dark

def circle(cx, cy, r):
    return f"M{cx - r:.2f} {cy:.2f} a{r} {r} 0 1 0 {2 * r} 0 a{r} {r} 0 1 0 {-2 * r} 0 Z"

# --- the mark, distilled from 许: the dot of 讠 became the lens (the bow), the vertical of 午 the shaft, and the
# two strokes of 午 the bits. The lens has brush contrast: the hole sits up and to the left, so the ring is thin
# at the top left and full at the bottom right, like the "O" of the wordmark; a thin glint marks the glass.
# The key is drawn upright, then tilted to a magnifier's angle, so it reads first as search, then as a key.
BOW = circle(50, 28, 19) + " " + circle(48.2, 26.2, 14.2)
GLINT = "M38.6 27.4 A10 10 0 0 1 46.4 16.4"
SHAFT = "M46.6 45 L53.4 45 L53.0 90.5 A3 3 0 0 1 47.0 90.5 Z"
BITS = "M52 76 L61 76 M52 86 L63.5 86"

def mark(color):
    return (f'<g transform="translate(49.4 49.5) scale(0.95) translate(-50 -50) rotate(-42 50 53)">'
            f'<g fill="{color}"><path d="{BOW}" fill-rule="evenodd"/><path d="{SHAFT}"/></g>'
            f'<g fill="none" stroke="{color}" stroke-linecap="round"><path d="{BITS}" stroke-width="5"/>'
            f'<path d="{GLINT}" stroke-width="1.8"/></g></g>')

# --- wordmark in Fraunces, as outlines ---
fr = instancer.instantiateVariableFont(TTFont(HERE / "fonts" / "Fraunces[SOFT,WONK,opsz,wght].ttf"),
                                       {"wght": 560, "opsz": 72, "SOFT": 30, "WONK": 0})
fgs, fcmap, fhm, FUPM = fr.getGlyphSet(), fr.getBestCmap(), fr["hmtx"], fr["head"].unitsPerEm
def text_path(s, x, baseline, size):
    p = SVGPathPen(fgs); k = size / FUPM
    for ch in s:
        g = fcmap[ord(ch)]; fgs[g].draw(TransformPen(p, (k, 0, 0, -k, x, baseline))); x += fhm[g][0] * k
    return p.getCommands(), x

def svg(w, h, body, extra=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d">'
            f'<title id="t">QorFinder</title><desc id="d">A key inspired by the strokes of 许 (Qor, to permit): '
            f'its bow is a magnifying lens.</desc>{extra}{body}</svg>\n')

def wordmark(x, baseline, size):
    q, x1 = text_path("Qor", x, baseline, size)
    f, x2 = text_path("Finder", x1, baseline, size)
    return f'<path d="{q}" fill="{OXBLOOD}"/><path d="{f}" fill="{INK}"/>', x2

files = {
    "qorfinder-logo-icon.svg": svg(100, 100, mark(OXBLOOD)),
    "qorfinder-logo-icon-mono.svg": svg(100, 100, mark(INK)),
    "qorfinder-logo-icon-reversed.svg": svg(100, 100, mark(ROSE), f'<rect width="100" height="100" rx="18" fill="{INK}"/>'),
}
word, x2 = wordmark(82, 66, 46)
files["qorfinder-logo-horizontal.svg"] = svg(round(x2) + 6, 100, f'<g transform="translate(-14 0)">{mark(OXBLOOD)}</g>{word}')
word, x2 = wordmark(0, 0, 34)
W = round(x2)
files["qorfinder-logo-vertical.svg"] = svg(W, 140, f'<g transform="translate({(W - 100) / 2:.1f} 0)">{mark(OXBLOOD)}</g>'
                                           f'<g transform="translate(0 132)">{word}</g>')
for name, text in files.items():
    (HERE / name).write_text(text)
# The favicon, also the app's header mark, turns rose in dark mode.
FAVICON.write_text(svg(100, 100, mark(OXBLOOD), f'<style>@media (prefers-color-scheme: dark){{[fill="{OXBLOOD}"]'
                                                f'{{fill:{ROSE}}}[stroke="{OXBLOOD}"]{{stroke:{ROSE}}}}}</style>'))
print("wrote", HERE, "and", FAVICON)
