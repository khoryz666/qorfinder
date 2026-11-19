import re, sys
sys.path.insert(0, sys.argv[3])
from pypdf import PdfReader
lst, folder = sys.argv[1], sys.argv[2]
norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
txt = open(lst, encoding="utf-8").read()
items = re.findall(r'\n\s*(\d+)\. .*?\n    (.+?) · page (\d+) · characters (\d+)-(\d+)\n    starts: "(.+?) \.\.\."', txt)
cache = {}
for n, f, page, a, b, start in items:
    if f not in cache: cache[f] = [norm(p.extract_text() or "") for p in PdfReader(f"{folder}/{f}").pages]
    pages = cache[f]; probe = norm(" ".join(start.split()[:8]))
    hits = [i + 1 for i, p in enumerate(pages) if probe in p]
    print(f"{n:>2}. reported page {page:>2}  found on {hits or 'none'}  {'ok' if int(page) in hits else 'CHECK'}")
