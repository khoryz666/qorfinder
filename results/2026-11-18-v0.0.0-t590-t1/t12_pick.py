import json, sys, urllib.parse, urllib.request
qfile, out = sys.argv[1], sys.argv[2]  # usage: t12_pick.py queries.jsonl out.txt <token>
U = "http://127.0.0.1:58545/api/search"  # the port printed by serve
H = {"x-qorfinder-token": sys.argv[3]}  # the token printed by serve
qs = [json.loads(l) for l in open(qfile, encoding="utf-8") if l.strip()]
picked, seen = [], set()
for q in qs:
    text = q.get("query") or q.get("text")
    req = urllib.request.Request(U + "?" + urllib.parse.urlencode({"q": text, "k": 10}), headers=H)
    res = json.load(urllib.request.urlopen(req))["results"]
    for rank, r in enumerate(res, 1):
        key = (r["file"], r["char_start"])
        if r["page"] is not None and key not in seen:
            seen.add(key); picked.append((q.get("id") or q.get("qid"), text, rank, r)); break
    if len(picked) == 20: break
with open(out, "w", encoding="utf-8") as f:
    f.write("T12: 20 PDF results from the test queries over data/t1-mixed (release v0.0.0, Windows 10)\n")
    f.write("For each, Open file in the UI, go to the page, and check the passage starts there. Mark ok or wrong.\n\n")
    for i, (qid, text, rank, r) in enumerate(picked, 1):
        words = " ".join(r["text"].split()[:12])
        f.write(f"{i:2}. [{qid}] rank {rank}: \"{text}\"\n    {r['file']} · page {r['page']} · characters {r['char_start']}-{r['char_end']}\n    starts: \"{words} ...\"\n    check: \n\n")
print(len(picked))
