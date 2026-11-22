// The landing page's animations: the demo search, scroll reveals, counting
// stats and growing bars. With reduced motion, everything shows in its final state.

document.documentElement.classList.add("js");
const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Real output of `qorfinder search` with the default (fine-tuned) model over
// the 30-contract CUAD sample (top 3). [[...]] marks the words highlighted.
const DEMO = [
  {
    query: "must not try to recruit or lure away the other company's staff",
    meta: "3 results in 41.6 ms",
    results: [
      ["0.554", "CORALGOLDRESOURCES,LTD_05_28_2020-EX-4.1-CONSULTING AGREEMENT.txt",
        "7.1 The Consultant covenants, undertakes and agrees with the Company that [[during the Term and for a period of one year]] from the date of expiration or termination of this Agreement for any reason whatsoever, [[it shall not, on its own behalf or on behalf of any person]], whether…"],
      ["0.553", "ROCKYMOUNTAINCHOCOLATEFACTORY,INC_12_23_2019-EX-10.2-STRATEGIC ALLIANCE AGREEMENT.txt",
        "…authorized to make such statement or announcement on behalf of Farids or EA, as applicable. [[Section 8.9 Non-Solicitation.]] (a) Each of EA, Farids and the Company shall not, and shall cause their respective…"],
      ["0.404", "ROCKYMOUNTAINCHOCOLATEFACTORY,INC_12_23_2019-EX-10.2-STRATEGIC ALLIANCE AGREEMENT.txt",
        "Affiliates from (i) responding to any statement made by the Company or any of its Affiliates or representatives that was made in breach of Section 8.8(a) above or (ii) if solicited by a Third Party…"],
    ],
  },
  {
    query: "is there a limit on how much either side has to pay for damages?",
    meta: "3 results in 43.9 ms",
    results: [
      ["0.548", "LegacyEducationAllianceInc_20200330_10-K_EX-10.18_12090678_EX-10.18_Development Agreement.txt",
        "…contractual liability and personal and advertising injury liability insurance in an amount [[no less than ten million dollars ($10,000,000.00) per occurrence]] and ten million dollars ($10,000,000.00) aggregate combined single limit…"],
      ["0.547", "UAGHINC_04_14_2004-EX-10.18-MAINTENANCE AGREEMENT.txt",
        "OR DEATH TO ANY PERSON PROXIMATELY CAUSED BY A PARTY'S NEGLIGENCE, [[THE AMOUNT OF DIRECT DAMAGES PROVEN]]; (b) FOR LOSS OR DAMAGE TO REAL PROPERTY OR TANGIBLE PERSONAL PROPERTY PROXIMATELY CAUSED BY A PARTY'S NEGLIGENCE, THE AMOUNT OF DIRECT DAMAGES PROVEN…"],
      ["0.533", "WaterNowInc_20191120_10-Q_EX-10.12_11900227_EX-10.12_Distributor Agreement.txt",
        "…the alternative dispute resolution rules of the American Arbitration Association. The arbitration proceeding shall be held before a single arbitrator unless (i) the matter in controversy exceeds Five Hundred Thousand Dollars ($500,000)…"],
    ],
  },
];

const typed = document.getElementById("typed");
const meta = document.getElementById("meta");
const list = document.getElementById("results");

function card([score, file, text]) {
  const li = document.createElement("li");
  const head = document.createElement("div");
  head.className = "head";
  head.innerHTML = `<span class="score"></span><span class="file"></span>`;
  head.children[0].textContent = score;
  head.children[1].textContent = file;
  const p = document.createElement("p");
  // Alternate plain text and highlighted spans; textContent keeps it safe.
  text.split(/\[\[|\]\]/).forEach((part, i) => {
    const node = i % 2 ? document.createElement("mark") : document.createTextNode("");
    node.textContent = part;
    p.append(node);
  });
  li.append(head, p);
  return li;
}

async function demo() {
  if (still) {
    typed.textContent = DEMO[0].query;
    meta.textContent = DEMO[0].meta;
    DEMO[0].results.forEach((r) => { const li = card(r); li.classList.add("in"); list.append(li); });
    return;
  }
  for (let i = 0; ; i = (i + 1) % DEMO.length) {
    const d = DEMO[i];
    for (const ch of d.query) { typed.textContent += ch; await sleep(38 + Math.random() * 40); }
    await sleep(450);
    meta.textContent = d.meta;
    const items = d.results.map(card);
    list.append(...items);
    for (const li of items) { await sleep(160); li.classList.add("in"); }
    await sleep(5200);
    items.forEach((li) => li.classList.remove("in"));
    await sleep(500);
    list.replaceChildren();
    meta.innerHTML = "&nbsp;";
    while (typed.textContent) { typed.textContent = typed.textContent.slice(0, -2); await sleep(14); }
    await sleep(500);
  }
}

// Count a stat up from 0 to its value.
function count(el) {
  const end = Number(el.dataset.count), unit = el.dataset.unit, start = performance.now();
  const step = (now) => {
    const t = Math.min((now - start) / 1200, 1);
    el.textContent = Math.round(end * (1 - (1 - t) ** 3)) + unit;
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

const seen = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    e.target.classList.add("in");
    const n = e.target.querySelector("[data-count]");
    if (n && !still) count(n);
    if (e.target.closest(".pipeline")) e.target.closest(".pipeline").classList.add("done");
    if (e.target.classList.contains("chart")) e.target.classList.add("grown");
    seen.unobserve(e.target);
  }
}, { threshold: 0.15 });
document.querySelectorAll(".reveal").forEach((el) => seen.observe(el));

demo();
