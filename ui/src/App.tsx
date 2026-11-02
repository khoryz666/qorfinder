import { useCallback, useEffect, useState } from "react";
import { api, type Job, type SearchResponse, type Status } from "./api";

type View = "search" | "folder" | "about";

export default function App() {
  const [view, setView] = useState<View>("search");
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api
      .status()
      .then((s) => {
        setStatus(s);
        setError(null);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(refresh, [refresh]);

  // Send a first-time user to the folder page.
  useEffect(() => {
    if (status && !status.ready) setView((v) => (v === "search" ? "folder" : v));
  }, [status]);

  if (!api.hasToken()) {
    return (
      <main className="page">
        <p className="notice error">
          Open QorFinder with the address <code>qorfinder serve</code> printed; it holds the
          key this page needs.
        </p>
      </main>
    );
  }

  return (
    <>
      <header className="bar">
        <h1>QorFinder</h1>
        <nav>
          {(["search", "folder", "about"] as View[]).map((v) => (
            <button
              key={v}
              className={view === v ? "tab active" : "tab"}
              onClick={() => setView(v)}
            >
              {v === "search" ? "Search" : v === "folder" ? "Documents" : "About"}
            </button>
          ))}
        </nav>
      </header>
      <main className="page">
        {error && <p className="notice error">{error}</p>}
        {view === "search" && <SearchView status={status} />}
        {view === "folder" && <FolderView status={status} onIndexed={refresh} />}
        {view === "about" && <AboutView status={status} />}
      </main>
    </>
  );
}

function SearchView({ status }: { status: Status | null }) {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [searched, setSearched] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function search(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setResponse(await api.search(query));
      setSearched(query);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function open(chunkId: number) {
    try {
      await api.open(chunkId);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <section>
      <form className="search" onSubmit={search}>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Describe the clause you need, e.g. a promise not to poach staff"
          aria-label="Search query"
          autoFocus
        />
        <button type="submit" className="primary" disabled={busy || !query.trim()}>
          {busy ? "Searching…" : "Search"}
        </button>
      </form>
      {status?.ready && (
        <p className="muted">
          Searching {status.documents} document(s), {status.chunks} passage(s), by meaning.
        </p>
      )}
      {error && <p className="notice error">{error}</p>}
      {response && (
        <>
          <p className="muted">
            {response.results.length} result(s) for “{searched}” in{" "}
            {(response.encode_ms + response.scan_ms).toFixed(0)} ms
          </p>
          <ol className="results">
            {response.results.map((r) => (
              <li key={r.chunk_id} className="result">
                <div className="result-head">
                  <span className="file">{r.file}</span>
                  <span className="score" title="Similarity to the query (cosine)">
                    {r.score.toFixed(3)}
                  </span>
                </div>
                <div className="place">
                  {r.folder}
                  {r.page !== null && ` · page ${r.page}`} · characters {r.char_start}–{r.char_end}
                </div>
                <p className="passage">{r.text}</p>
                <button onClick={() => open(r.chunk_id)}>Open file</button>
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}

function FolderView({ status, onIndexed }: { status: Status | null; onIndexed: () => void }) {
  const [folder, setFolder] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Poll the run while it lasts.
  useEffect(() => {
    let stop = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const j = await api.indexStatus();
        if (stop) return;
        setJob(j);
        if (j.running) {
          timer = window.setTimeout(poll, 500);
        } else if (j.message) {
          onIndexed();
        }
      } catch (err) {
        if (!stop) setError((err as Error).message);
      }
    };
    poll();
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
    // Restart polling whenever a run is started.
  }, [job?.running, onIndexed]);

  async function pick() {
    setError(null);
    try {
      const { folder } = await api.pickFolder();
      if (folder) setFolder(folder);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function start(target: string | null) {
    setError(null);
    try {
      await api.index(target);
      setJob(await api.indexStatus());
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const running = job?.running ?? false;

  return (
    <section>
      <h2>Documents</h2>
      {status?.ready ? (
        <p>
          The index holds <strong>{status.documents}</strong> document(s) in{" "}
          <strong>{status.chunks}</strong> passage(s)
          {status.source_folder && (
            <>
              , from <code>{status.source_folder}</code>
            </>
          )}
          .
        </p>
      ) : (
        <p className="notice">
          No documents are indexed yet. Choose the folder with your PDF, Word (.docx) and text
          files; QorFinder reads every file in it and its subfolders. Nothing leaves this
          computer.
        </p>
      )}

      <div className="row">
        <button onClick={pick} disabled={running}>
          Choose folder…
        </button>
        <input
          type="text"
          value={folder}
          onChange={(e) => setFolder(e.target.value)}
          placeholder="or type a folder path"
          aria-label="Folder to index"
          disabled={running}
        />
        <button
          className="primary"
          onClick={() => start(folder)}
          disabled={running || !folder.trim()}
        >
          Index this folder
        </button>
      </div>
      {status?.source_folder && (
        <p>
          <button onClick={() => start(null)} disabled={running}>
            Rebuild the index
          </button>{" "}
          <span className="muted">reads {status.source_folder} again</span>
        </p>
      )}
      {error && <p className="notice error">{error}</p>}

      {job?.running && <JobProgress job={job} />}
      {job && !job.running && job.message && (
        <div className={job.failed ? "notice error" : "notice ok"}>
          <p>{job.message}</p>
          {job.skipped.length > 0 && (
            <details>
              <summary>{job.skipped.length} file(s) skipped</summary>
              <ul className="skipped">
                {job.skipped.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
      <p className="muted">
        Index folder: <code>{status?.index_dir}</code>
      </p>
    </section>
  );
}

/** Progress of a run: files read, then passages embedded. Elapsed time
 * rather than an estimate, since passages are embedded shortest first. */
function JobProgress({ job }: { job: Job }) {
  const label = job.stage === "embedding" ? "Understanding passages" : "Reading files";
  const unit = job.stage === "embedding" ? "passages" : "files";
  const share = job.total > 0 ? job.done / job.total : 0;
  return (
    <div className="progress-box" aria-live="polite">
      <p>
        Indexing <code>{job.folder}</code>
      </p>
      <p>
        {job.stage ? label : "Starting"}: {job.done} of {job.total} {unit} ·{" "}
        {Math.round(job.elapsed_s)} s elapsed
      </p>
      <progress value={share} max={1} />
    </div>
  );
}

function AboutView({ status }: { status: Status | null }) {
  return (
    <section className="about">
      <h2>About QorFinder</h2>
      <p>
        QorFinder finds passages in your legal documents by what they mean, not only by the
        words they use. Ask for “a promise not to poach staff” and it can find a
        non-solicitation clause that never uses those words.
      </p>
      <p>
        Everything runs on this computer. Your documents, the index and your searches never
        leave it: the page talks only to the QorFinder program at 127.0.0.1.
      </p>
      <ul>
        <li>Reads PDF (with a text layer), Word (.docx) and plain text files.</li>
        <li>
          Splits each document into passages along paragraphs and numbered clauses, and turns
          each into a vector with the all-MiniLM-L6-v2 language model.
        </li>
        <li>Scanned PDFs without a text layer are skipped and listed.</li>
      </ul>
      {status && (
        <p className="muted">
          Index folder: <code>{status.index_dir}</code>
        </p>
      )}
    </section>
  );
}
