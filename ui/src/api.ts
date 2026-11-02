// Calls to the local QorFinder API. Every call carries the token the
// server put in the address it opened; without it the server refuses.

const TOKEN_KEY = "qorfinder-token";

/** The token from ?token=..., kept for this tab and removed from the address bar. */
function readToken(): string {
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get("token");
  if (fromUrl) {
    sessionStorage.setItem(TOKEN_KEY, fromUrl);
    url.searchParams.delete("token");
    window.history.replaceState(null, "", url.toString());
    return fromUrl;
  }
  return sessionStorage.getItem(TOKEN_KEY) ?? "";
}

const token = readToken();

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method,
    headers: {
      "x-qorfinder-token": token,
      ...(body === undefined ? {} : { "content-type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error ?? `${response.status} ${response.statusText}`);
  }
  if (response.status === 202 || response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export interface Status {
  index_dir: string;
  ready: boolean;
  documents: number;
  chunks: number;
  source_folder: string | null;
  problem: string | null;
}

export interface Result {
  chunk_id: number;
  score: number;
  file: string;
  folder: string;
  page: number | null;
  char_start: number;
  char_end: number;
  text: string;
}

export interface SearchResponse {
  results: Result[];
  encode_ms: number;
  scan_ms: number;
}

export interface Job {
  running: boolean;
  folder: string | null;
  stage: "reading" | "embedding" | null;
  done: number;
  total: number;
  elapsed_s: number;
  message: string | null;
  failed: boolean;
  skipped: string[];
}

export const api = {
  hasToken: () => token !== "",
  status: () => call<Status>("GET", "/status"),
  search: (q: string, k = 10) =>
    call<SearchResponse>("GET", `/search?q=${encodeURIComponent(q)}&k=${k}`),
  index: (folder: string | null) => call<void>("POST", "/index", { folder }),
  indexStatus: () => call<Job>("GET", "/index/status"),
  pickFolder: () => call<{ folder: string | null }>("POST", "/folders/pick"),
  open: (chunkId: number) => call<void>("POST", "/open", { chunk_id: chunkId }),
};
