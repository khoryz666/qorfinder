//! `qorfinder serve`: the local API the browser UI talks to, bound to
//! 127.0.0.1 only. It also serves the built UI from `ui/`.
//!
//! Every request must name the host as `127.0.0.1:PORT`, and every API
//! request must carry the random token made at start-up, which only the
//! address the server opens in the browser contains. Another web page, also
//! through DNS rebinding, can therefore neither search the index nor open
//! files (report 3.6, T8).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::extract::{Query, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use qf_core::chunk::ChunkConfig;
use qf_core::embed::Embedder;
use qf_core::index::index_meta;
use qf_core::search::Searcher;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tower_http::services::ServeDir;

use crate::cli::load_embedder;

/// Header the UI sends the token in.
pub const TOKEN_HEADER: &str = "x-qorfinder-token";

/// Where the built UI sits next to the executable in a release, and in
/// the repository in development.
const UI_RELEASE_DIR: &str = "ui";
const UI_REPO_DIR: &str = "ui/dist";

/// What every handler shares.
pub struct App {
    pub token: String,
    pub index_dir: PathBuf,
    pub config: ChunkConfig,
    pub embedder: &'static Embedder,
    /// `None` while there is no usable index.
    pub searcher: Mutex<Option<Searcher<'static>>>,
    /// Why there is no usable index, shown in the UI.
    pub problem: Mutex<Option<String>>,
}

impl App {
    /// Open the index in `index_dir`, or say why it cannot be searched.
    pub fn open_searcher(&self) {
        let meta = index_meta(self.embedder, self.config);
        let mut slot = self.searcher.lock().unwrap();
        *slot = None;
        let opened = Searcher::open(&self.index_dir, self.embedder, &meta);
        *self.problem.lock().unwrap() = opened.as_ref().err().map(|e| e.to_string());
        *slot = opened.ok();
    }
}

/// Start the server on 127.0.0.1 and open the UI in the browser.
pub fn run(port: u16, index_dir: &Path, model_dir: &Path, config: ChunkConfig) -> Result<()> {
    // The server lives as long as the process, so the model can too.
    let embedder: &'static Embedder = Box::leak(Box::new(load_embedder(model_dir)?));
    let app = Arc::new(App {
        token: new_token()?,
        index_dir: index_dir.to_path_buf(),
        config,
        embedder,
        searcher: Mutex::new(None),
        problem: Mutex::new(None),
    });
    app.open_searcher();
    let ui_dir = ui_dir().context("cannot find the built UI (run `npm run build` in ui/)")?;

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
        let port = listener.local_addr()?.port();
        let url = format!("http://127.0.0.1:{port}/?token={}", app.token);
        println!("QorFinder is running at {url}");
        println!("Press Ctrl+C to stop.");
        if let Err(err) = opener::open_browser(&url) {
            eprintln!("warning: could not open the browser ({err}); open the address above");
        }
        let router = router(app, port, &ui_dir);
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    })
}

/// The API and the UI's files.
pub fn router(app: Arc<App>, port: u16, ui_dir: &Path) -> Router {
    let token = app.token.clone();
    let api = Router::new()
        .route("/status", get(status))
        .route("/search", get(search))
        .with_state(app);
    guard(api, token, port, ui_dir)
}

/// `api` under `/api`, open only to requests with `token`, and the UI's
/// files everywhere else; all of it only for requests to
/// `127.0.0.1:port`.
fn guard(api: Router, token: String, port: u16, ui_dir: &Path) -> Router {
    let api = api.route_layer(middleware::from_fn(move |req: Request, next: Next| {
        let ok = header_is(&req, TOKEN_HEADER, &token);
        async move {
            if ok {
                next.run(req).await
            } else {
                error(StatusCode::FORBIDDEN, "missing or wrong token")
            }
        }
    }));
    let host = format!("127.0.0.1:{port}");
    Router::new()
        .nest("/api", api)
        .fallback_service(ServeDir::new(ui_dir))
        .layer(middleware::from_fn(move |req: Request, next: Next| {
            let ok = header_is(&req, header::HOST.as_str(), &host);
            async move {
                if ok {
                    next.run(req).await
                } else {
                    error(StatusCode::FORBIDDEN, "wrong host")
                }
            }
        }))
}

fn header_is(req: &Request, name: &str, value: &str) -> bool {
    req.headers()
        .get(name)
        .is_some_and(|v| v.as_bytes() == value.as_bytes())
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": message.into() }))).into_response()
}

/// 32 random hex digits.
fn new_token() -> Result<String> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).map_err(|e| anyhow::anyhow!("no random numbers: {e}"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// The built UI next to the executable, or in the repository.
fn ui_dir() -> Option<PathBuf> {
    let beside_exe = std::env::current_exe()
        .ok()
        .and_then(|exe| Some(exe.parent()?.join(UI_RELEASE_DIR)));
    beside_exe
        .into_iter()
        .chain([PathBuf::from(UI_REPO_DIR)])
        .find(|dir| dir.join("index.html").is_file())
}

#[derive(Serialize)]
struct Status {
    index_dir: String,
    ready: bool,
    documents: usize,
    chunks: usize,
    source_folder: Option<String>,
    problem: Option<String>,
}

async fn status(State(app): State<Arc<App>>) -> Response {
    let slot = app.searcher.lock().unwrap();
    let (documents, chunks, source_folder) = match &*slot {
        Some(s) => (
            s.store().documents().map(|d| d.len()).unwrap_or(0),
            s.store().chunk_count().unwrap_or(0),
            s.store().source_folder().ok().flatten(),
        ),
        None => (0, 0, None),
    };
    Json(Status {
        index_dir: app.index_dir.display().to_string(),
        ready: slot.is_some(),
        documents,
        chunks,
        source_folder: source_folder.map(|f| f.display().to_string()),
        problem: app.problem.lock().unwrap().clone(),
    })
    .into_response()
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    #[serde(default = "default_k")]
    k: usize,
}

fn default_k() -> usize {
    10
}

#[derive(Serialize)]
struct ResultItem {
    chunk_id: u32,
    score: f32,
    file: String,
    folder: String,
    page: Option<u32>,
    char_start: usize,
    char_end: usize,
    text: String,
}

async fn search(State(app): State<Arc<App>>, Query(params): Query<SearchParams>) -> Response {
    let query = params.q.trim().to_string();
    if query.is_empty() {
        return error(StatusCode::BAD_REQUEST, "type something to search for");
    }
    let k = params.k.clamp(1, 100);
    let result = tokio::task::spawn_blocking(move || {
        let slot = app.searcher.lock().unwrap();
        let Some(searcher) = &*slot else {
            return Err((
                StatusCode::CONFLICT,
                "no index yet: choose a folder to index".to_string(),
            ));
        };
        let result = searcher
            .search(&query, k)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let mut items = Vec::with_capacity(result.hits.len());
        for hit in &result.hits {
            let lookup = || -> Result<_, qf_core::store::StoreError> {
                let chunk = searcher.store().chunk(hit.chunk_id)?;
                let doc = match &chunk {
                    Some(c) => searcher.store().document(c.doc_id)?,
                    None => None,
                };
                Ok(chunk.zip(doc))
            };
            let (chunk, doc) = lookup()
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
                .ok_or((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "index is incomplete".to_string(),
                ))?;
            items.push(ResultItem {
                chunk_id: hit.chunk_id,
                score: hit.score,
                file: doc
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                folder: doc
                    .path
                    .parent()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default(),
                page: chunk.page,
                char_start: chunk.char_start,
                char_end: chunk.char_end,
                text: chunk.text,
            });
        }
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        Ok(json!({
            "results": items,
            "encode_ms": ms(result.encode_time),
            "scan_ms": ms(result.scan_time),
        }))
    })
    .await;
    match result {
        Ok(Ok(body)) => Json(body).into_response(),
        Ok(Err((status, message))) => error(status, message),
        Err(err) => error(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use tower::ServiceExt;

    use super::*;

    const PORT: u16 = 4321;

    fn app(ui_dir: &Path) -> Router {
        let api = Router::new().route("/ping", get(|| async { "pong" }));
        guard(api, "secret".into(), PORT, ui_dir)
    }

    async fn status(
        ui_dir: &Path,
        path: &str,
        host: Option<&str>,
        token: Option<&str>,
    ) -> StatusCode {
        let mut req = Request::builder().uri(path);
        if let Some(host) = host {
            req = req.header(header::HOST, host);
        }
        if let Some(token) = token {
            req = req.header(TOKEN_HEADER, token);
        }
        let req = req.body(Body::empty()).unwrap();
        app(ui_dir).oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn api_needs_the_token_and_the_exact_host() {
        let ui = tempfile::tempdir().unwrap();
        let good = Some("127.0.0.1:4321");
        assert_eq!(
            status(ui.path(), "/api/ping", good, Some("secret")).await,
            StatusCode::OK
        );
        assert_eq!(
            status(ui.path(), "/api/ping", good, None).await,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            status(ui.path(), "/api/ping", good, Some("guess")).await,
            StatusCode::FORBIDDEN
        );
        for host in [
            "localhost:4321",
            "127.0.0.1",
            "127.0.0.1:80",
            "evil.example:4321",
        ] {
            assert_eq!(
                status(ui.path(), "/api/ping", Some(host), Some("secret")).await,
                StatusCode::FORBIDDEN,
                "{host}"
            );
        }
        assert_eq!(
            status(ui.path(), "/api/ping", None, Some("secret")).await,
            StatusCode::FORBIDDEN
        );
    }

    #[tokio::test]
    async fn ui_files_need_the_host_but_no_token() {
        let ui = tempfile::tempdir().unwrap();
        std::fs::write(ui.path().join("index.html"), "<p>hi</p>").unwrap();
        let good = Some("127.0.0.1:4321");
        assert_eq!(status(ui.path(), "/", good, None).await, StatusCode::OK);
        assert_eq!(
            status(ui.path(), "/", Some("localhost:4321"), None).await,
            StatusCode::FORBIDDEN
        );
    }

    #[test]
    fn tokens_are_32_hex_digits_and_differ() {
        let (a, b) = (new_token().unwrap(), new_token().unwrap());
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }
}
