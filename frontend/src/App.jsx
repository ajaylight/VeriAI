import { useEffect, useMemo, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

const examples = [
  {
    label: "Supported claim",
    claim: "Magic Johnson played for the Lakers.",
    tone: "supports",
  },
  {
    label: "Contradicted claim",
    claim: "Saturn is a moon.",
    tone: "refutes",
  },
  {
    label: "Insufficient evidence",
    claim: "Magic Johnson's favorite color is purple.",
    tone: "neutral",
  },
];

const verdictCopy = {
  SUPPORTS: {
    title: "Supported by the retrieved evidence",
    description: "The model found evidence that entails the submitted claim.",
    tone: "supports",
  },
  REFUTES: {
    title: "Contradicted by the retrieved evidence",
    description: "The model found evidence that contradicts the submitted claim.",
    tone: "refutes",
  },
  "NOT ENOUGH INFO": {
    title: "Not enough information",
    description: "The retrieved evidence does not establish whether the claim is true or false.",
    tone: "neutral",
  },
};

function readablePageId(pageId = "") {
  return pageId
    .replace(/_-LRB-/g, " (")
    .replace(/-RRB-/g, ")")
    .replace(/_/g, " ")
    .replace(/-COLON-/g, ":");
}

function formatEvidenceText(text = "") {
  return String(text)
    .split("\\t", 1)[0]
    .replace(/-LRB-/g, "(")
    .replace(/-RRB-/g, ")")
    .replace(/-LSB-/g, "[")
    .replace(/-RSB-/g, "]")
    .replace(/-LCB-/g, "{")
    .replace(/-RCB-/g, "}")
    .replace(/-COLON-/g, ":")
    .replace(/``|''/g, '"')
    .replace(/"\\s+([^"]*?)\\s+"/g, '"$1"')
    .replace(/\\s+([,.;:!?])/g, "$1")
    .replace(/([([{])\\s+/g, "$1")
    .replace(/\\s+([)\\]}])/g, "$1")
    .replace(/([)\\]}])(?=[^\\s,.;:!?])/g, "$1 ")
    .replace(/\\s{2,}/g, " ")
    .trim();
}

function formatRelation(relation = "neutral") {
  return relation.replaceAll("_", " ");
}

function StatusPill({ state, children }) {
  return (
    <span className={`status-pill ${state || "muted"}`}>
      <span className="status-dot" />
      {children}
    </span>
  );
}

function EvidenceCard({ item, index }) {
  const relation = item.predicted_relation || "neutral";
  return (
    <article className="evidence-card">
      <div className="evidence-topline">
        <span className="evidence-number">Evidence {index + 1}</span>
        <span className={`relation-tag ${relation}`}>{formatRelation(relation)}</span>
      </div>
      <p className="evidence-text">{formatEvidenceText(item.text)}</p>
      <div className="evidence-meta">
        <span>{readablePageId(item.page_id)}</span>
        <span>Sentence {item.sentence_id}</span>
        {item.verdict_eligible === false && (
          <span className="excluded-note">Not used for final verdict</span>
        )}
      </div>
    </article>
  );
}

export default function App() {
  const [claim, setClaim] = useState("");
  const [topK, setTopK] = useState("5");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiState, setApiState] = useState("checking");
  const [modelState, setModelState] = useState("checking");

  useEffect(() => {
    let active = true;

    async function checkServices() {
      try {
        const healthResponse = await fetch(`${API_BASE}/health`);
        if (!healthResponse.ok) throw new Error("Backend health check failed");
        if (active) setApiState("online");
      } catch {
        if (active) {
          setApiState("offline");
          setModelState("unknown");
        }
        return;
      }

      try {
        const modelResponse = await fetch(`${API_BASE}/verification/status`);
        if (!modelResponse.ok) throw new Error("Model status check failed");
        const model = await modelResponse.json();
        if (active) setModelState(model.loaded ? "loaded" : "not-loaded");
      } catch {
        if (active) setModelState("unknown");
      }
    }

    checkServices();
    return () => {
      active = false;
    };
  }, []);

  const verdict = result?.verdict;
  const verdictInfo = verdict ? verdictCopy[verdict] : null;
  const evidence = useMemo(
    () => (Array.isArray(result?.evidence_scores) ? result.evidence_scores : []),
    [result],
  );
  const score = typeof result?.confidence === "number"
    ? `${(result.confidence * 100).toFixed(1)}%`
    : "—";

  async function handleVerify(event) {
    event.preventDefault();
    const cleanClaim = claim.trim();
    if (!cleanClaim) {
      setError("Enter a claim to verify.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/verification/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim: cleanClaim, top_k: Number(topK) }),
      });
      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        const detail = typeof payload.detail === "string"
          ? payload.detail
          : `Verification failed (HTTP ${response.status}).`;
        throw new Error(detail);
      }

      setResult(payload);
    } catch (requestError) {
      setError(
        requestError instanceof TypeError
          ? "Could not reach the VeriAI API. Check that the backend is running at " + API_BASE + "."
          : requestError.message || "An unexpected error occurred.",
      );
    } finally {
      setLoading(false);
    }
  }

  function chooseExample(example) {
    setClaim(example.claim);
    setResult(null);
    setError("");
  }

  function clearForm() {
    setClaim("");
    setResult(null);
    setError("");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="VeriAI home">
          <span className="brand-mark">V</span>
          <span>Veri<span className="brand-accent">AI</span></span>
        </a>
        <div className="service-status" aria-live="polite">
          <StatusPill state={apiState === "online" ? "online" : apiState === "offline" ? "offline" : "muted"}>
            API {apiState === "checking" ? "checking" : apiState}
          </StatusPill>
          <StatusPill state={modelState === "loaded" ? "online" : modelState === "not-loaded" ? "muted" : modelState === "unknown" ? "offline" : "muted"}>
            Model {modelState === "checking" ? "checking" : modelState === "loaded" ? "loaded" : modelState === "not-loaded" ? "not loaded" : modelState}
          </StatusPill>
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="eyebrow"><span className="eyebrow-line" /> EVIDENCE-GROUNDED FACT CHECKING</div>
          <h1>Check the claim.<br /><span>Follow the evidence.</span></h1>
          <p className="hero-copy">
            Submit a statement and VeriAI will retrieve relevant Wikipedia evidence
            and assess whether that evidence supports, contradicts, or cannot establish the claim.
          </p>
        </section>

        <section className="workspace">
          <div className="input-panel panel">
            <div className="panel-heading">
              <div>
                <span className="section-kicker">01 / YOUR INPUT</span>
                <h2>Enter a claim</h2>
              </div>
              <span className="field-hint">One statement at a time</span>
            </div>

            <form onSubmit={handleVerify}>
              <label className="sr-only" htmlFor="claim-input">Claim to verify</label>
              <textarea
                id="claim-input"
                value={claim}
                onChange={(event) => setClaim(event.target.value)}
                placeholder="e.g. Magic Johnson played for the Lakers."
                maxLength={2000}
                rows={5}
                disabled={loading}
              />
              <div className="input-footer">
                <span>{claim.length}/2000 characters</span>
                <div className="input-actions">
                  <label className="top-k-control">
                    Evidence
                    <select value={topK} onChange={(event) => setTopK(event.target.value)} disabled={loading}>
                      <option value="3">3 sentences</option>
                      <option value="5">5 sentences</option>
                      <option value="8">8 sentences</option>
                      <option value="10">10 sentences</option>
                    </select>
                  </label>
                  <button className="clear-button" type="button" onClick={clearForm} disabled={loading && !claim}>
                    Clear
                  </button>
                  <button className="verify-button" type="submit" disabled={loading || !claim.trim()}>
                    {loading ? <><span className="spinner" /> Verifying…</> : <>Verify claim <span aria-hidden="true">→</span></>}
                  </button>
                </div>
              </div>
            </form>

            <div className="examples">
              <span className="examples-label">TRY AN EXAMPLE</span>
              <div className="example-list">
                {examples.map((example) => (
                  <button
                    key={example.label}
                    className={`example-chip ${example.tone}`}
                    type="button"
                    onClick={() => chooseExample(example)}
                    disabled={loading}
                  >
                    {example.label}
                  </button>
                ))}
              </div>
            </div>
            {error && <div className="error-banner" role="alert">{error}</div>}
          </div>

          <section className="result-panel panel" aria-live="polite">
            {!result && !loading && (
              <div className="empty-state">
                <div className="empty-icon"><span>✓</span></div>
                <span className="section-kicker">02 / VERIFICATION RESULT</span>
                <h2>Your result will appear here</h2>
                <p>VeriAI retrieves evidence first, then compares each evidence sentence with your claim using a natural language inference model.</p>
                <div className="verdict-legend">
                  <span><i className="legend-dot supports" /> Supports</span>
                  <span><i className="legend-dot refutes" /> Refutes</span>
                  <span><i className="legend-dot neutral" /> Not enough info</span>
                </div>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <span className="large-spinner" />
                <h2>Checking the evidence</h2>
                <p>Retrieving relevant sentences and evaluating the claim with DeBERTa.</p>
              </div>
            )}

            {result && verdictInfo && (
              <div className="result-content">
                <div className="result-heading">
                  <span className="section-kicker">02 / VERIFICATION RESULT</span>
                  <span className={`verdict-badge ${verdictInfo.tone}`}>{verdict}</span>
                </div>
                <h2>{verdictInfo.title}</h2>
                <p className="result-description">{verdictInfo.description}</p>

                <div className="result-stats">
                  <div className="stat-card">
                    <span>Retrieved evidence</span>
                    <strong>{result.retrieved_evidence_count ?? evidence.length}</strong>
                    <small>sentences</small>
                  </div>
                  <div className="stat-card">
                    <span>NLI softmax score (uncalibrated)</span>
                    <strong>{score}</strong>
                    <small>not a factual confidence score</small>
                  </div>
                  <div className="stat-card">
                    <span>Inference device</span>
                    <strong>{result.device || "—"}</strong>
                    <small>model execution</small>
                  </div>
                </div>

                <div className="decision-block">
                  <div className="subsection-heading">
                    <h3>Why this verdict?</h3>
                    <span>{result.decision_basis?.replaceAll("_", " ")}</span>
                  </div>
                  {result.selected_evidence ? (
                    <EvidenceCard item={result.selected_evidence} index={0} />
                  ) : (
                    <div className="no-decision-evidence">
                      No retrieved sentence was decisive. The available evidence was classified as neutral.
                    </div>
                  )}
                </div>

                <div className="evidence-list">
                  <div className="subsection-heading">
                    <h3>Evidence analysis</h3>
                    <span>{evidence.length} evaluated</span>
                  </div>
                  {evidence.length ? (
                    evidence.map((item, index) => (
                      <EvidenceCard key={`${item.page_id}-${item.sentence_id}-${index}`} item={item} index={index} />
                    ))
                  ) : (
                    <p className="muted-copy">No evidence sentences were scored.</p>
                  )}
                </div>

                <div className="model-note">
                  <strong>About the score:</strong> {result.confidence_note || "The NLI softmax score is uncalibrated and is not a probability that the claim is factually correct."} It is not a probability that the claim is factually correct.
                  <br />
                  <strong>Model:</strong> {result.model_name || "DeBERTa NLI"}
                </div>
              </div>
            )}
          </section>
        </section>

        <section className="process-strip">
          <div><span>01</span><strong>Retrieve</strong><small>Wikipedia + BM25</small></div>
          <div className="process-arrow">→</div>
          <div><span>02</span><strong>Compare</strong><small>DeBERTa NLI</small></div>
          <div className="process-arrow">→</div>
          <div><span>03</span><strong>Classify</strong><small>Three FEVER labels</small></div>
        </section>
      </main>

      <footer className="site-footer">
        <span>VeriAI · Evidence-grounded claim verification</span>
        <span>Prototype backed by a FEVER Wikipedia snapshot</span>
      </footer>
    </div>
  );
}
