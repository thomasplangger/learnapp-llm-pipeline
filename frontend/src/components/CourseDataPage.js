
import React, { useEffect, useMemo, useState } from "react";
import { useParams, useLocation, Link } from "react-router-dom";

const API = process.env.REACT_APP_BACKEND_URL || "";



const Card = ({ title, actions, children, style }) => (
  <div
    style={{
      border: "1px solid #e5e7eb",
      borderRadius: 12,
      padding: 16,
      background: "#fff",
      boxShadow: "0 1px 2px rgba(0,0,0,.04)",
      ...style,
    }}
  >
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 8,
      }}
    >
      <h3 style={{ margin: 0, fontSize: 16 }}>{title}</h3>
      <div>{actions}</div>
    </div>
    {children}
  </div>
);

function StatPill({ label, value, hint }) {
  return (
    <div
      style={{
        padding: "6px 10px",
        background: "#f5f5f7",
        borderRadius: 999,
        fontSize: 12,
        marginRight: 8,
      }}
    >
      <strong>{label}:</strong> {value}
      {hint ? ` ${hint}` : ""}
    </div>
  );
}

function approxTokens(text) {
  return Math.max(1, Math.floor((text || "").length / 4));
}

/* ------------------------------ Main Page --------------------------------- */

export default function CourseDataPage() {
  const { courseId } = useParams();
  const location = useLocation();
  const searchParams = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const compactMode = (searchParams.get("mode") || "").toLowerCase() === "auto";
  const autoLosParam = searchParams.get("autoLos") || "";

  const [chunks, setChunks] = useState([]);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [chunking, setChunking] = useState(false);
  const [refreshingMeta, setRefreshingMeta] = useState(false);
  // PDFs for this course (used to label chunk sources)
  const [pdfs, setPdfs] = useState([]);

  // Methods no longer include token-based "Other"
  const [method, setMethod] = useState("LLM [JSON]");
  const [methodsList, setMethodsList] = useState([
    "LLM [JSON]",
    "LLM [Raw]",
    "LLM [PDF]",
    "Context",
  ]);
  const [desiredChunks, setDesiredChunks] = useState("");
  // Learning Objectives
  const [desiredLOs, setDesiredLOs] = useState("");
  useEffect(() => {
    if (compactMode && autoLosParam) {
      setDesiredLOs(autoLosParam);
    }
  }, [compactMode, autoLosParam]);
  const [loMethod, setLoMethod] = useState("llm"); // "heuristic" | "llm"
  const [loadingLOs, setLoadingLOs] = useState(false);
  const [learningObjectives, setLearningObjectives] = useState([]);
  const [loEdits, setLoEdits] = useState({});
  const [savingLO, setSavingLO] = useState({});
  const [suggestions, setSuggestions] = useState({});
  const [loEditMode, setLoEditMode] = useState({});
  const [loDirty, setLoDirty] = useState({});
  const loadLOs = async () => {
    try {
      const r = await fetch(`${API}/api/course/${courseId}/learning-objectives`);
      if (!r.ok) return; // no LOs persisted yet
      const data = await r.json();
      const groups = Array.isArray(data.groups) ? data.groups : [];
      setLearningObjectives(groups);
      const edits = {};
      groups.forEach((g, i) => {
        edits[i] = {
          title: g.title || "",
          summary: g.summary || "",
          objectivesText: Array.isArray(g.objectives) ? g.objectives.join("\n") : "",
        };
      });
      setLoEdits(edits);
    } catch (e) {
      // ignore if none
    }
  };

  // independent expand states
  const [expandedTextMap, setExpandedTextMap] = useState({});
  const [expandedMetaMap, setExpandedMetaMap] = useState({});
  const toggleText = (idx) =>
    setExpandedTextMap((m) => ({ ...m, [idx]: !m[idx] }));
  const toggleMeta = (idx) =>
    setExpandedMetaMap((m) => ({ ...m, [idx]: !m[idx] }));

  useEffect(() => {
    fetch(`${API}/api/chunking/methods`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((list) => {
        const allowed = ["LLM [JSON]", "LLM [Raw]", "LLM [PDF]", "Context"];
        if (Array.isArray(list) && list.length) {
          setMethodsList(list.filter((m) => allowed.includes(m)));
        }
      })
      .catch(() => {});
  }, []);

  // Load course PDFs to map chunk source ids -> filenames
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        // Preferred endpoint
        const res = await fetch(`${API}/api/course/${courseId}/pdfs`);
        if (res.ok) {
          const items = await res.json();
          if (active) setPdfs(Array.isArray(items) ? items : []);
          return;
        }
      } catch {}
      // Fallback to legacy approach
      try {
        const courseRes = await fetch(`${API}/api/course/${courseId}`);
        const body = courseRes.ok ? await courseRes.json() : null;
        const ids = body?.course?.pdf_ids || [];
        if (ids.length) {
          const listRes = await fetch(`${API}/api/pdfs`);
          const list = listRes.ok ? await listRes.json() : [];
          if (active) setPdfs(list.filter((p) => ids.includes(p.pdf_id)));
        } else if (active) {
          setPdfs([]);
        }
      } catch {
        if (active) setPdfs([]);
      }
    })();
    return () => { active = false; };
  }, [courseId]);

  // Quick lookup: pdf_id -> filename
  const pdfNameById = useMemo(() => {
    const m = {};
    (pdfs || []).forEach((p) => {
      if (p && p.pdf_id) m[p.pdf_id] = p.filename || "document.pdf";
    });
    return m;
  }, [pdfs]);

  // Build a user-facing label for a chunk source
  const sourceLabelForChunk = (c) => {
    const idx = c?.meta?.source_pdf_index;
    if (idx === undefined || idx === null) return "";
    const id = c?.meta?.source_pdf_id;
    const name = id && pdfNameById[id] ? pdfNameById[id] : null;
    const suffix = ` (PDF #${Number(idx) + 1})`;
    return name ? ` [${name}]${suffix}` : suffix;
  };

  const loadChunks = async () => {
    setLoadingChunks(true);
    try {
      const r = await fetch(`${API}/api/chunks?course_id=${courseId}`);
      if (!r.ok) throw new Error(`GET /chunks failed: ${r.status}`);
      const data = await r.json();
      setChunks(Array.isArray(data) ? data : []);
    } catch (e) {
      console.error(e);
      alert("Loading chunks failed. See console for details.");
      setChunks([]);
    } finally {
      setLoadingChunks(false);
    }
  };

  useEffect(() => {
    setExpandedTextMap({});
    setExpandedMetaMap({});
    setLearningObjectives([]);
    loadChunks();
    loadLOs();
  }, [courseId]);

  // Keep edit buffers in sync when learning objectives change
  useEffect(() => {
    const edits = {};
    (learningObjectives || []).forEach((g, i) => {
      edits[i] = {
        title: g.title || "",
        summary: g.summary || "",
        objectivesText: Array.isArray(g.objectives) ? g.objectives.join("\n") : "",
      };
    });
    setLoEdits(edits);
  }, [learningObjectives]);

  const runChunking = async () => {
    setChunking(true);
    try {
      const body = {
        method,
        overwrite: true,
        dry_run: false,
        // Optional: strict_count could be added later if you expose it
      };
      if (desiredChunks !== "") body.desired_chunks = Number(desiredChunks);

      const r = await fetch(`${API}/api/course/${courseId}/chunk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const msg = await r.text();
        throw new Error(`Chunking failed: ${r.status} ${msg}`);
        }
      await loadChunks();
    } catch (e) {
      console.error(e);
      alert(e.message || "Chunking request failed.");
    } finally {
      setChunking(false);
    }
  };

  const refreshChunkMetadata = async () => {
    setRefreshingMeta(true);
    try {
      const r = await fetch(`${API}/api/course/${courseId}/chunks/refresh-metadata`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ persist: true }),
      });
      if (!r.ok) throw new Error(`POST /chunks/refresh-metadata failed: ${r.status}`);
      const data = await r.json();
      const arr = Array.isArray(data?.metadata) ? data.metadata : [];
      const byIdx = new Map(arr.filter(Boolean).map((m) => [Number(m.chunk_index), m]));
      setChunks((prev) =>
        (prev || []).map((c) => {
          const m = byIdx.get(Number(c.index));
          if (!m) return c;
          const meta = { ...(c.meta || {}), enriched: { ...m } };
          return { ...c, meta };
        })
      );
      // Also refresh from server to ensure DB-persisted metadata is in sync
      await loadChunks();
    } catch (e) {
      console.error(e);
      alert("Second-pass metadata generation failed.");
    } finally {
      setRefreshingMeta(false);
    }
  };

  const generateLOs = async () => {
    setLoadingLOs(true);
    // Clear current UI to avoid showing stale LOs while regenerating
    setLearningObjectives([]);
    setLoEditMode({});
    setLoDirty({});
    setSuggestions({});
    try {
      const body = { method: loMethod, persist: true };
      if (desiredLOs !== "") body.N = Number(desiredLOs);
      const r = await fetch(`${API}/api/course/${courseId}/learning-objectives`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`POST /learning-objectives failed: ${r.status}`);
      const data = await r.json();
      const groups = Array.isArray(data.groups) ? data.groups : [];
      // Overwrite UI with new LO set and reset local state
      setLearningObjectives(groups);
      setLoEditMode({});
      setLoDirty({});
      setSuggestions({});
      const edits = {};
      groups.forEach((g, i) => {
        edits[i] = {
          title: g.title || "",
          summary: g.summary || "",
          objectivesText: Array.isArray(g.objectives) ? g.objectives.join("\n") : "",
        };
      });
      setLoEdits(edits);
      showToast("Generated new LOs");
      // Trust returned payload; optionally refresh persisted copy after a short delay
      // setTimeout(() => loadLOs(), 300);
    } catch (e) {
      console.error(e);
      alert("Generating learning objectives failed. See console for details.");
      setLearningObjectives([]);
    } finally {
      setLoadingLOs(false);
    }
  };

  // Simple toast
  const [toast, setToast] = useState(null);
  const showToast = (msg) => {
    setToast({ msg, t: Date.now() });
    setTimeout(() => setToast(null), 2200);
  };

  const saveLO = async (idx) => {
    const edit = loEdits[idx] || {};
    const body = {
      lo_index: idx,
      title: edit.title ?? "",
      summary: edit.summary ?? "",
      objectives: (edit.objectivesText || "").split(/\r?\n/).map((s) => s.trim()).filter(Boolean),
    };
    try {
      setSavingLO((m) => ({ ...m, [idx]: true }));
      const r = await fetch(`${API}/api/course/${courseId}/learning-objectives`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`PATCH /learning-objectives failed: ${r.status}`);
      const data = await r.json();
      setLearningObjectives(Array.isArray(data.groups) ? data.groups : []);
    } catch (e) {
      console.error(e);
      alert("Saving LO failed.");
    } finally {
      setSavingLO((m) => ({ ...m, [idx]: false }));
    }
  };

  const regenerateLO = async (idx, mode = "all") => {
    try {
      setSavingLO((m) => ({ ...m, [idx]: true }));
      const r = await fetch(`${API}/api/course/${courseId}/learning-objectives/regenerate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lo_index: idx, mode, persist: true }),
      });
      if (!r.ok) throw new Error(`POST /learning-objectives/regenerate failed: ${r.status}`);
      const data = await r.json();
      setLearningObjectives((prev) => {
        const copy = [...(prev || [])];
        copy[idx] = data.group || copy[idx];
        return copy;
      });
      showToast("Refreshed LO");
    } catch (e) {
      console.error(e);
      alert("Regenerate failed.");
    } finally {
      setSavingLO((m) => ({ ...m, [idx]: false }));
    }
  };

  const refreshDirtyLos = async () => {
    const dirty = Object.entries(loDirty).filter(([_, v]) => v).map(([k]) => Number(k));
    if (!dirty.length) {
      showToast("No modified LOs");
      return;
    }
    for (const i of dirty) {
      await regenerateLO(i, "all");
    }
    showToast(`Refreshed ${dirty.length} LO${dirty.length === 1 ? "" : "s"}`);
  };

  const createLessonsFromLOs = async () => {
    if (!learningObjectives || learningObjectives.length === 0) {
      alert("No learning objectives found. Generate LOs first.");
      return;
    }
    try {
      const r = await fetch(`${API}/api/course/${courseId}/create-lessons-from-los`, {
        method: "POST",
      });
      if (!r.ok) throw new Error(`POST /course/${courseId}/create-lessons-from-los failed: ${r.status}`);
      await r.json();
      showToast("Created lessons from LOs");
      // Navigate to course page to see lessons grouped by LO
      window.location.assign(`/course/${courseId}`);
    } catch (e) {
      console.error(e);
      alert("Failed to create lessons from LOs.");
    }
  };

  const suggestMembers = async (idx) => {
    try {
      const title = loEdits[idx]?.title || learningObjectives[idx]?.title || "";
      const r = await fetch(`${API}/api/course/${courseId}/learning-objectives/suggest-members`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lo_index: idx, title }),
      });
      if (!r.ok) throw new Error(`POST /learning-objectives/suggest-members failed: ${r.status}`);
      const data = await r.json();
      setSuggestions((m) => ({ ...m, [idx]: { indices: data.suggested_indices || [], scores: data.scores || [] } }));
    } catch (e) {
      console.error(e);
      alert("Suggest members failed.");
    }
  };

  const applySuggestion = async (idx) => {
    const sg = suggestions[idx];
    if (!sg || !Array.isArray(sg.indices)) return;
    try {
      const r = await fetch(`${API}/api/course/${courseId}/learning-objectives`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lo_index: idx, chunk_indices: sg.indices }),
      });
      if (!r.ok) throw new Error(`PATCH /learning-objectives failed: ${r.status}`);
      const data = await r.json();
      setLearningObjectives(Array.isArray(data.groups) ? data.groups : []);
      showToast("Applied suggestions");
      setSuggestions((m) => ({ ...m, [idx]: undefined }));
    } catch (e) {
      console.error(e);
      alert("Applying suggestion failed.");
    }
  };

  const moveChunkTo = async (chunkIndex, toLoIndex, position) => {
    try {
      const r = await fetch(`${API}/api/course/${courseId}/learning-objectives/move-chunk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chunk_index: chunkIndex, to_lo_index: toLoIndex, position }),
      });
      if (!r.ok) throw new Error(`POST /learning-objectives/move-chunk failed: ${r.status}`);
      const data = await r.json();
      setLearningObjectives(Array.isArray(data.groups) ? data.groups : []);
      showToast("Moved chunk");
    } catch (e) {
      console.error(e);
      alert("Move chunk failed.");
    }
  };

  const totalTokens = useMemo(
    () => (chunks || []).reduce((a, c) => a + approxTokens(c.text || ""), 0),
    [chunks]
  );
  const percentOfTotal = (n) =>
    totalTokens ? ` (${Math.round((n / totalTokens) * 100)}%)` : "";

  const perChunkStats = useMemo(() => {
    const per = chunks.map((c) => {
      const t = c.text || "";
      return {
        index: c.index,
        tokens: approxTokens(t),
        header: (c.meta && c.meta.header) || "Section",
      };
    });
    const minTok = per.length ? Math.min(...per.map((x) => x.tokens)) : 0;
    const maxTok = per.length ? Math.max(...per.map((x) => x.tokens)) : 0;
    const avgTok = per.length
      ? Math.round(per.reduce((a, x) => a + x.tokens, 0) / per.length)
      : 0;
    return { per, minTok, maxTok, avgTok };
  }, [chunks]);

  return (
    <div
      style={{
        maxWidth: 1280,
        margin: "0 auto",
        padding: "16px 12px 80px",
        fontFamily:
          'Inter, ui-sans-serif, system-ui, "Segoe UI", Roboto, Helvetica, Arial',
      }}
    >
      {toast && (
        <div style={{ position: "fixed", top: 16, right: 16, background: "#111827", color: "#fff", padding: "10px 14px", borderRadius: 8, boxShadow: "0 4px 18px rgba(0,0,0,.2)", zIndex: 50 }}>
          {toast.msg}
        </div>
      )}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <h1 style={{ margin: "6px 0 12px" }}>Course Data</h1>
        <Link to={`/course/${courseId}`} style={{ textDecoration: "none" }}>
          ← Back to Course
        </Link>
      </div>

      <div
        style={{
          display: "inline-flex",
          border: "1px solid #ddd",
          borderRadius: 8,
          overflow: "hidden",
          marginBottom: 16,
        }}
      >
        <button
          style={{ padding: "8px 14px", border: "none", background: "#eef2ff" }}
          disabled
        >
          Chunks
        </button>
      </div>

      {/* Controls */}
      {!compactMode && (
        <Card
        title="Chunking"
        actions={
          <>
            <button
              onClick={loadChunks}
              disabled={loadingChunks || chunking}
              style={{
                marginRight: 8,
                padding: "6px 10px",
                border: "1px solid #ccd",
                borderRadius: 8,
                background: "#fff",
              }}
            >
              {loadingChunks ? "Loading…" : "Refresh"}
            </button>
            <button
              onClick={runChunking}
              disabled={chunking}
              style={{
                padding: "6px 10px",
                border: "1px solid #ccd",
                borderRadius: 8,
                background: "#fff",
              }}
            >
              {chunking ? "Running…" : "Run Chunking"}
            </button>
          </>
        }
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(2, minmax(260px, 1fr))",
            gap: 12,
            marginBottom: 12,
          }}
        >
          <label>
            <div style={{ fontSize: 12, color: "#444" }}>Method</div>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              className="w-full border rounded p-2"
            >
              {methodsList.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <label>
            <div style={{ fontSize: 12, color: "#444" }}>
              Desired chunks (optional)
            </div>
            <input
              type="number"
              min={1}
              value={desiredChunks}
              onChange={(e) => setDesiredChunks(e.target.value)}
              className="w-full border rounded p-2"
              placeholder="leave empty for topic-first"
            />
          </label>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
          <StatPill label="Chunks" value={chunks.length} />
          <StatPill label="Tokens total" value={totalTokens} />
          <StatPill label="Avg tokens" value={perChunkStats.avgTok} />
          <StatPill label="Min tokens" value={perChunkStats.minTok} />
          <StatPill label="Max tokens" value={perChunkStats.maxTok} />
        </div>
        <div style={{ marginTop: 10 }}>
          <button
            onClick={refreshChunkMetadata}
            disabled={refreshingMeta}
            style={{
              padding: "6px 10px",
              border: "1px solid #ccd",
              borderRadius: 8,
              background: "#fff",
            }}
            title="Run a second LLM pass to refresh chunk metadata"
          >
            {refreshingMeta ? "Refreshing meta…" : "Second pass: refresh chunk metadata"}
          </button>
        </div>
        </Card>
      )}

      {compactMode && (
        <div
          style={{
            marginTop: 12,
            padding: "10px 14px",
            borderRadius: 10,
            background: "#eef2ff",
            border: "1px solid #c7d2fe",
            color: "#1e1b4b",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span>
            Chunking and learning objectives were generated automatically. Need manual controls?
          </span>
          <Link
            to={`/course/${courseId}/data?mode=debug`}
            style={{ color: "#4338ca", textDecoration: "underline", fontWeight: 600 }}
          >
            Switch to Debug Mode
          </Link>
        </div>
      )}
      <Card
        title="Learning Objectives"
        actions={
          compactMode ? (
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={refreshDirtyLos}
                style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}
                title="Regenerate LOs you've edited"
              >
                Refresh LOs
              </button>
              <button
                onClick={createLessonsFromLOs}
                style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}
                title="Create course lessons from current LOs"
              >
                Create Course (Generate Lessons)
              </button>
            </div>
          ) : (
            <div style={{ display: "flex", gap: 8 }}>
              <select
                value={loMethod}
                onChange={(e) => setLoMethod(e.target.value)}
                className="w-full border rounded p-2"
                style={{ width: 160 }}
                title="Generation method"
              >
                <option value="llm">LLM</option>
                <option value="heuristic">Heuristic</option>
              </select>
              <input
                type="number"
                min={1}
                value={desiredLOs}
                onChange={(e) => setDesiredLOs(e.target.value)}
                className="w-full border rounded p-2"
                placeholder="Leave empty to auto-select"
                style={{ width: 220 }}
              />
              <button
                onClick={generateLOs}
                disabled={loadingLOs}
                style={{
                  padding: "6px 10px",
                  border: "1px solid #ccd",
                  borderRadius: 8,
                  background: "#fff",
                }}
              >
                {loadingLOs ? "Generating…" : "Generate Learning Objectives"}
              </button>
              <button
                onClick={createLessonsFromLOs}
                style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}
                title="Create course lessons from current LOs"
              >
                Create Course (Generate Lessons)
              </button>
              <button
                onClick={refreshDirtyLos}
                style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}
                title="Regenerate text for modified LOs"
              >
                Refresh Modified LOs
              </button>
            </div>
          )
        }
        style={{ marginTop: 16 }}
      >
        {learningObjectives.length === 0 ? (
          <div style={{ color: "#666" }}>
            {loadingLOs ? "Generating…" : "No learning objectives yet."}
          </div>
        ) : (
          <div style={{ display: "grid", gap: 14 }}>
            {learningObjectives.map((lo, idx) => {
              const borderColor = lo.color || "#22c55e";
              const memberChunks = lo.chunk_indices.map((ci) => chunks[ci]).filter(Boolean);
              const totalTok = memberChunks.reduce((a, c) => a + approxTokens(c?.text || ""), 0);
              return (
                <div key={idx} style={{ position: "relative", border: `2px solid ${borderColor}`, borderRadius: 10, padding: 12, opacity: savingLO[idx] ? 0.7 : 1 }}>
                  {savingLO[idx] && (
                    <div style={{ position: "absolute", top: 8, right: 8, background: "#111827", color: "#fff", fontSize: 12, padding: "4px 8px", borderRadius: 6 }}>
                      Updating…
                    </div>
                  )}
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                    <div>
                      <div style={{ fontWeight: 700, marginBottom: 4 }}>
                        LO {idx + 1}: {lo.title || `Learning Objective ${idx + 1}`}
                      </div>
                      <div style={{ fontSize: 12, color: "#666" }}>~{totalTok} tok</div>
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                      <button
                        onClick={async () => {
                          const currentlyEditing = !!loEditMode[idx];
                          if (currentlyEditing && loDirty[idx]) {
                            await saveLO(idx);
                            setLoDirty((d)=>({ ...d, [idx]: false }));
                          }
                          setLoEditMode((m)=>({ ...m, [idx]: !m[idx] }));
                        }}
                        style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}
                        title={loEditMode[idx] ? "Finish editing" : "Edit this LO"}
                      >
                        {loEditMode[idx] ? "Done" : "Edit"}
                      </button>
                      <button
                        onClick={() => regenerateLO(idx, "all")}
                        style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}
                        title="Refresh this LO's text from members"
                      >
                        Refresh
                      </button>
                    </div>
                  </div>
                  {lo.summary && (
                    <div style={{ marginTop: 8 }}>{lo.summary}</div>
                  )}
                  {Array.isArray(lo.objectives) && lo.objectives.length > 0 && (
                    <ul style={{ marginTop: 8 }}>
                      {lo.objectives.map((o, j) => (
                        <li key={j}>• {o}</li>
                      ))}
                    </ul>
                  )}
                  {!!loEditMode[idx] && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 8 }}>
                      <div>
                        <div style={{ fontSize: 12, color: "#666", marginBottom: 4 }}>Title</div>
                        <input
                          value={(loEdits[idx] && loEdits[idx].title) ?? lo.title ?? ""}
                        onChange={(e) => { setLoEdits((m) => ({ ...m, [idx]: { ...(m[idx] || {}), title: e.target.value } })); setLoDirty((d)=>({ ...d, [idx]: true })); }}
                          placeholder="Title"
                          className="w-full border rounded p-2"
                        />
                      </div>
                      <div>
                        <div style={{ fontSize: 12, color: "#666", marginBottom: 4 }}>Summary</div>
                        <textarea
                          value={(loEdits[idx] && loEdits[idx].summary) ?? lo.summary ?? ""}
                        onChange={(e) => { setLoEdits((m) => ({ ...m, [idx]: { ...(m[idx] || {}), summary: e.target.value } })); setLoDirty((d)=>({ ...d, [idx]: true })); }}
                          rows={3}
                          className="w-full border rounded p-2"
                        />
                        {lo.summary && (
                          <div style={{ marginTop: 6, fontSize: 12, color: "#666" }}>Original: {lo.summary}</div>
                        )}
                      </div>
                      <div>
                        <div style={{ fontSize: 12, color: "#666", marginBottom: 4 }}>Objectives (one per line)</div>
                        <textarea
                          value={(loEdits[idx] && loEdits[idx].objectivesText) ?? (Array.isArray(lo.objectives) ? lo.objectives.join("\n") : "")}
                          onChange={(e) => setLoEdits((m) => ({ ...m, [idx]: { ...(m[idx] || {}), objectivesText: e.target.value } }))}
                          rows={4}
                          className="w-full border rounded p-2"
                        />
                      </div>
                      <div style={{ display: "flex", gap: 8 }}>
                        <button onClick={() => { saveLO(idx); setLoDirty((d)=>({ ...d, [idx]: false })); }} style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}>Save</button>
                        <button onClick={() => regenerateLO(idx, "all")} style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}>Regenerate text</button>
                        <button onClick={() => suggestMembers(idx)} style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}>Suggest members</button>
                      </div>
                      {suggestions[idx] && Array.isArray(suggestions[idx].indices) && suggestions[idx].indices.length > 0 && (
                        <div style={{ border: "1px dashed #ccd", borderRadius: 8, padding: 10, background: "#fbfbff" }}>
                          <div style={{ fontWeight: 600, marginBottom: 6 }}>Suggested members</div>
                          <div style={{ fontSize: 12, color: "#666", marginBottom: 6 }}>Indices: {suggestions[idx].indices.join(", ")}</div>
                          <button onClick={() => applySuggestion(idx)} style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}>Apply suggestions</button>
                        </div>
                      )}
                    </div>
                  </div>
                  )}

                  <div style={{ display: "grid", gap: 8, marginTop: 10 }} onDragOver={(e)=>e.preventDefault()} onDrop={(e)=>{ const data=e.dataTransfer.getData('text/plain'); const from = Number(data); if(!Number.isNaN(from)) { moveChunkTo(from, idx); } }}>
                    {memberChunks.map((c, i) => {
                      const idxAbs = lo.chunk_indices[i];
                      const tokens = approxTokens(c?.text || "");
                      const title = (c?.meta?.enriched?.title) || (c?.meta?.header) || `Chunk #${idxAbs+1}`;
                      const srcLbl = sourceLabelForChunk(c);
                      return (
                        <div key={i} style={{ border: "1px solid #eee", borderRadius: 8, padding: 10 }} onDragOver={(e)=>e.preventDefault()} onDrop={(e)=>{ const data=e.dataTransfer.getData('text/plain'); const from = Number(data); if(!Number.isNaN(from)) { moveChunkTo(from, idx, i); } }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }} draggable={true} onDragStart={(e)=>{ e.dataTransfer.setData('text/plain', String(idxAbs)); }}>
                            <div style={{ fontWeight: 600 }}>
                              #{idxAbs + 1} — {title}
                              <span style={{ marginLeft: 8, fontSize: 12, color: "#666" }}>~{tokens} tok{srcLbl}</span>
                            </div>
                            <div style={{ display: "flex", gap: 8 }} onDragOver={(e)=>e.preventDefault()} onDrop={(e)=>{ const data=e.dataTransfer.getData('text/plain'); const from = Number(data); if(!Number.isNaN(from)) { moveChunkTo(from, idx); } }}>
                              <button
                                onClick={() => toggleMeta(idxAbs)}
                                style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}
                              >
                                {expandedMetaMap[idxAbs] ? "Hide meta" : "Show meta"}
                              </button>
                              <button
                                onClick={() => toggleText(idxAbs)}
                                style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}
                              >
                                {expandedTextMap[idxAbs] ? "Hide text" : "Show text"}
                              </button>
                            </div>
                          </div>

                          {expandedMetaMap[idxAbs] && (
                            <div
                              style={{
                                marginTop: 10,
                                border: "1px solid #eee",
                                borderRadius: 8,
                                padding: 12,
                                background: "#fafafa",
                              }}
                            >
                              {(() => {
                                const enriched = c?.meta?.enriched || {};
                                const hasKeywords = Array.isArray(enriched.keywords) && enriched.keywords.length > 0;
                                return (
                                  <div>
                                    <div style={{ fontSize: 13, color: "#444", marginBottom: 6, fontWeight: 600 }}>Metadata</div>
                                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                                      <div>
                                        <div style={{ fontSize: 12, color: "#666" }}>Title</div>
                                        <div>{enriched.title ?? ""}</div>
                                      </div>
                                      <div>
                                        <div style={{ fontSize: 12, color: "#666" }}>Topic</div>
                                        <div>{enriched.topic ?? ""}</div>
                                      </div>
                                      <div style={{ gridColumn: "1 / -1" }}>
                                        <div style={{ fontSize: 12, color: "#666" }}>Summary</div>
                                        <div>{enriched.summary ?? ""}</div>
                                      </div>
                                      <div style={{ gridColumn: "1 / -1" }}>
                                        <div style={{ fontSize: 12, color: "#666" }}>Keywords</div>
                                        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                          {hasKeywords ? (
                                            enriched.keywords.map((k, j) => (
                                              <span key={j} style={{ fontSize: 12, background: "#eef2ff", border: "1px solid #dbe2ff", padding: "2px 6px", borderRadius: 999 }}>{k}</span>
                                            ))
                                          ) : (
                                            ""
                                          )}
                                        </div>
                                      </div>
                                      <div>
                                        <div style={{ fontSize: 12, color: "#666" }}>Confidence</div>
                                        <div>{enriched.confidence ?? ""}</div>
                                      </div>
                                      <div>
                                        <div style={{ fontSize: 12, color: "#666" }}>Embedded with</div>
                                        <div>{enriched.embedded_with ?? ""}</div>
                                      </div>
                                      <div>
                                        <div style={{ fontSize: 12, color: "#666" }}>Enriched at</div>
                                        <div>{enriched.enriched_at ?? ""}</div>
                                      </div>
                                    </div>
                                  </div>
                                );
                              })()}
                            </div>
                          )}

                          {expandedTextMap[idxAbs] && (
                            <pre
                              style={{
                                marginTop: 10,
                                whiteSpace: "pre-wrap",
                                fontFamily: "ui-monospace, Menlo, Consolas, monospace",
                              }}
                            >
                              {c?.text || ""}
                            </pre>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Card>
      <Card title="Chunks" actions={null} style={{ marginTop: 16 }}>
        {chunks.length === 0 ? (
          <div style={{ color: "#666" }}>
            {loadingChunks
              ? "Loading chunks…"
              : "No chunks found. Run chunking or refresh."}
          </div>
        ) : (
          <div style={{ display: "grid", gap: 10 }}>
            {chunks.map((c, i) => {
              const tokens = approxTokens(c.text || "");
              const srcLbl = sourceLabelForChunk(c);

              const enriched = c?.meta?.enriched || {};
              const hasKeywords =
                Array.isArray(enriched.keywords) && enriched.keywords.length > 0;

              return (
                <div
                  key={i}
                  style={{
                    border: "1px solid #eee",
                    borderRadius: 8,
                    padding: 12,
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      gap: 12,
                      alignItems: "flex-start",
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 600, marginBottom: 2 }}>
                        #{i + 1} — {(c?.meta?.enriched?.title) || (c?.meta?.header) || "Section"}
                      </div>
                      <div style={{ fontSize: 12, color: "#666" }}>
                        ~{tokens} tok{percentOfTotal(tokens)}
                        {srcLbl}
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                      <button
                        onClick={() => toggleMeta(i)}
                        style={{
                          padding: "6px 10px",
                          border: "1px solid #ccd",
                          borderRadius: 8,
                          background: "#fff",
                        }}
                      >
                        {expandedMetaMap[i] ? "Hide meta" : "Show meta"}
                      </button>
                      <button
                        onClick={() => toggleText(i)}
                        style={{
                          padding: "6px 10px",
                          border: "1px solid #ccd",
                          borderRadius: 8,
                          background: "#fff",
                        }}
                      >
                        {expandedTextMap[i] ? "Hide text" : "Show text"}
                      </button>
                    </div>
                  </div>

                  {expandedMetaMap[i] && (
                    <div
                      style={{
                        marginTop: 10,
                        border: "1px solid #eee",
                        borderRadius: 8,
                        padding: 12,
                        background: "#fafafa",
                      }}
                    >
                      <div
                        style={{
                          fontSize: 13,
                          color: "#444",
                          marginBottom: 6,
                          fontWeight: 600,
                        }}
                      >
                        Metadata
                      </div>

                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "1fr 1fr",
                          gap: 10,
                        }}
                      >
                        <div>
                          <div style={{ fontSize: 12, color: "#666" }}>
                            Title
                          </div>
                          <div>{enriched.title ?? "—"}</div>
                        </div>
                        <div>
                          <div style={{ fontSize: 12, color: "#666" }}>
                            Topic
                          </div>
                          <div>{enriched.topic ?? "—"}</div>
                        </div>

                        <div style={{ gridColumn: "1 / -1" }}>
                          <div style={{ fontSize: 12, color: "#666" }}>
                            Summary
                          </div>
                          <div>{enriched.summary ?? "—"}</div>
                        </div>

                        <div style={{ gridColumn: "1 / -1" }}>
                          <div style={{ fontSize: 12, color: "#666" }}>
                            Keywords
                          </div>
                          <div
                            style={{
                              display: "flex",
                              flexWrap: "wrap",
                              gap: 6,
                            }}
                          >
                            {hasKeywords
                              ? enriched.keywords.map((k, idx2) => (
                                  <span
                                    key={idx2}
                                    style={{
                                      fontSize: 12,
                                      background: "#eef2ff",
                                      border: "1px solid #dbe2ff",
                                      padding: "2px 6px",
                                      borderRadius: 999,
                                    }}
                                  >
                                    {k}
                                  </span>
                                ))
                              : "—"}
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize: 12, color: "#666" }}>
                            Confidence
                          </div>
                          <div>{enriched.confidence ?? "—"}</div>
                        </div>
                        <div>
                          <div style={{ fontSize: 12, color: "#666" }}>
                            Embedded with
                          </div>
                          <div>{enriched.embedded_with ?? "—"}</div>
                        </div>
                        <div>
                          <div style={{ fontSize: 12, color: "#666" }}>
                            Enriched at
                          </div>
                          <div>{enriched.enriched_at ?? "—"}</div>
                        </div>
                      </div>
                    </div>
                  )}
                  {expandedTextMap[i] && (
                    <pre
                      style={{
                        marginTop: 10,
                        whiteSpace: "pre-wrap",
                        fontFamily:
                          "ui-monospace, Menlo, Consolas, monospace",
                      }}
                    >
                      {c.text || ""}
                    </pre>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </Card>
    </div>
  );
}
