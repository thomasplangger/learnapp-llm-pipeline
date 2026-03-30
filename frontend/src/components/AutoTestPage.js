
import React, { useEffect, useMemo, useState, useRef, useCallback } from "react";
import { Link } from "react-router-dom";

const API = process.env.REACT_APP_BACKEND_URL || "";


const DEFAULT_BASE_DIR = "backend/generated";
const DEFAULT_OUT_DIR  = "backend/testdata";



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
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
      <h3 style={{ margin: 0, fontSize: 16 }}>{title}</h3>
      <div>{actions}</div>
    </div>
    {children}
  </div>
);

const Radio = (props) => (
  <label style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
    <input type="radio" {...props} />
    <span>{props.label}</span>
  </label>
);

const Checkbox = (props) => (
  <label style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
    <input type="checkbox" {...props} />
    <span>{props.label}</span>
  </label>
);

export default function AutoTestPage() {
  
  const [courses, setCourses] = useState([]);
  const [selectedCourseId, setSelectedCourseId] = useState("standalone");

  useEffect(() => {
    let active = true;
    fetch(`${API}/api/courses`)
      .then(r => (r.ok ? r.json() : []))
      .then(list => { if (active) setCourses(list || []); })
      .catch(() => {});
    return () => { active = false; };
  }, []);

  const [runs, setRuns] = useState(5);
  const [minPages, setMinPages] = useState(4);
  const [maxPages, setMaxPages] = useState(12);
  const [minPdfs, setMinPdfs] = useState(1);
  const [maxPdfs, setMaxPdfs] = useState(3);

  const [testing, setTesting] = useState(false);
  const [progress, setProgress] = useState({ completed: 0, total: 0, current: "", status: "idle", cache_dir: null, use_cached: null });
  const pollRef = useRef(null);
  const [jobId, setJobId] = useState(null);

  const [topics, setTopics] = useState([]);
  const [loadingTopics, setLoadingTopics] = useState(false);
  const [topicMode, setTopicMode] = useState("mixed");
  const [chosen, setChosen] = useState({});
  const [lengthMode, setLengthMode] = useState("mixed");

  const [datasetMode, setDatasetMode] = useState("new");
  const [shouldCacheDataset, setShouldCacheDataset] = useState(false);
  const [cacheLabel, setCacheLabel] = useState("");
  const [cacheOnlyMode, setCacheOnlyMode] = useState(false);
  const [cacheList, setCacheList] = useState([]);
  const [selectedCache, setSelectedCache] = useState("");
  const [loadingCaches, setLoadingCaches] = useState(false);
  const [cachesBase, setCachesBase] = useState("");
  const [modWindowSize, setModWindowSize] = useState("");
  const [modHeadingStrength, setModHeadingStrength] = useState("1.0");
  const [modSimilarityThreshold, setModSimilarityThreshold] = useState("");
  const [modHierarchical, setModHierarchical] = useState(false);
  const [modPrechunking, setModPrechunking] = useState(false);
  const [modPrechunkingPages, setModPrechunkingPages] = useState("30");
  const [modMergeSmall, setModMergeSmall] = useState("");
  const [modSplitLarge, setModSplitLarge] = useState("");
  const [batchLabel, setBatchLabel] = useState("");
  const [batchVariant, setBatchVariant] = useState("");

  const selectedTopics = useMemo(
    () => Object.entries(chosen).filter(([, v]) => v).map(([k]) => k),
    [chosen]
  );
  const selectedCount = selectedTopics.length;
  const selectedCacheMeta = useMemo(
    () => cacheList.find((c) => c.name === selectedCache),
    [cacheList, selectedCache]
  );
  const modSummary = useMemo(() => {
    const parts = [];
    if (modWindowSize.trim()) parts.push(`window=${modWindowSize}`);
    if (modHeadingStrength.trim() && modHeadingStrength !== "1.0") parts.push(`heading=${modHeadingStrength}`);
    if (modSimilarityThreshold.trim()) parts.push(`similarity<th=${modSimilarityThreshold}`);
    if (modHierarchical) parts.push("hierarchical");
    if (modPrechunking) parts.push(`prechunk>=${modPrechunkingPages || "auto"}`);
    if (modMergeSmall.trim()) parts.push(`merge<${modMergeSmall}`);
    if (modSplitLarge.trim()) parts.push(`split>${modSplitLarge}`);
    return parts;
  }, [
    modWindowSize,
    modHeadingStrength,
    modSimilarityThreshold,
    modHierarchical,
    modPrechunking,
    modPrechunkingPages,
    modMergeSmall,
    modSplitLarge,
  ]);
  const resetMods = useCallback(() => {
    setModWindowSize("");
    setModHeadingStrength("1.0");
    setModSimilarityThreshold("");
    setModHierarchical(false);
    setModPrechunking(false);
    setModPrechunkingPages("30");
    setModMergeSmall("");
    setModSplitLarge("");
  }, []);

  // Terminal-like log
  const [log, setLog] = useState("Idle");
  const appendLog = useCallback(
    (line) => setLog((prev) => (prev ? prev + "\n" : "") + line),
    []
  );

  const fetchCaches = useCallback(async () => {
    setLoadingCaches(true);
    try {
      const resp = await fetch(`${API}/api/auto-test/caches`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const list = Array.isArray(data.caches) ? data.caches : [];
      setCacheList(list);
      setCachesBase(data.base || "");
      if (list.length === 0) {
        setSelectedCache("");
      } else if (!selectedCache || !list.some((c) => c.name === selectedCache)) {
        setSelectedCache(list[0].name);
      }
      appendLog(`Caches refreshed (${list.length})`);
    } catch (e) {
      appendLog(`ERROR fetching caches: ${e.message || e}`);
      setCacheList([]);
      setCachesBase("");
      setSelectedCache("");
    } finally {
      setLoadingCaches(false);
    }
  }, [appendLog, selectedCache]);

  useEffect(() => {
    fetchCaches();
  }, [fetchCaches]);

  // Fetch topics from backend
  const fetchTopics = useCallback(async () => {
    setLoadingTopics(true);
    try {
      const resp = await fetch(`${API}/api/topics?base=${encodeURIComponent(DEFAULT_BASE_DIR)}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const list = Array.isArray(data.topics) ? data.topics : [];
      setTopics(list);
      setChosen((prev) => {
        const next = {};
        list.forEach((t) => (next[t] = prev[t] ?? false));
        return next;
      });
      appendLog(`Topics refreshed (${list.length}) from ${DEFAULT_BASE_DIR}`);
    } catch (e) {
      appendLog(`ERROR fetching topics: ${e.message || e}`);
      setTopics([]);
    } finally {
      setLoadingTopics(false);
    }
  }, []);

  useEffect(() => { fetchTopics(); }, [fetchTopics]);
  useEffect(() => {
    if (datasetMode === "cache" && !selectedCache && cacheList.length) {
      setSelectedCache(cacheList[0].name);
    }
  }, [datasetMode, cacheList, selectedCache]);

  // Start Auto Test job
  const startAutoTest = async () => {
    if (minPages > maxPages) { alert("Min pages must be ≤ max pages."); return; }
    if (minPdfs > maxPdfs) { alert("Min PDFs/run must be ≤ max PDFs/run."); return; }
    if (topicMode === "topics" && selectedCount === 0) { alert("Pick at least one topic or switch to Random mix."); return; }
    if (datasetMode === "cache" && !selectedCache) { alert("Select a cached dataset to use."); return; }
    if (datasetMode === "new") {
      if (cacheOnlyMode && !shouldCacheDataset) { alert("Enable caching and provide a label before choosing cache-only."); return; }
      if (shouldCacheDataset && !cacheLabel.trim()) { alert("Provide a cache label when saving the dataset."); return; }
    }

    try {
      setTesting(true);
      const expectedRuns = datasetMode === "cache"
        ? (selectedCacheMeta?.runs || Number(runs) || 0)
        : Number(runs) || 0;
      setProgress({
        completed: 0,
        total: expectedRuns,
        current: "",
        status: "starting",
        cache_dir: null,
        use_cached: datasetMode === "cache" ? selectedCache : null,
      });
      appendLog(`Starting auto test: runs=${runs}, pages=[${minPages}-${maxPages}], PDFs/run=[${minPdfs}-${maxPdfs}]`);
      appendLog(`Mode=${topicMode}${topicMode==="topics" ? ` topics=[${selectedTopics.join(", ")}]` : ""}`);
      appendLog(`Lengths=${lengthMode}`);
      appendLog(`Base=${DEFAULT_BASE_DIR}  Out=${DEFAULT_OUT_DIR}`);
      if (datasetMode === "cache") {
        appendLog(`Using cached dataset '${selectedCache}'`);
      } else if (shouldCacheDataset) {
        appendLog(`Caching dataset as '${cacheLabel.trim()}'${cacheOnlyMode ? " (cache only)" : ""}`);
      } else {
        appendLog("Generating fresh dataset (not cached).");
      }

      const parseNumber = (value) => {
        if (value === null || value === undefined) return null;
        const trimmed = `${value}`.trim();
        if (!trimmed) return null;
        const num = Number(trimmed);
        return Number.isFinite(num) ? num : null;
      };
      const modWindowVal = parseNumber(modWindowSize);
      const headingRaw = (modHeadingStrength || "").trim();
      const modHeadingVal = parseNumber(headingRaw);
      const shouldSendHeading = modHeadingVal !== null && headingRaw !== "" && Number(headingRaw) !== 1;
      const modSimVal = parseNumber(modSimilarityThreshold);
      const modMergeVal = parseNumber(modMergeSmall);
      const modSplitVal = parseNumber(modSplitLarge);
      const modPrechunkPagesVal = parseNumber(modPrechunkingPages);
      const batchLabelSafe = (batchLabel || "").trim();
      const batchVariantSafe = (batchVariant || "").trim();

      const payload = {
        course_id: selectedCourseId || "standalone",
        runs: Number(runs),
        min_pages: Number(minPages),
        max_pages: Number(maxPages),
        min_pdfs: Number(minPdfs),
        max_pdfs: Number(maxPdfs),
        selection_mode: topicMode,
        topics: topicMode === "topics" ? selectedTopics : [],
        length_mode: lengthMode,
        base_dir: DEFAULT_BASE_DIR,
        out_dir: DEFAULT_OUT_DIR,
        cache_only: datasetMode === "new" && shouldCacheDataset ? cacheOnlyMode : false,
        cache_name: datasetMode === "new" && shouldCacheDataset ? cacheLabel.trim() : null,
        use_cached: datasetMode === "cache" ? selectedCache : null,
      };
      if (batchLabelSafe) payload.batch_label = batchLabelSafe;
      if (batchVariantSafe) payload.batch_variant = batchVariantSafe;
      if (modWindowVal !== null) payload.mod_window_size = modWindowVal;
      if (shouldSendHeading) payload.mod_heading_strength = modHeadingVal;
      if (modSimVal !== null) payload.mod_similarity_threshold = modSimVal;
      if (modHierarchical) payload.mod_hierarchical = true;
      if (modPrechunking) {
        payload.mod_prechunking = true;
        if (modPrechunkPagesVal !== null) payload.mod_prechunking_min_pages = modPrechunkPagesVal;
      }
      if (modMergeVal !== null) payload.mod_merge_small = modMergeVal;
      if (modSplitVal !== null) payload.mod_split_large = modSplitVal;
      appendLog(modSummary.length ? `Mods: ${modSummary.join("; ")}` : "Mods: default settings.");

      const r = await fetch(`${API}/api/auto-test/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) {
        const msg = await r.text();
        throw new Error(`Auto Test start failed: ${r.status} ${msg}`);
      }
      const { job_id } = await r.json();
      setJobId(job_id);
      appendLog(`Job started: ${job_id}`);
      pollRef.current = setInterval(async () => {
        try {
          const pr = await fetch(`${API}/api/auto-test/status?job_id=${encodeURIComponent(job_id)}`);
          if (pr.ok) {
            const data = await pr.json();
            setProgress(data);
            if (data.current) appendLog(`• ${data.current}`);
            if (data.status === "done" || data.status === "error") {
              clearInterval(pollRef.current); pollRef.current = null;
              setTesting(false);
              appendLog(data.status === "done" ? "Job finished." : `Job errored: ${data.error || ""}`);
              if (data.cache_dir) {
                appendLog(`Cache saved to ${data.cache_dir}`);
                fetchCaches();
              }
            }
          }
        } catch {}
      }, 1000);
    } catch (e) {
      appendLog(`ERROR starting test: ${e.message || e}`);
      setTesting(false);
    }
  };

  // cleanup polling
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const progressPct = progress.total ? Math.min(100, Math.round((progress.completed / progress.total) * 100)) : 0;

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "16px 12px 80px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <h1 style={{ margin: "6px 0 12px" }}>Auto Test System</h1>
        <Link to="/" style={{ textDecoration: "none" }}>
          ← Back to Courses
        </Link>
      </div>

      <Card
        title="Auto Test"
        actions={
          <button
            onClick={startAutoTest}
            disabled={testing}
            style={{ padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}
          >
            {testing ? "Running…" : "Start"}
          </button>
        }
      >
        {/* Course association (optional) */}
        <div style={{ marginBottom: 12, display: "flex", gap: 12, alignItems: "center" }}>
          <div style={{ fontWeight: 600 }}>Associate with Course (optional):</div>
          <select
            value={selectedCourseId}
            onChange={(e) => setSelectedCourseId(e.target.value)}
            style={{ border: "1px solid #ddd", borderRadius: 8, padding: "6px 8px" }}
          >
            <option value="standalone">Standalone</option>
            {courses.map(c => (
              <option key={c.id} value={c.id}>{c.title}</option>
            ))}
          </select>
          </div>

          <div style={{ border: "1px solid #f0f0f0", borderRadius: 10, padding: 12, marginBottom: 16, background: "#fafafa" }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Dataset source</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginBottom: 8 }}>
              <Radio
                name="dataset-mode"
                value="new"
                checked={datasetMode === "new"}
                onChange={() => setDatasetMode("new")}
                label="Generate new dataset"
              />
              <Radio
                name="dataset-mode"
                value="cache"
                checked={datasetMode === "cache"}
                onChange={() => setDatasetMode("cache")}
                label="Use cached dataset"
              />
            </div>
            {datasetMode === "new" ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <Checkbox
                  label="Save generated dataset to cache"
                  checked={shouldCacheDataset}
                  onChange={(e) => {
                    setShouldCacheDataset(e.target.checked);
                    if (!e.target.checked) {
                      setCacheLabel("");
                      setCacheOnlyMode(false);
                    }
                  }}
                />
                {shouldCacheDataset && (
                  <>
                    <label style={{ fontSize: 12, color: "#444" }}>
                      Cache label
                      <input
                        type="text"
                        value={cacheLabel}
                        onChange={(e) => setCacheLabel(e.target.value)}
                        placeholder="e.g., week42_mix"
                        style={{ width: "100%", border: "1px solid #ddd", borderRadius: 8, padding: "6px 8px", marginTop: 4 }}
                      />
                    </label>
                    <Checkbox
                      label="Cache only (skip chunking & evaluation)"
                      checked={cacheOnlyMode}
                      onChange={(e) => setCacheOnlyMode(e.target.checked)}
                    />
                  </>
                )}
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <select
                    value={selectedCache}
                    onChange={(e) => setSelectedCache(e.target.value)}
                    style={{ border: "1px solid #ddd", borderRadius: 8, padding: "6px 8px", minWidth: 200 }}
                    disabled={!cacheList.length}
                  >
                    <option value="">Select cached dataset</option>
                    {cacheList.map((cache) => (
                      <option key={cache.name} value={cache.name}>
                        {cache.name} {cache.runs ? `(${cache.runs} runs)` : ""}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={fetchCaches}
                    type="button"
                    style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #dcdfe6", background: "#fff" }}
                  >
                    {loadingCaches ? "Refreshing…" : "Refresh"}
                  </button>
                </div>
                {selectedCache && (
                  <div style={{ fontSize: 12, color: "#555" }}>
                    Source folder: {cachesBase ? `${cachesBase}/${selectedCache}` : selectedCache}
                    {selectedCacheMeta?.runs !== undefined && (
                      <> — runs: {selectedCacheMeta?.runs ?? "?"}</>
                    )}
                  </div>
                )}
                {!cacheList.length && !loadingCaches && (
                  <div style={{ fontSize: 12, color: "#c2410c" }}>No cached datasets available.</div>
                )}
              </div>
            )}
          </div>

          <div style={{ border: "1px dashed #e2e8f0", borderRadius: 10, padding: 12, marginBottom: 16, background: "#fafafa" }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Batch tagging (optional)</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <label>
                <div style={{ fontSize: 12, color: "#444" }}>Batch label</div>
                <input
                  type="text"
                  value={batchLabel}
                  onChange={(e) => setBatchLabel(e.target.value)}
                  placeholder="e.g., batch_A_scaling"
                  style={{ width: "100%", border: "1px solid #ddd", borderRadius: 8, padding: "6px 8px" }}
                />
              </label>
              <label>
                <div style={{ fontSize: 12, color: "#444" }}>Variant / subgroup</div>
                <input
                  type="text"
                  value={batchVariant}
                  onChange={(e) => setBatchVariant(e.target.value)}
                  placeholder="e.g., A1_pages5"
                  style={{ width: "100%", border: "1px solid #ddd", borderRadius: 8, padding: "6px 8px" }}
                />
              </label>
            </div>
            <div style={{ fontSize: 12, color: "#555", marginTop: 6 }}>
              Use these to tag multiple testruns that belong to the same batch. Leave blank for ad-hoc runs.
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "minmax(320px, 1fr) minmax(320px, 1fr)", gap: 16 }}>
          <div>
            <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 8 }}>
              <Radio
                name="topic-mode"
                value="mixed"
                checked={topicMode === "mixed"}
                onChange={() => setTopicMode("mixed")}
                label="Random mix (all topics)"
              />
              <Radio
                name="topic-mode"
                value="topics"
                checked={topicMode === "topics"}
                onChange={() => setTopicMode("topics")}
                label="Pick specific topics"
              />
              <button
                onClick={fetchTopics}
                disabled={loadingTopics}
                style={{ marginLeft: "auto", padding: "6px 10px", border: "1px solid #ccd", borderRadius: 8, background: "#fff" }}
              >
                {loadingTopics ? "Refreshing…" : "Refresh topics"}
              </button>
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center", margin: "6px 0 10px" }}>
              <div style={{ fontWeight: 600, minWidth: 110 }}>Text length</div>
              <Radio name="length-mode" value="mixed" checked={lengthMode === "mixed"} onChange={() => setLengthMode("mixed")} label="Mixed" />
              <Radio name="length-mode" value="short" checked={lengthMode === "short"} onChange={() => setLengthMode("short")} label="Short only" />
              <Radio name="length-mode" value="medium" checked={lengthMode === "medium"} onChange={() => setLengthMode("medium")} label="Medium only" />
              <Radio name="length-mode" value="long" checked={lengthMode === "long"} onChange={() => setLengthMode("long")} label="Long only" />
            </div>

            <div
              style={{
                opacity: topicMode === "topics" ? 1 : 0.55,
                pointerEvents: topicMode === "topics" ? "auto" : "none",
                borderTop: "1px dashed #e5e7eb",
                paddingTop: 8,
              }}
            >
              <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 6 }}>
                <div style={{ fontWeight: 600, minWidth: 150 }}>Available topics</div>
                <div style={{ fontSize: 12, color: "#666" }}>
                  {topics.length ? `${topics.length} found in ${DEFAULT_BASE_DIR}` : "No topics found"}
                </div>
                <div style={{ marginLeft: "auto", fontSize: 12 }}>
                  Selected: <strong>{selectedCount}</strong>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 8, maxHeight: 320, overflow: "auto", paddingRight: 4 }}>
                {topics.map((t) => (
                  <Checkbox
                    key={t}
                    label={t}
                    checked={!!chosen[t]}
                    onChange={(e) => setChosen(prev => ({ ...prev, [t]: e.target.checked }))}
                  />
                ))}
              </div>
            </div>
          </div>
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <label>
                <div style={{ fontSize: 12, color: "#444" }}>Runs</div>
                <input type="number" min={1} value={runs} onChange={(e)=>setRuns(+e.target.value)} className="w-full border rounded p-2" />
              </label>
              <label>
                <div style={{ fontSize: 12, color: "#444" }}>PDFs per run (min)</div>
                <input type="number" min={1} value={minPdfs} onChange={(e)=>setMinPdfs(+e.target.value)} className="w-full border rounded p-2" />
              </label>
              <label>
                <div style={{ fontSize: 12, color: "#444" }}>PDFs per run (max)</div>
                <input type="number" min={1} value={maxPdfs} onChange={(e)=>setMaxPdfs(+e.target.value)} className="w-full border rounded p-2" />
              </label>
              <label>
                <div style={{ fontSize: 12, color: "#444" }}>Pages per sub-PDF (min)</div>
                <input type="number" min={1} value={minPages} onChange={(e)=>setMinPages(+e.target.value)} className="w-full border rounded p-2" />
              </label>
              <label>
                <div style={{ fontSize: 12, color: "#444" }}>Pages per sub-PDF (max)</div>
                <input type="number" min={1} value={maxPages} onChange={(e)=>setMaxPages(+e.target.value)} className="w-full border rounded p-2" />
              </label>
            </div>
            <div style={{ marginTop: 16 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#555", marginBottom: 4 }}>
                <span>Status: {progress.status}{progress.phase ? ` (${progress.phase})` : ""}</span>
                <span>{progress.completed} / {progress.total}</span>
              </div>
              <div style={{ background: "#e5e7eb", height: 8, borderRadius: 999 }}>
                <div style={{ width: `${progressPct}%`, height: 8, borderRadius: 999, background: "#4f46e5" }} />
              </div>
              <div style={{ fontSize: 12, color: "#666", marginTop: 6 }}>{progress.current}</div>
              {(progress.cache_dir || progress.use_cached) && (
                <div style={{ fontSize: 11, color: "#666", marginTop: 4 }}>
                  {progress.cache_dir ? <>Cache dir: {progress.cache_dir}</> : null}
                  {progress.use_cached ? <> {progress.cache_dir ? "• " : ""}Using cached: {progress.use_cached}</> : null}
                </div>
              )}
            </div>
          </div>
        </div>
        <div style={{ marginTop: 16 }}>
          <div style={{ fontSize: 12, color: "#666", marginBottom: 6 }}>Log</div>
          <textarea
            value={log}
            readOnly
            style={{ width: "100%", minHeight: 160, border: "1px solid #e5e7eb", borderRadius: 8, padding: 10, fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace" }}
          />
        </div>
      </Card>

      <Card
        title="Segmentation Mods"
        actions={
          <button
            onClick={resetMods}
            style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #ccd", background: "#fff" }}
          >
            Reset
          </button>
        }
        style={{ marginTop: 20 }}
      >
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16 }}>
          <label>
            <div style={{ fontSize: 12, color: "#444" }}>Window size override</div>
            <input
              type="number"
              min={1}
              placeholder="e.g. 400"
              value={modWindowSize}
              onChange={(e) => setModWindowSize(e.target.value)}
              className="w-full border rounded p-2"
            />
          </label>
          <label>
            <div style={{ fontSize: 12, color: "#444" }}>Heading strength</div>
            <input
              type="number"
              min={1}
              max={5}
              step={0.1}
              value={modHeadingStrength}
              onChange={(e) => setModHeadingStrength(e.target.value)}
              className="w-full border rounded p-2"
            />
          </label>
          <label>
            <div style={{ fontSize: 12, color: "#444" }}>Similarity threshold</div>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={modSimilarityThreshold}
              onChange={(e) => setModSimilarityThreshold(e.target.value)}
              className="w-full border rounded p-2"
              placeholder="0.40"
            />
          </label>
          <label>
            <div style={{ fontSize: 12, color: "#444" }}>Merge small chunks &lt;= (paras)</div>
            <input
              type="number"
              min={1}
              value={modMergeSmall}
              onChange={(e) => setModMergeSmall(e.target.value)}
              className="w-full border rounded p-2"
            />
          </label>
          <label>
            <div style={{ fontSize: 12, color: "#444" }}>Split large chunks &gt;= (paras)</div>
            <input
              type="number"
              min={1}
              value={modSplitLarge}
              onChange={(e) => setModSplitLarge(e.target.value)}
              className="w-full border rounded p-2"
            />
          </label>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 20, marginTop: 16 }}>
          <Checkbox
            label="Enable hierarchical segmentation"
            checked={modHierarchical}
            onChange={(e) => setModHierarchical(e.target.checked)}
          />
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <Checkbox
              label="Enable document pre-chunking"
              checked={modPrechunking}
              onChange={(e) => setModPrechunking(e.target.checked)}
            />
            {modPrechunking && (
              <input
                type="number"
                min={1}
                value={modPrechunkingPages}
                onChange={(e) => setModPrechunkingPages(e.target.value)}
                className="border rounded p-2"
                style={{ width: 90 }}
                placeholder="min pages"
              />
            )}
          </div>
        </div>
        <div style={{ fontSize: 12, color: "#666", marginTop: 12 }}>
          {modSummary.length ? (
            <>Active mods: {modSummary.join(", ")}</>
          ) : (
            "No mods applied; default chunking heuristics will be used."
          )}
        </div>
      </Card>

      <Card
        title="Cached datasets"
        actions={
          <button
            onClick={fetchCaches}
            disabled={loadingCaches}
            style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #ccd", background: "#fff" }}
          >
            {loadingCaches ? "Refreshing…" : "Refresh"}
          </button>
        }
      >
        <div style={{ fontSize: 12, color: "#555", marginBottom: 12 }}>
          Base folder: {cachesBase || "backend/testdata/pdf_cache"}
        </div>
        {cacheList.length === 0 ? (
          <div style={{ fontSize: 13, color: "#555" }}>
            {loadingCaches ? "Loading cached datasets…" : "No cached datasets found."}
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {cacheList.map((cache) => (
              <div
                key={cache.name}
                style={{
                  border: "1px solid #eee",
                  borderRadius: 8,
                  padding: 10,
                  background: selectedCache === cache.name ? "#f5f3ff" : "#fff",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <strong>{cache.name}</strong>
                    <div style={{ fontSize: 12, color: "#666" }}>
                      Runs: {cache.runs ?? "?"}
                      {Array.isArray(cache.docs_per_run) && cache.docs_per_run.length
                        ? ` • PDFs/run: ${cache.docs_per_run.join(", ")}`
                        : ""}
                    </div>
                  </div>
                  <button
                    style={{ padding: "4px 8px", borderRadius: 6, border: "1px solid #d0d7ff", background: "#eef2ff" }}
                    onClick={() => { setDatasetMode("cache"); setSelectedCache(cache.name); }}
                  >
                    Use this
                  </button>
                </div>
                <div style={{ fontSize: 11, color: "#777", marginTop: 4 }}>
                  Updated: {cache.last_modified || "n/a"}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
