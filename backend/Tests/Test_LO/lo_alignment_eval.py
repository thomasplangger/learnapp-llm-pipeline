import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def _build_tfidf(docs: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    tokens_per_doc = [ _tokenize(d) for d in docs ]
    vocab: Dict[str, int] = {}
    for toks in tokens_per_doc:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)

    if not vocab:
        return np.zeros((len(docs), 0), dtype=float), {}

    df = np.zeros(len(vocab), dtype=float)
    for toks in tokens_per_doc:
        seen = set(toks)
        for t in seen:
            df[vocab[t]] += 1.0

    N = max(1, len(docs))
    idf = np.log((1.0 + N) / (1.0 + df)) + 1.0

    mat = np.zeros((len(docs), len(vocab)), dtype=float)
    for i, toks in enumerate(tokens_per_doc):
        if not toks:
            continue
        counts: Dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        length = float(len(toks))
        for t, c in counts.items():
            idx = vocab[t]
            tf = c / length
            mat[i, idx] = tf * idf[idx]
    return mat, vocab


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=float)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B_norm = np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return (A @ B.T) / (A_norm @ B_norm.T)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_los(bundle: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for it in bundle.get("items", []):
        title = (it.get("title") or "").strip()
        topic = (it.get("topic") or it.get("category") or "").strip()
        if title:
            out.append(title)
        elif topic:
            out.append(topic)
    return [x for x in out if x]


def _build_chunks(paragraphs: List[Dict[str, Any]], bounds: List[int]) -> List[str]:
    chunks: List[str] = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        texts = []
        for p in paragraphs[a:b]:
            txt = (p.get("text") or "").strip()
            if txt:
                texts.append(txt)
        chunks.append("\n".join(texts))
    return chunks


def _normalize_bounds(bounds: List[Any], total: int) -> List[int]:
    vals: List[int] = []
    for b in bounds:
        if isinstance(b, (int, float)):
            vals.append(int(b))
    vals.extend([0, total])
    vals = sorted(set(max(0, min(total, v)) for v in vals))
    if not vals:
        return [0, total]
    if vals[0] != 0:
        vals.insert(0, 0)
    if vals[-1] != total:
        vals.append(total)
    return vals


def _evaluate_run(run_dir: Path, *, threshold: float) -> Dict[str, Any]:
    paras_p = run_dir / "paragraphs_used.json"
    bundle_p = run_dir / "combined_bundle.json"
    pred_p = run_dir / "llm_output.context.json"
    if not (paras_p.exists() and bundle_p.exists() and pred_p.exists()):
        raise FileNotFoundError("Missing run artifacts (paragraphs_used.json, combined_bundle.json, llm_output.context.json)")

    paragraphs = _read_json(paras_p)
    bundle = _read_json(bundle_p)
    pred = _read_json(pred_p)

    M = len(paragraphs)
    bounds = _normalize_bounds(pred.get("boundaries") or [], M)
    chunks = _build_chunks(paragraphs, bounds)
    los = _extract_los(bundle)

    if not los or not chunks:
        return {
            "run": run_dir.name,
            "los": len(los),
            "chunks": len(chunks),
            "mean_best_sim": 0.0,
            "median_best_sim": 0.0,
            "coverage_rate": 0.0,
        }

    docs = los + chunks
    tfidf, _ = _build_tfidf(docs)
    lo_vecs = tfidf[: len(los)]
    chunk_vecs = tfidf[len(los) :]
    sims = _cosine_sim_matrix(lo_vecs, chunk_vecs)

    best_idx = np.argmax(sims, axis=1)
    best_sim = np.max(sims, axis=1)
    coverage_rate = float(np.mean(best_sim >= threshold)) if len(best_sim) else 0.0

    # Chunk coverage: fraction of chunks that are selected by at least one LO
    covered_chunks = len(set(int(i) for i in best_idx)) if len(best_idx) else 0
    chunk_coverage = covered_chunks / max(1, len(chunks))

    return {
        "run": run_dir.name,
        "los": len(los),
        "chunks": len(chunks),
        "mean_best_sim": float(np.mean(best_sim)),
        "median_best_sim": float(np.median(best_sim)),
        "coverage_rate": coverage_rate,
        "chunk_coverage": float(chunk_coverage),
        "best_sim": best_sim.tolist(),
        "best_chunk_idx": best_idx.tolist(),
    }


def _plot_best_sim(best_sims: List[float], out_path: Path, title: str):
    if not best_sims:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(best_sims, bins=20, alpha=0.8)
    plt.xlabel("Best similarity per LO")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testrun-dir", required=True, help="Path to testrun_* directory (contains run_* folders)")
    ap.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold for coverage (default: 0.75)")
    args = ap.parse_args()

    testrun_dir = Path(args.testrun_dir)
    if not testrun_dir.exists():
        raise SystemExit(f"testrun dir not found: {testrun_dir}")

    out_dir = testrun_dir / "lo_alignment_eval"
    out_dir.mkdir(exist_ok=True)

    run_dirs = sorted([p for p in testrun_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
    summaries: List[Dict[str, Any]] = []
    all_best_sims: List[float] = []

    for run_dir in run_dirs:
        try:
            res = _evaluate_run(run_dir, threshold=float(args.threshold))
        except Exception as e:
            print(f"[SKIP] {run_dir.name}: {e}")
            continue

        summaries.append({k: v for k, v in res.items() if k not in {"best_sim", "best_chunk_idx"}})

        # per-run outputs
        per_run = {
            "run": res["run"],
            "best_sim": res.get("best_sim", []),
            "best_chunk_idx": res.get("best_chunk_idx", []),
        }
        all_best_sims.extend(per_run["best_sim"])
        (out_dir / f"{run_dir.name}_lo_alignment.json").write_text(
            json.dumps(per_run, indent=2), encoding="utf-8"
        )
        _plot_best_sim(
            res.get("best_sim", []),
            out_dir / f"{run_dir.name}_lo_similarity_hist.png",
            title=f"Best LO->chunk similarity ({run_dir.name})",
        )

    if not summaries:
        raise SystemExit("No runs processed.")

    df = pd.DataFrame(summaries)
    df.to_csv(out_dir / "lo_alignment_summary.csv", index=False)

    # Summary plots across runs
    _plot_best_sim(
        all_best_sims,
        out_dir / "lo_similarity_hist_all.png",
        title="Best LO->chunk similarity (all runs)",
    )

    print("Done. Outputs:", out_dir.resolve())


if __name__ == "__main__":
    main()
