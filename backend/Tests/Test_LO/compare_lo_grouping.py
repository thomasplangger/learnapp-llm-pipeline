import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve()
BACKEND_DIR = HERE.parents[1]
if (BACKEND_DIR / "app").exists() and str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
elif (Path.cwd() / "app").exists() and str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

try:
    from dotenv import load_dotenv  # type: ignore
    _env_path = BACKEND_DIR / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except Exception:
    pass

try:
    from app.providers import AI
except ModuleNotFoundError:
    # Fallback: add repo root, then retry
    repo_root = HERE.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from app.providers import AI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PALETTE = ["#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#a855f7", "#06b6d4", "#84cc16", "#f97316", "#14b8a6", "#eab308"]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    try:
        return AI.batch_embed_texts(texts)
    except Exception:
        return [AI.embed_text(t) for t in texts]


def _ensure_embeddings(meta: List[Dict[str, Any]], chunks: List[str]) -> List[Dict[str, Any]]:
    # If embeddings are missing, generate them from metadata text, else fallback to chunk text.
    if meta and any(m.get("embedding") for m in meta):
        return meta

    embed_inputs = []
    for m in meta:
        embed_inputs.append(
            "\n".join(filter(None, [m.get("title"), m.get("summary"), ", ".join(m.get("keywords", []))]))
        )
    vecs = _embed_texts(embed_inputs) if embed_inputs else []

    if vecs and any(v for v in vecs):
        for m, v in zip(meta, vecs):
            m["embedding"] = v or []
        return meta

    # Fallback: embed chunk text directly
    vecs = _embed_texts(chunks)
    if vecs and any(v for v in vecs):
        for i, v in enumerate(vecs):
            if i < len(meta):
                meta[i]["embedding"] = v or []
            else:
                meta.append({"chunk_index": i, "embedding": v or []})
        return meta

    return meta


def _load_or_generate_metadata(run_dir: Path, chunks: List[str], *, reuse: bool, cache_dir: Path) -> List[Dict[str, Any]]:
    meta_path_cache = cache_dir / "chunk_metadata.json"
    meta_path_run = run_dir / "chunk_metadata.json"
    if reuse:
        if meta_path_cache.exists():
            meta = _read_json(meta_path_cache)
            meta = _ensure_embeddings(meta, chunks)
            if meta and any(m.get("embedding") for m in meta):
                return meta
        if meta_path_run.exists():
            meta = _read_json(meta_path_run)
            meta = _ensure_embeddings(meta, chunks)
            if meta and any(m.get("embedding") for m in meta):
                return meta

    items = [{"index": i, "text": c[:1200]} for i, c in enumerate(chunks)]
    meta = AI.generate_chunk_metadata(items)
    meta = _ensure_embeddings(meta, chunks)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path_cache.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _extract_embeddings(meta: List[Dict[str, Any]]) -> List[List[float]]:
    vecs: List[List[float]] = []
    dim = None
    for m in meta:
        v = m.get("embedding") or []
        if dim is None and v:
            dim = len(v)
        vecs.append(list(v) if v else [])

    if dim is None:
        return [[] for _ in meta]

    out = []
    for v in vecs:
        vv = list(v)[:dim]
        if len(vv) < dim:
            vv += [0.0] * (dim - len(vv))
        out.append(vv)
    return out


def _pca_2d(X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros((0, 2))
    if X.shape[1] < 2:
        return np.zeros((X.shape[0], 2))
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    comps = vt[:2].T
    return Xc @ comps


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms

def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", (text or "").lower())]

def _tfidf_vectors(texts: List[str]) -> np.ndarray:
    tokens_per = [_tokenize(t) for t in texts]
    vocab: Dict[str, int] = {}
    for toks in tokens_per:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    if not vocab:
        return np.zeros((len(texts), 0), dtype=float)
    df = np.zeros(len(vocab), dtype=float)
    for toks in tokens_per:
        for t in set(toks):
            df[vocab[t]] += 1.0
    N = max(1, len(texts))
    idf = np.log((1.0 + N) / (1.0 + df)) + 1.0
    X = np.zeros((len(texts), len(vocab)), dtype=float)
    for i, toks in enumerate(tokens_per):
        if not toks:
            continue
        counts: Dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        total = float(len(toks))
        for t, c in counts.items():
            X[i, vocab[t]] = (c / total) * idf[vocab[t]]
    return X


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) or 1.0
    nb = np.linalg.norm(b) or 1.0
    return float(np.dot(a, b) / (na * nb))


def _kmeans_cosine(X: List[List[float]], K: int, seed: int = 42, iters: int = 20) -> List[int]:
    if not X:
        return []
    Xn = np.array(X, dtype=float)
    Xn = _normalize_rows(Xn)
    M = Xn.shape[0]
    K = max(1, min(K, M))
    rng = np.random.default_rng(seed)
    cents = Xn[rng.choice(M, K, replace=False)]
    labels = np.zeros(M, dtype=int)
    for _ in range(iters):
        sims = Xn @ cents.T
        labels = np.argmax(sims, axis=1)
        for k in range(K):
            members = Xn[labels == k]
            if len(members) == 0:
                cents[k] = Xn[rng.integers(0, M)]
            else:
                v = members.mean(axis=0)
                v = v / (np.linalg.norm(v) or 1.0)
                cents[k] = v
    return labels.tolist()


def _llm_group(meta: List[Dict[str, Any]], desired_N: int | None) -> Dict[str, Any]:
    items = []
    for i, m in enumerate(meta):
        items.append({
            "index": i,
            "title": m.get("title") or "",
            "topic": m.get("topic") or "",
            "keywords": (m.get("keywords") or [])[:10],
            "summary": m.get("summary") or "",
        })
    return AI.group_chunks_semantic(items, desired_N=desired_N)


def _labels_from_llm(groups: List[Dict[str, Any]], total: int) -> Tuple[List[int], int]:
    labels = [-1] * total
    for gi, g in enumerate(groups):
        for idx in g.get("members") or []:
            try:
                ii = int(idx)
            except Exception:
                continue
            if 0 <= ii < total:
                labels[ii] = gi
    unassigned = sum(1 for x in labels if x < 0)
    return labels, unassigned


def _group_metrics(labels: List[int], X: np.ndarray) -> Dict[str, float]:
    total = len(labels)
    valid = [i for i, g in enumerate(labels) if g >= 0]
    groups = sorted(set(g for g in labels if g >= 0))
    if not groups or not valid:
        return {
            "groups": 0,
            "unassigned": total,
            "coverage_rate": 0.0,
            "mean_chunks_per_group": 0.0,
            "median_chunks_per_group": 0.0,
            "std_chunks_per_group": 0.0,
            "mean_intra_cosine": 0.0,
            "mean_inter_cosine": 0.0,
            "gini_group_sizes": 0.0,
            "max_group_share": 0.0,
            "silhouette_cosine": 0.0,
        }

    sizes = []
    intra = []
    Xn = _normalize_rows(X)
    for g in groups:
        idxs = [i for i, lab in enumerate(labels) if lab == g]
        sizes.append(len(idxs))
        if len(idxs) == 1:
            intra.append(1.0)
            continue
        centroid = Xn[idxs].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        sims = [float(np.dot(Xn[i], centroid)) for i in idxs]
        intra.append(float(np.mean(sims)))

    # Mean inter-group cosine similarity (centroid-to-centroid)
    centroids = []
    for g in groups:
        idxs = [i for i, lab in enumerate(labels) if lab == g]
        c = Xn[idxs].mean(axis=0)
        c = c / (np.linalg.norm(c) or 1.0)
        centroids.append(c)
    inter = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter.append(float(np.dot(centroids[i], centroids[j])))
    mean_inter = float(np.mean(inter)) if inter else 0.0

    # Gini for group size imbalance
    sizes_sorted = sorted(sizes)
    n = len(sizes_sorted)
    if n > 0:
        cum = 0.0
        for i, v in enumerate(sizes_sorted, start=1):
            cum += i * v
        gini = (2.0 * cum) / (n * sum(sizes_sorted)) - (n + 1) / n
    else:
        gini = 0.0

    max_share = max(sizes) / max(1, sum(sizes))

    # Silhouette (cosine distance) with sampling for speed
    def _silhouette_cosine(sample_idx: List[int]) -> float:
        if len(sample_idx) < 2 or len(groups) < 2:
            return 0.0
        sims = Xn[sample_idx] @ Xn.T
        dists = 1.0 - sims
        sil_vals = []
        for ii in sample_idx:
            gi = labels[ii]
            same = [j for j in valid if labels[j] == gi and j != ii]
            if same:
                a = float(np.mean(dists[ii, same]))
            else:
                a = 0.0
            b = None
            for g in groups:
                if g == gi:
                    continue
                others = [j for j in valid if labels[j] == g]
                if not others:
                    continue
                dist = float(np.mean(dists[ii, others]))
                b = dist if b is None else min(b, dist)
            if b is None:
                continue
            sil = (b - a) / max(a, b) if max(a, b) > 1e-9 else 0.0
            sil_vals.append(sil)
        return float(np.mean(sil_vals)) if sil_vals else 0.0

    if len(valid) > 400:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(valid, size=min(300, len(valid)), replace=False).tolist()
    else:
        sample_idx = valid
    silhouette = _silhouette_cosine(sample_idx)

    return {
        "groups": float(len(groups)),
        "unassigned": float(total - len(valid)),
        "coverage_rate": float(len(valid) / max(1, total)),
        "mean_chunks_per_group": float(np.mean(sizes)),
        "median_chunks_per_group": float(np.median(sizes)),
        "std_chunks_per_group": float(np.std(sizes)),
        "mean_intra_cosine": float(np.mean(intra)),
        "mean_inter_cosine": mean_inter,
        "gini_group_sizes": float(gini),
        "max_group_share": float(max_share),
        "silhouette_cosine": float(silhouette),
    }


def _topic_similarity(meta: List[Dict[str, Any]], labels: List[int]) -> float:
    groups = sorted(set(g for g in labels if g >= 0))
    sims = []
    for g in groups:
        idxs = [i for i, lab in enumerate(labels) if lab == g]
        topics = []
        for i in idxs:
            t = (meta[i].get("topic") or "").strip().lower()
            if t:
                topics.append(set(re.findall(r"[a-z0-9]+", t)))
        if len(topics) < 2:
            continue
        pair_sims = []
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                a, b = topics[i], topics[j]
                if not a or not b:
                    continue
                pair_sims.append(len(a & b) / max(1, len(a | b)))
        if pair_sims:
            sims.append(float(np.mean(pair_sims)))
    return float(np.mean(sims)) if sims else 0.0


def _keyword_similarity(meta: List[Dict[str, Any]], labels: List[int]) -> float:
    groups = sorted(set(g for g in labels if g >= 0))
    sims = []
    for g in groups:
        idxs = [i for i, lab in enumerate(labels) if lab == g]
        kws = []
        for i in idxs:
            ks = meta[i].get("keywords") or []
            if ks:
                toks = set()
                for k in ks:
                    toks.update(re.findall(r"[a-z0-9]+", str(k).lower()))
                if toks:
                    kws.append(toks)
        if len(kws) < 2:
            continue
        pair_sims = []
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                a, b = kws[i], kws[j]
                if not a or not b:
                    continue
                pair_sims.append(len(a & b) / max(1, len(a | b)))
        if pair_sims:
            sims.append(float(np.mean(pair_sims)))
    return float(np.mean(sims)) if sims else 0.0


def _group_summary_text(group: Dict[str, Any], meta: List[Dict[str, Any]], members: List[int]) -> str:
    title = (group.get("title") or "").strip()
    summary = (group.get("summary") or "").strip()
    keywords = group.get("keywords") or []
    if title or summary or keywords:
        return "\n".join(filter(None, [title, summary, ", ".join([str(k) for k in keywords])]))

    # Fallback: derive from member meta
    topic_counts: Dict[str, int] = {}
    keyword_counts: Dict[str, int] = {}
    for i in members:
        t = (meta[i].get("topic") or "").strip()
        if t:
            topic_counts[t] = topic_counts.get(t, 0) + 1
        for k in (meta[i].get("keywords") or []):
            ks = str(k).strip()
            if ks:
                keyword_counts[ks] = keyword_counts.get(ks, 0) + 1

    top_topics = [k for k, _ in sorted(topic_counts.items(), key=lambda x: (-x[1], x[0]))[:3]]
    top_kw = [k for k, _ in sorted(keyword_counts.items(), key=lambda x: (-x[1], x[0]))[:8]]
    return "\n".join(filter(None, [", ".join(top_topics), ", ".join(top_kw)]))


def _summary_alignment_metrics(
    labels: List[int],
    groups: List[Dict[str, Any]],
    meta: List[Dict[str, Any]],
    X: np.ndarray,
    *,
    use_embedding_model: bool,
    fallback_tfidf: bool,
) -> Dict[str, float]:
    groups_idx = sorted(set(g for g in labels if g >= 0))
    if not groups_idx:
        return {"mean_summary_chunk_sim": 0.0, "summary_margin": 0.0}

    Xn = _normalize_rows(X)
    group_sims = []
    margins = []
    for gi in groups_idx:
        members = [i for i, lab in enumerate(labels) if lab == gi]
        if not members:
            continue
        non_members = [i for i, lab in enumerate(labels) if lab != gi and lab >= 0]
        if use_embedding_model:
            g = groups[gi] if gi < len(groups) else {}
            text = _group_summary_text(g, meta, members)
            vecs = _embed_texts([text]) if text else []
            if not vecs or not vecs[0]:
                if fallback_tfidf:
                    centroid = Xn[members].mean(axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) or 1.0)
                else:
                    continue
            else:
                centroid = np.array(vecs[0], dtype=float)
                centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        else:
            centroid = Xn[members].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) or 1.0)

        mem_sims = [float(np.dot(Xn[i], centroid)) for i in members]
        group_sims.append(float(np.mean(mem_sims)) if mem_sims else 0.0)
        if non_members:
            non_sims = [float(np.dot(Xn[i], centroid)) for i in non_members]
            margins.append(float(np.mean(mem_sims) - np.mean(non_sims)))
        else:
            margins.append(0.0)

    return {
        "mean_summary_chunk_sim": float(np.mean(group_sims)) if group_sims else 0.0,
        "summary_margin": float(np.mean(margins)) if margins else 0.0,
    }


def _plot_scatter(coords: np.ndarray, labels: List[int], title: str, out_path: Path, *, show_legend: bool = True):
    plt.figure(figsize=(7, 5))
    labels_arr = np.array(labels)
    unique = [u for u in sorted(set(labels_arr)) if u >= 0]
    for i, u in enumerate(unique):
        idxs = np.where(labels_arr == u)[0]
        color = PALETTE[i % len(PALETTE)]
        plt.scatter(coords[idxs, 0], coords[idxs, 1], s=20, alpha=0.7, color=color, label=f"LO{u+1}")

    unassigned = np.where(labels_arr < 0)[0]
    if len(unassigned):
        plt.scatter(coords[unassigned, 0], coords[unassigned, 1], s=16, alpha=0.5, color="gray", label="unassigned")

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    if show_legend and len(unique) <= 12:
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _build_eval_matrix(chunks: List[str], meta: List[Dict[str, Any]], embeds_meta: List[List[float]], *, eval_mode: str, fallback_tfidf: bool) -> Tuple[np.ndarray, List[List[float]]]:
    if eval_mode == "fulltext":
        vecs = _embed_texts(chunks)
        if vecs and any(v for v in vecs):
            X = np.array(vecs, dtype=float)
            return X, vecs
        if fallback_tfidf:
            X = _tfidf_vectors(chunks)
            if X.shape[1] == 0:
                raise SystemExit("TF-IDF fallback produced no features (empty texts).")
            return X, X.tolist()
        raise SystemExit("No full-text embeddings available.")

    # default: metadata embeddings
    if embeds_meta and any(len(v) for v in embeds_meta):
        X = np.array(embeds_meta, dtype=float)
        return X, embeds_meta
    if fallback_tfidf:
        X = _tfidf_vectors(chunks)
        if X.shape[1] == 0:
            raise SystemExit("TF-IDF fallback produced no features (empty texts).")
        return X, X.tolist()
    raise SystemExit("No embeddings found in metadata.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testrun-dir", required=True, help="Path to testrun_* directory")
    ap.add_argument("--run-index", type=int, default=None, help="Run index (default: all runs)")
    ap.add_argument("--N", type=int, default=None, help="Desired LO count (optional)")
    ap.add_argument("--reuse-metadata", action="store_true", help="Reuse chunk_metadata.json if present")
    ap.add_argument("--fallback-tfidf", action="store_true", help="If embeddings are missing, use TF-IDF vectors")
    ap.add_argument("--eval-embeddings", default="metadata", choices=["metadata", "fulltext"], help="Embeddings for evaluation/plots (default: metadata)")
    ap.add_argument("--output-root", default=None, help="Output root (default: backend/Tests/Test_LO)")
    args = ap.parse_args()

    testrun = Path(args.testrun_dir)
    if not testrun.exists():
        raise SystemExit(f"testrun dir not found: {testrun}")

    run_dirs = sorted([p for p in testrun.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if args.run_index is not None:
        run_dirs = [testrun / f"run_{args.run_index}"]
    run_dirs = [p for p in run_dirs if p.exists()]
    if not run_dirs:
        raise SystemExit("No run directories found.")

    output_root = Path(args.output_root) if args.output_root else (Path(__file__).resolve().parent / "Test_LO")
    out_testrun = output_root / testrun.name
    out_testrun.mkdir(parents=True, exist_ok=True)

    # First pass: load chunks + metadata, and collect embeddings availability
    runs_payload: List[Dict[str, Any]] = []
    all_chunks: List[str] = []
    all_meta: List[List[Dict[str, Any]]] = []
    all_embeds: List[List[List[float]]] = []
    offsets: List[Tuple[int, int]] = []

    for run_dir in run_dirs:
        paras = _read_json(run_dir / "paragraphs_used.json")
        pred = _read_json(run_dir / "llm_output.context.json")
        bounds = _normalize_bounds(pred.get("boundaries") or [], len(paras))
        chunks = _build_chunks(paras, bounds)
        if not chunks:
            print(f"[SKIP] {run_dir.name}: no chunks")
            continue

        run_out = out_testrun / run_dir.name
        meta = _load_or_generate_metadata(run_dir, chunks, reuse=bool(args.reuse_metadata), cache_dir=run_out)
        embeds = _extract_embeddings(meta)

        start = len(all_chunks)
        all_chunks.extend(chunks)
        all_meta.append(meta)
        all_embeds.append(embeds)
        offsets.append((start, len(all_chunks)))
        runs_payload.append({
            "run_dir": run_dir,
            "chunks": chunks,
            "meta": meta,
            "embeds": embeds,
            "out_dir": run_out,
        })

    if not runs_payload:
        raise SystemExit("No runs with chunks to process.")

    # Build evaluation vectors (metadata or full-text embeddings)
    if args.eval_embeddings == "metadata":
        embeds_ok = all(any(len(v) for v in emb) for emb in all_embeds)
        if embeds_ok:
            X_all = np.array([v for emb in all_embeds for v in emb], dtype=float)
            embeds_for_kmeans_all = [v for emb in all_embeds for v in emb]
            eval_uses_embedding_model = True
        else:
            if not args.fallback_tfidf:
                raise SystemExit("No embeddings found in metadata.")
            X_all = _tfidf_vectors(all_chunks)
            if X_all.shape[1] == 0:
                raise SystemExit("TF-IDF fallback produced no features (empty texts).")
            embeds_for_kmeans_all = X_all.tolist()
            eval_uses_embedding_model = False
    else:
        X_all, embeds_for_kmeans_all = _build_eval_matrix(all_chunks, [], [], eval_mode="fulltext", fallback_tfidf=bool(args.fallback_tfidf))
        eval_uses_embedding_model = bool(args.eval_embeddings == "fulltext")

    coords_all = _pca_2d(X_all)

    summary_rows: List[Dict[str, Any]] = []
    combined_llm_labels: List[int] = []
    combined_heur_labels: List[int] = []

    for idx, payload in enumerate(runs_payload):
        run_dir = payload["run_dir"]
        chunks = payload["chunks"]
        meta = payload["meta"]
        run_out = payload["out_dir"]
        run_out.mkdir(parents=True, exist_ok=True)

        start, end = offsets[idx]
        X_run = X_all[start:end]
        coords_run = coords_all[start:end]
        embeds_run = embeds_for_kmeans_all[start:end]

        if args.N is not None:
            N = max(1, int(args.N))
        else:
            N = max(3, min(10, int(math.sqrt(max(1, len(chunks))))))

        grouped = _llm_group(meta, desired_N=N)
        llm_groups = grouped.get("groups") or []
        llm_labels, llm_unassigned = _labels_from_llm(llm_groups, len(chunks))
        llm_metrics = _group_metrics(llm_labels, X_run)
        llm_metrics["unassigned"] = float(llm_unassigned)
        llm_metrics["chunks"] = float(len(chunks))
        llm_metrics["run"] = run_dir.name
        llm_metrics["method"] = "llm"
        llm_metrics["topic_similarity"] = _topic_similarity(meta, llm_labels)
        llm_metrics["keyword_similarity"] = _keyword_similarity(meta, llm_labels)
        llm_metrics.update(
            _summary_alignment_metrics(
                llm_labels,
                llm_groups,
                meta,
                X_run,
                use_embedding_model=eval_uses_embedding_model,
                fallback_tfidf=bool(args.fallback_tfidf),
            )
        )

        heur_labels = _kmeans_cosine(embeds_run, K=N)
        heur_metrics = _group_metrics(heur_labels, X_run)
        heur_metrics["unassigned"] = 0.0
        heur_metrics["chunks"] = float(len(chunks))
        heur_metrics["run"] = run_dir.name
        heur_metrics["method"] = "heuristic"
        heur_metrics["topic_similarity"] = _topic_similarity(meta, heur_labels)
        heur_metrics["keyword_similarity"] = _keyword_similarity(meta, heur_labels)
        heur_metrics.update(
            _summary_alignment_metrics(
                heur_labels,
                [],
                meta,
                X_run,
                use_embedding_model=eval_uses_embedding_model,
                fallback_tfidf=bool(args.fallback_tfidf),
            )
        )

        (run_out / "lo_grouping_llm_raw.json").write_text(json.dumps(grouped, indent=2), encoding="utf-8")
        if not llm_groups:
            print(f"[WARN] {run_dir.name}: LLM grouping returned no groups.")
            print(f"[WARN] AI provider type: {type(AI).__name__}")

        _plot_scatter(coords_run, llm_labels, "Chunks mapped to LOs (LLM grouping)", run_out / "lo_scatter_llm.png")
        _plot_scatter(coords_run, heur_labels, "Chunks mapped to LOs (heuristic grouping)", run_out / "lo_scatter_heuristic.png")

        (run_out / "lo_grouping_llm.json").write_text(json.dumps(llm_groups, indent=2), encoding="utf-8")
        (run_out / "lo_grouping_heuristic.json").write_text(json.dumps({"labels": heur_labels}, indent=2), encoding="utf-8")
        (run_out / "lo_grouping_compare_summary.csv").write_text(
            pd.DataFrame([llm_metrics, heur_metrics]).to_csv(index=False),
            encoding="utf-8",
        )

        summary_rows.extend([llm_metrics, heur_metrics])
        combined_llm_labels.extend(llm_labels)
        combined_heur_labels.extend(heur_labels)

    summary_df = pd.DataFrame(summary_rows)
    if "run" in summary_df.columns and "method" in summary_df.columns:
        order = pd.Categorical(summary_df["method"], categories=["llm", "heuristic"], ordered=True)
        summary_df = summary_df.assign(_method_order=order).sort_values(["_method_order", "run"]).drop(columns=["_method_order"])
    cols = ["run"] + [c for c in summary_df.columns if c != "run"]
    summary_df = summary_df[cols]
    summary_df.to_csv(out_testrun / "lo_grouping_compare_summary_all.csv", index=False)

    _plot_scatter(
        coords_all,
        combined_llm_labels,
        "Chunks mapped to LOs (LLM grouping) — all runs",
        out_testrun / "lo_scatter_llm_all.png",
        show_legend=False,
    )
    _plot_scatter(
        coords_all,
        combined_heur_labels,
        "Chunks mapped to LOs (heuristic grouping) — all runs",
        out_testrun / "lo_scatter_heuristic_all.png",
        show_legend=False,
    )

    print("Done. Outputs:", out_testrun.resolve())


if __name__ == "__main__":
    main()
