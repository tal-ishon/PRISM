#!/usr/bin/env python3
"""
Generate topic-word heatmaps (words=rows, topics=cols) for every model
under:
  BASE/Distributions-Results/<run_num>/<dataset>/<num_topics>/

Saves to:
  OUT/figures/<dataset>/<num_topics>/<model>_heatmap.png

Supported input formats inside each <num_topics> dir:
1) CSV/TSV where first column is words (string) and remaining columns are topics.
   - Header row optional; if present, topic columns can be named.
2) NPZ with arrays:
   - phi: 2D numpy array (K,V) or (V,K)
   - vocab: 1D array/list of words length V
3) NPY matrix (K,V) or (V,K) plus a vocab file in the same dir:
   - vocab.txt or vocab.tsv or vocab.csv (one word per line OR first column)
"""

import argparse
from html import parser
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

MAX_FIG_W_IN = 18   # paper-friendly
MAX_FIG_H_IN = 24
MAX_PIX = 65000     # must be < 2^16

# ----------------------------- IO helpers ----------------------------- #

def _read_vocab_file(vocab_path: Path) -> List[str]:
    # Supports one word per line OR CSV/TSV with words in first column
    words = []
    with vocab_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split by comma/tab if present, take first token
            token = re.split(r"[,\t]", line)[0].strip()
            if token:
                words.append(token)
    return words


def _try_find_vocab_in_dir(d: Path) -> Optional[List[str]]:
    for name in ["vocab.txt", "vocabulary.txt", "vocab.tsv", "vocab.csv", "words.txt"]:
        p = d / name
        if p.exists():
            return _read_vocab_file(p)
    # fallback: any file matching *vocab*.* or *vocabulary*.*
    for p in d.glob("*vocab*.*"):
        if p.is_file():
            return _read_vocab_file(p)
    for p in d.glob("*vocabulary*.*"):
        if p.is_file():
            return _read_vocab_file(p)
    return None


def load_phi_from_npz(npz_path: Path) -> Optional[Tuple[np.ndarray, List[str]]]:
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None

    # Common keys
    phi = None
    vocab = None
    for k in ["phi", "Phi", "topic_word", "beta", "topic_word_dist"]:
        if k in data:
            phi = data[k]
            break
    for k in ["vocab", "vocabulary", "words", "term", "terms"]:
        if k in data:
            vocab = data[k]
            break

    if phi is None or vocab is None:
        return None

    vocab_list = [str(x) for x in list(vocab)]
    phi = np.asarray(phi, dtype=float)

    return phi, vocab_list


def load_phi_from_npy(npy_path: Path, vocab: Optional[List[str]]) -> Optional[Tuple[np.ndarray, List[str]]]:
    try:
        mat = np.load(npy_path, allow_pickle=True)
    except Exception:
        return None

    mat = np.asarray(mat, dtype=float)
    if vocab is None:
        # try to find vocab in same directory
        vocab = _try_find_vocab_in_dir(npy_path.parent)
    if vocab is None:
        return None

    return mat, vocab


def load_phi_from_delimited(file_path: Path) -> Optional[Tuple[np.ndarray, List[str], List[str]]]:
    """
    Supports two delimited layouts:

    A) words x topics  (your intended format)
       header: word, T0, T1, ...
       rows:   token, p(w|T0), p(w|T1), ...

    B) topics x words  (common alternative)
       header: topic, w1, w2, ...
       rows:   T0, p(w1|T0), p(w2|T0), ...
    """
    sep = "\t" if file_path.suffix.lower() in [".tsv", ".tab"] else ","
    try:
        raw = file_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    if not raw:
        return None

    first = raw[0].strip()
    if not first:
        return None
    parts = first.split(sep)
    if len(parts) < 2:
        return None

    def _is_float(x: str) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    has_header = not _is_float(parts[1])
    start_idx = 1 if has_header else 0

    header_cols = [p.strip() for p in parts]
    header_tail = [p.strip() for p in parts[1:]] if has_header else []

    row_ids: List[str] = []
    rows: List[List[float]] = []
    for line in raw[start_idx:]:
        line = line.strip()
        if not line:
            continue
        cols = line.split(sep)
        if len(cols) < 2:
            continue
        rid = cols[0].strip()
        vals = cols[1:]
        try:
            row = [float(v) for v in vals]
        except Exception:
            continue
        row_ids.append(rid)
        rows.append(row)

    if not rows:
        return None

    mat = np.asarray(rows, dtype=float)

    if not has_header:
        # No header: assume words x topics (cannot infer vocab otherwise)
        vocab = row_ids
        topic_labels = [f"T{j}" for j in range(mat.shape[1])]
        return mat, vocab, topic_labels

    # --- Orientation detection ---
    # If header tail is huge (likely words), and number of rows is small (likely topics),
    # treat as topics x words and transpose.
    V_like = len(header_tail)
    K_like = len(row_ids)

    if V_like > K_like and V_like > 50 and K_like <= 500:
        # Layout B: topics x words
        vocab = header_tail                      # words
        topic_labels = row_ids                   # topics
        mat_wt = mat.T                           # -> words x topics
        return mat_wt, vocab, topic_labels

    # Otherwise assume Layout A: words x topics
    vocab = row_ids
    topic_labels = header_tail if header_tail else [f"T{j}" for j in range(mat.shape[1])]
    return mat, vocab, topic_labels

# ----------------------------- selection logic ----------------------------- #

def row_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = mat.sum(axis=1, keepdims=True)
    return mat / np.clip(s, eps, None)


def ensure_words_by_topics(
    mat: np.ndarray,
    vocab: List[str],
    topic_labels: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Ensure matrix is words x topics for plotting (words rows, topics cols).
    Accepts mat in either (V,K) or (K,V).
    """
    mat = np.asarray(mat, dtype=float)
    V = len(vocab)

    if mat.shape[0] == V:
        # already words x topics
        K = mat.shape[1]
        
        if topic_labels is None or len(topic_labels) != K:
            topic_labels = [f"T{j}" for j in range(K)]
        return mat, vocab, topic_labels

    if mat.shape[1] == V:
            mat = mat.T
            K = mat.shape[1]

            # If topic_labels mistakenly holds words (len==V), discard and regenerate
            if topic_labels is None or len(topic_labels) != K:
                topic_labels = [f"T{j}" for j in range(K)]
            return mat, vocab, topic_labels

    # Optional recovery: sometimes topic_labels==V indicates header words got misread
    # In that case, do NOT try to guess; the loader should be fixed.
    raise ValueError(
        f"Cannot infer orientation: mat shape={mat.shape}, |vocab|={V}. "
        "Expected (V,K) or (K,V). Check loader/delimiter/header."
    )

def reorder_words_by_dominant_topic(mat_sel, vocab_sel):
    """
    mat_sel: words x topics
    Returns reordered (mat, vocab, boundaries) where boundaries are row indices
    to draw separators between topic blocks.
    """
    # dominant topic per word
    dom = np.argmax(mat_sel, axis=1)
    dom_val = mat_sel[np.arange(mat_sel.shape[0]), dom]

    # sort by (dominant topic, dominant prob desc)
    order = np.lexsort((-dom_val, dom))
    mat_r = mat_sel[order, :]
    vocab_r = [vocab_sel[i] for i in order]
    dom_r = dom[order]

    # boundaries where dominant topic changes
    boundaries = []
    for i in range(1, len(dom_r)):
        if dom_r[i] != dom_r[i-1]:
            boundaries.append(i)
    return mat_r, vocab_r, boundaries


def union_top_words_per_topic(mat_words_topics: np.ndarray, vocab: List[str], top_n: int) -> List[int]:
    """
    Select union of top_n words per topic (by probability/weight).
    Returns word indices.
    """
    W, K = mat_words_topics.shape
    top_idx = set()
    for k in range(K):
        idx = np.argsort(mat_words_topics[:, k])[::-1][:top_n]
        top_idx.update(idx.tolist())

    # Order by total mass across topics (more informative columns first)
    ordered = sorted(list(top_idx), key=lambda i: mat_words_topics[i, :].sum(), reverse=True)
    return ordered


def log_transform(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(mat, eps, None))


# ----------------------------- plotting ----------------------------- #

def plot_heatmap_words_topics(
    mat_words_topics: np.ndarray,
    vocab: List[str],
    topic_labels: List[str],
    out_path: Path,
    title: str,
    top_n: int = 12,
    log_scale: bool = True,
    max_topics: Optional[int] = None,
    max_words: Optional[int] = None,
    dpi: int = 200,
):
    """
    mat_words_topics: words x topics
    """
    mat = np.asarray(mat_words_topics, dtype=float)
    mat, vocab, topic_labels = ensure_words_by_topics(mat, vocab, topic_labels)

    # Optional limit topics (columns) for readability
    if max_topics is not None and mat.shape[1] > max_topics:
        # choose topics with highest peakiness
        peaks = mat.max(axis=0)
        keep_cols = np.argsort(peaks)[::-1][:max_topics]
        keep_cols = np.sort(keep_cols)  # stable left-to-right
        mat = mat[:, keep_cols]
        topic_labels = [topic_labels[j] for j in keep_cols]

    # Select words by union of top_n per remaining topic
    keep_words = union_top_words_per_topic(mat, vocab, top_n=top_n)
    if max_words is not None and len(keep_words) > max_words:
        keep_words = keep_words[:max_words]

    mat_sel = mat[keep_words, :]
    vocab_sel = [vocab[i] for i in keep_words]

    # Normalize per topic for comparability (optional but usually helpful)
    # convert words x topics -> topics x words, normalize each topic, back
    mat_tw = mat_sel.T
    mat_tw = row_normalize(mat_tw)
    mat_sel = mat_tw.T
    
    # NEW: reorder words by dominant topic (creates readable blocks)
    mat_sel, vocab_sel, boundaries = reorder_words_by_dominant_topic(mat_sel, vocab_sel)

    # NEW: optional topic reordering can help too (comment out if you want original order)
    # order_topics = np.argsort(mat_sel.max(axis=0))[::-1]
    # mat_sel = mat_sel[:, order_topics]
    # topic_labels = [topic_labels[i] for i in order_topics]

    mat_plot = log_transform(mat_sel) if log_scale else mat_sel

    # Figure size heuristic
    width = max(6, 0.35 * mat_plot.shape[1])
    height = max(6, 0.20 * mat_plot.shape[0])
    
    # ---- SAFE FIGURE SIZE ----
    n_words, n_topics = mat_plot.shape  # words x topics

    # Heuristic inch sizing, but bounded
    width_in  = min(MAX_FIG_W_IN, max(6, 0.45 * n_topics))
    height_in = min(MAX_FIG_H_IN, max(6, 0.22 * n_words))

    # Ensure pixel bounds are respected
    # (Matplotlib fails if width_in*dpi or height_in*dpi >= 2^16)
    while int(width_in * dpi) >= MAX_PIX:
        width_in *= 0.9
    while int(height_in * dpi) >= MAX_PIX:
        height_in *= 0.9

    plt.figure(figsize=(width_in, height_in))
    # plt.figure(figsize=(width, height))
    im = plt.imshow(mat_plot, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.02, pad=0.02)

    # NEW: draw separators between topic blocks
    for b in boundaries:
        plt.axhline(b - 0.5, linewidth=1)

    plt.title(title)
    plt.xticks(range(len(topic_labels)), topic_labels, rotation=45, ha="right", fontsize=9)
    plt.yticks(range(len(vocab_sel)), vocab_sel, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# ----------------------------- discovery logic ----------------------------- #

def discover_model_files(num_topics_dir: Path) -> List[Path]:
    """
    Heuristic: treat each CSV/TSV/NPZ/NPY file as a potential model output,
    skipping obvious non-distribution files.
    """
    exts = {".csv", ".tsv", ".tab", ".npz", ".npy"}
    candidates = []
    for p in num_topics_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        name = p.name.lower()
        # skip typical non-phi artifacts
        if any(x in name for x in ["theta", "doc_topic", "doctopic", "doc-topic", "doc_topics"]):
            continue
        if any(x in name for x in ["state", "assign", "z_", "topic_assign", "topicassign"]):
            continue
        candidates.append(p)
    return sorted(candidates)


def model_name_from_file(p: Path) -> str:
    base = p.stem
    # Remove common prefixes/suffixes to get a clean model id
    base = re.sub(r"(topic[_-]?word|word[_-]?topic|phi|beta|distribution|dist)", "", base, flags=re.I)
    base = re.sub(r"__+", "_", base)
    base = base.strip("_- ")
    return base if base else p.stem


def load_any_distribution(file_path: Path, vocab_hint: Optional[List[str]]) -> Optional[Tuple[np.ndarray, List[str], List[str]]]:
    """
    Return (words x topics matrix, vocab list, topic_labels).
    """
    if file_path.suffix.lower() in [".csv", ".tsv", ".tab"]:
        loaded = load_phi_from_delimited(file_path)
        if loaded is None:
            return None
        mat_wt, vocab, topic_labels = loaded
        return mat_wt, vocab, topic_labels

    if file_path.suffix.lower() == ".npz":
        loaded = load_phi_from_npz(file_path)
        if loaded is None:
            return None
        mat, vocab = loaded
        mat_wt, vocab, topic_labels = ensure_words_by_topics(mat, vocab, None)
        return mat_wt, vocab, topic_labels

    if file_path.suffix.lower() == ".npy":
        loaded = load_phi_from_npy(file_path, vocab_hint)
        if loaded is None:
            return None
        mat, vocab = loaded
        mat_wt, vocab, topic_labels = ensure_words_by_topics(mat, vocab, None)
        return mat_wt, vocab, topic_labels

    return None


def generate_all_heatmaps(
    base_dir: Path,
    run_num: str,
    out_dir: Path,
    top_n: int,
    log_scale: bool,
    max_topics: Optional[int],
    max_words: Optional[int],
    dataset_filter: Optional[str] = None,
    num_topics_filter: Optional[str] = None,
):

    run_dir = base_dir / "Distributions-Results" / run_num
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    # Loop datasets
    for dataset_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        dataset = dataset_dir.name
        
        if dataset_filter is not None and dataset != dataset_filter:
            continue

        # Loop num_topics subdirs
        for k_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            num_topics = k_dir.name
            
            if num_topics_filter is not None and num_topics != num_topics_filter:
                continue

            # vocab hint if available
            vocab_hint = _try_find_vocab_in_dir(k_dir)

            # discover model outputs
            files = discover_model_files(k_dir)
            if not files:
                continue

            for f in files:
                model = model_name_from_file(f)

                loaded = load_any_distribution(f, vocab_hint=vocab_hint)
                if loaded is None:
                    continue
                mat_wt, vocab, topic_labels = loaded

                # Make topic labels consistent / readable
                # If labels look like numbers, prefix with T
                cleaned_labels = []
                for j, lab in enumerate(topic_labels):
                    lab = str(lab)
                    if re.fullmatch(r"\d+", lab):
                        lab = f"T{lab}"
                    cleaned_labels.append(lab)
                topic_labels = cleaned_labels

                fig_path = out_dir / "figures" / dataset / num_topics / f"{model}_heatmap.png"
                title = f"{dataset} | K={num_topics} | {model}"

                plot_heatmap_words_topics(
                    mat_words_topics=mat_wt,
                    vocab=vocab,
                    topic_labels=topic_labels,
                    out_path=fig_path,
                    title=title,
                    top_n=top_n,
                    log_scale=log_scale,
                    max_topics=max_topics,
                    max_words=max_words,
                )

                print(f"[OK] saved: {fig_path}")


# ----------------------------- CLI ----------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base path that contains Distributions-Results/, e.g., /home/tal/dev/TopicDistributions")
    parser.add_argument("--run_num", type=str, default="1",
                        help="Run number directory under Distributions-Results (default: 1)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output base dir (default: <base_dir>)")
    parser.add_argument("--top_n", type=int, default=12,
                        help="Top-N words per topic used to build union word set (default: 12)")
    parser.add_argument("--log_scale", action="store_true",
                        help="Log-transform probabilities for better contrast")
    parser.add_argument("--max_topics", type=int, default=None,
                        help="Optional: limit number of topic columns in the plot")
    parser.add_argument("--max_words", type=int, default=80,
                        help="Optional: cap number of word rows selected after union (default: 80)")
    parser.add_argument("--dataset", type=str, default=None,
                    help="If set, process only this dataset directory name (e.g., 20NewsGroup). Default: all.")
    parser.add_argument("--num_topics", type=str, default=None,
                    help="If set, process only this num_topics subdir (e.g., 20). Default: all.")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base_dir

    generate_all_heatmaps(
        base_dir=base_dir,
        run_num=args.run_num,
        out_dir=out_dir,
        top_n=args.top_n,
        log_scale=args.log_scale,
        max_topics=args.max_topics,
        max_words=args.max_words,
        dataset_filter=args.dataset,
        num_topics_filter=args.num_topics,
)



if __name__ == "__main__":
    main()


"""
python generate_heatmaps.py \
--base_dir /home/tal/dev/TopicDistributions \
--run_num FINAL \
--top_n 10 \
--log_scale \
--max_words 50 \
--dataset BBC \
--num_topics 5

"""