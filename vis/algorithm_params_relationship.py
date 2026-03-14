import argparse
import sys
import csv
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ALGO_REGEX_TO_LABEL: List[Tuple[str, str]] = [
    (r"\bk[\s-]?means\b", "k-means"),
    (r"\bmini[\s-]?batch[\s-]?k[\s-]?means\b", "mini-batch k-means"),
    (r"\bk[\s-]?medoids\b|\bpam\b", "k-medoids/PAM"),
    (r"\bfuzzy\s*c[\s-]?means\b|\bc[\s-]?means\b", "c-means/fuzzy c-means"),
    (r"\bhierarchical\s+clustering\b|\bagglomerative\s+clustering\b|\bdivisive\s+clustering\b", "hierarchical"),
    (r"\bward(?:'s)?\b\s+method\b", "ward"),
    (r"\bdbscan\b", "DBSCAN"),
    (r"\bhdbscan\b", "HDBSCAN"),
    (r"\boptics\b", "OPTICS"),
    (r"\bmean\s*shift\b", "mean shift"),
    (r"\bbirch\b", "BIRCH"),
    (r"\bspectral\s+clustering\b", "spectral"),
    (r"\baffinity\s+propagation\b", "affinity propagation"),
    (r"\bgaussian\s+mixture(?:\s+model)?\b|\bgmm\b|\bmixture\s+model\b", "GMM/mixture"),
    (r"\bdirichlet\s+process\s+mixture\b|\bdpm\b", "DPM"),
    (r"\bself[-\s]?organizing\s+map\b|\bsom\b", "SOM"),
    (r"\bneural\s+gas\b", "neural gas"),
    (r"\bmarkov\s+clustering\b|\bmcl\b", "MCL"),
    (r"\bgraph[-\s]?based\s+clustering\b", "graph-based"),
    (r"\bcommunity\s+detection\b", "community detection"),
    (r"\blouvain\b", "louvain"),
    (r"\bleiden\b", "leiden"),
    (r"\bshared\s+nearest\s+neighbor\b|\bsnn\b", "SNN"),
    (r"\bphenograph\b", "Phenograph"),
    (r"\bseurat\b", "Seurat"),
    (r"\bscanpy\b", "Scanpy"),
    (r"\bsc3\b", "SC3"),
    (r"\bcluster(?:ing|ed|s)?\b", "generic cluster*"),
]

LABEL_ORDER = [lab for _, lab in ALGO_REGEX_TO_LABEL]

def compile_labelers() -> List[Tuple[re.Pattern, str]]:
    out = []
    for pat, lab in ALGO_REGEX_TO_LABEL:
        out.append((re.compile(pat, flags=re.IGNORECASE), lab))
    return out

def split_hits(cell: str) -> List[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip()]

def labels_from_algorithms_found(hits: List[str], labelers: List[Tuple[re.Pattern, str]]) -> List[str]:
    text = " ; ".join(hits)
    labs = []
    for rx, lab in labelers:
        if rx.search(text):
            labs.append(lab)
    return labs

def bh_fdr(pvals: List[float]) -> List[float]:
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        j = order[-rank]
        pv = pvals[j]
        k = n - rank + 1
        adj = pv * n / k
        if adj > prev:
            adj = prev
        prev = adj
        q[j] = min(1.0, adj)
    return q

def two_prop_ztest(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
    n1 = a + b
    n2 = c + d
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p_pool = (a + c) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = ((a / n1) - (c / n2)) / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))
    return z, p

def safe_int(x, default=0) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default

def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return list(r)

def find_year_from_name(p: Path) -> str:
    m = re.search(r"(19\d{2}|20\d{2})", p.name)
    return m.group(1) if m else ""

def summarize_by_algorithm(rows: List[Dict[str, str]], labelers: List[Tuple[re.Pattern, str]], year: str) -> Dict[str, Dict[str, float]]:
    stats = {}
    counts = defaultdict(int)
    miss_params = defaultdict(int)
    miss_just = defaultdict(int)
    miss_eval = defaultdict(int)
    miss_tune = defaultdict(int)
    miss_any = defaultdict(int)
    parse_err = defaultdict(int)

    for row in rows:
        if row.get("error"):
            continue
        hits = split_hits(row.get("algorithms_found", ""))
        labs = labels_from_algorithms_found(hits, labelers)
        if not labs:
            continue
        mp = safe_int(row.get("missing_params", "0"))
        mj = safe_int(row.get("missing_justification", "0"))
        me = safe_int(row.get("missing_evaluation", "0"))
        mt = safe_int(row.get("missing_tuning", "0"))
        ma = safe_int(row.get("missing_reporting_signals", "0"))
        pe = 1 if row.get("error") else 0
        for lab in labs:
            counts[lab] += 1
            miss_params[lab] += mp
            miss_just[lab] += mj
            miss_eval[lab] += me
            miss_tune[lab] += mt
            miss_any[lab] += ma
            parse_err[lab] += pe

    for lab in LABEL_ORDER:
        n = counts.get(lab, 0)
        if n == 0:
            continue
        stats[lab] = {
            "year": year,
            "algorithm": lab,
            "n_articles": n,
            "pct_missing_params": 100.0 * miss_params[lab] / n,
            "pct_missing_justification": 100.0 * miss_just[lab] / n,
            "pct_missing_evaluation": 100.0 * miss_eval[lab] / n,
            "pct_missing_tuning": 100.0 * miss_tune[lab] / n,
            "pct_missing_reporting_signals": 100.0 * miss_any[lab] / n,
        }
    return stats

def overall_baseline(rows: List[Dict[str, str]]) -> Dict[str, float]:
    n = 0
    mp = mj = me = mt = ma = 0
    for row in rows:
        if row.get("error"):
            continue
        algos = (row.get("algorithms_found") or "").strip()
        if not algos:
            continue
        n += 1
        mp += safe_int(row.get("missing_params", "0"))
        mj += safe_int(row.get("missing_justification", "0"))
        me += safe_int(row.get("missing_evaluation", "0"))
        mt += safe_int(row.get("missing_tuning", "0"))
        ma += safe_int(row.get("missing_reporting_signals", "0"))
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "pct_missing_params": 100.0 * mp / n,
        "pct_missing_justification": 100.0 * mj / n,
        "pct_missing_evaluation": 100.0 * me / n,
        "pct_missing_tuning": 100.0 * mt / n,
        "pct_missing_reporting_signals": 100.0 * ma / n,
    }

def write_tsv(out_path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--results-dir", required=True)
    ap.add_argument("-o", "--out-prefix", default="algo_effect")
    ap.add_argument("--min-n", type=int, default=50)
    args = ap.parse_args()

    labelers = compile_labelers()

    tsvs = sorted(Path(args.results_dir).glob("results_*.tsv"))
    if not tsvs:
        raise SystemExit(f"No results_*.tsv found under {args.results_dir}")

    all_rows = []
    per_year = []
    for tsv in tsvs:
        rows = read_tsv(tsv)
        year = find_year_from_name(tsv)
        per_algo = summarize_by_algorithm(rows, labelers, year=year)
        for lab, d in per_algo.items():
            if d["n_articles"] >= args.min_n:
                per_year.append(d)
        all_rows.extend(rows)

    baseline = overall_baseline(all_rows)

    summary_rows = []
    for d in per_year:
        summary_rows.append(d)

    out1 = Path(f"{args.out_prefix}_by_year.tsv")
    write_tsv(
        out1,
        summary_rows,
        ["year", "algorithm", "n_articles", "pct_missing_params", "pct_missing_justification", "pct_missing_evaluation", "pct_missing_tuning", "pct_missing_reporting_signals"],
    )

    agg_counts = defaultdict(int)
    agg_mp = defaultdict(int)
    agg_mj = defaultdict(int)
    agg_me = defaultdict(int)
    agg_mt = defaultdict(int)
    agg_ma = defaultdict(int)

    labelers = compile_labelers()
    for row in all_rows:
        if row.get("error"):
            continue
        hits = split_hits(row.get("algorithms_found", ""))
        labs = labels_from_algorithms_found(hits, labelers)
        if not labs:
            continue
        mp = safe_int(row.get("missing_params", "0"))
        mj = safe_int(row.get("missing_justification", "0"))
        me = safe_int(row.get("missing_evaluation", "0"))
        mt = safe_int(row.get("missing_tuning", "0"))
        ma = safe_int(row.get("missing_reporting_signals", "0"))
        for lab in labs:
            agg_counts[lab] += 1
            agg_mp[lab] += mp
            agg_mj[lab] += mj
            agg_me[lab] += me
            agg_mt[lab] += mt
            agg_ma[lab] += ma

    agg_rows = []
    pvals = []
    labels = []
    tests = []
    for lab in LABEL_ORDER:
        n = agg_counts.get(lab, 0)
        if n < args.min_n:
            continue
        a = agg_ma[lab]
        b = n - a
        c = int(round(baseline["n"] * baseline["pct_missing_reporting_signals"] / 100.0))
        d = baseline["n"] - c
        z, p = two_prop_ztest(a, b, c, d)
        labels.append(lab)
        pvals.append(p if not (p is None or math.isnan(p)) else 1.0)
        tests.append((z, p))
        agg_rows.append({
            "algorithm": lab,
            "n_articles": n,
            "pct_missing_params": 100.0 * agg_mp[lab] / n,
            "pct_missing_justification": 100.0 * agg_mj[lab] / n,
            "pct_missing_evaluation": 100.0 * agg_me[lab] / n,
            "pct_missing_tuning": 100.0 * agg_mt[lab] / n,
            "pct_missing_reporting_signals": 100.0 * agg_ma[lab] / n,
            "delta_missing_reporting_vs_overall_pp": (100.0 * agg_ma[lab] / n) - baseline["pct_missing_reporting_signals"],
            "z_vs_overall": z,
            "p_vs_overall": p,
        })

    qvals = bh_fdr(pvals) if pvals else []
    for i in range(len(agg_rows)):
        agg_rows[i]["q_fdr_bh"] = qvals[i] if qvals else ""

    out2 = Path(f"{args.out_prefix}_overall.tsv")
    write_tsv(
        out2,
        agg_rows,
        ["algorithm", "n_articles",
         "pct_missing_params", "pct_missing_justification", "pct_missing_evaluation", "pct_missing_tuning", "pct_missing_reporting_signals",
         "delta_missing_reporting_vs_overall_pp", "z_vs_overall", "p_vs_overall", "q_fdr_bh"],
    )

if __name__ == "__main__":
    main()
