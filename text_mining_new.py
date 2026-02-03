#!/usr/bin/env python3
#python text_mining_new.py -i articles -o ./new_output/ --plots --recursive
import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import xml.etree.ElementTree as ET

from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


csv.field_size_limit(sys.maxsize)

WS_RE = re.compile(r"\s+")


def norm_ws(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip())


def safe_join_itertext(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return norm_ws("".join(el.itertext()))


def detect_ns(root: ET.Element) -> Optional[Dict[str, str]]:
    # JATS tags may be namespaced as "{uri}tag". If so, pull uri.
    tag = root.tag
    if tag.startswith("{") and "}" in tag:
        uri = tag.split("}", 1)[0][1:]
        return {"ns": uri}
    return None


def find_first(root: ET.Element, path_no_ns: str, path_ns: str, ns: Optional[Dict[str, str]]) -> Optional[ET.Element]:
    if ns:
        el = root.find(path_ns, ns)
        if el is not None:
            return el
    return root.find(path_no_ns)


def find_all(root: ET.Element, path_no_ns: str, path_ns: str, ns: Optional[Dict[str, str]]) -> List[ET.Element]:
    if ns:
        els = root.findall(path_ns, ns)
        if els:
            return els
    return root.findall(path_no_ns)


def extract_full_text_jats(xml_path: Path) -> Tuple[str, str, str]:
    """
    Returns: (title, abstract, body_text)
    Robust to namespace/no-namespace and inline tags.
    """
    root = ET.parse(str(xml_path)).getroot()
    ns = detect_ns(root)

    title_el = find_first(root, ".//article-title", ".//ns:article-title", ns)
    title = safe_join_itertext(title_el)

    abs_els = find_all(root, ".//abstract", ".//ns:abstract", ns)
    abstracts = [safe_join_itertext(a) for a in abs_els]
    abstracts = [a for a in abstracts if a]
    abstract = norm_ws("\n\n".join(abstracts))

    body_el = find_first(root, ".//body", ".//ns:body", ns)
    body_text = safe_join_itertext(body_el)

    return title, abstract, body_text


@dataclass(frozen=True)
class PatternBank:
    algorithm: List[re.Pattern]
    params: List[re.Pattern]
    justification: List[re.Pattern]
    evaluation: List[re.Pattern]
    tuning: List[re.Pattern]


def compile_patterns(patterns: Dict[str, List[str]], flags: int = re.IGNORECASE) -> PatternBank:
    def _c(key: str) -> List[re.Pattern]:
        return [re.compile(p, flags) for p in patterns.get(key, [])]

    return PatternBank(
        algorithm=_c("algorithm"),
        params=_c("params"),
        justification=_c("justification"),
        evaluation=_c("evaluation"),
        tuning=_c("tuning"),
    )


DEFAULT_PATTERNS: Dict[str, List[str]] = {
    "algorithm": [
        # Core clustering families
        r"\bk[\s-]?means\b",
        r"\bmini[\s-]?batch[\s-]?k[\s-]?means\b",
        r"\bk[\s-]?medoids\b|\bpam\b",
        r"\bfuzzy\s*c[\s-]?means\b|\bc[\s-]?means\b",
        r"\bhierarchical\s+clustering\b|\bagglomerative\s+clustering\b|\bdivisive\s+clustering\b",
        r"\bward(?:'s)?\b\s+method\b",
        r"\bdbscan\b",
        r"\bhdbscan\b",
        r"\boptics\b",
        r"\bmean\s*shift\b",
        r"\bbirch\b",
        r"\bspectral\s+clustering\b",
        r"\baffinity\s+propagation\b",
        r"\bgaussian\s+mixture(?:\s+model)?\b|\bgmm\b|\bmixture\s+model\b",
        r"\bdirichlet\s+process\s+mixture\b|\bdpm\b",
        r"\bself[-\s]?organizing\s+map\b|\bsom\b",
        r"\bneural\s+gas\b",
        r"\bmarkov\s+clustering\b|\bmcl\b",

        # Graph/community clustering (very common in modern bio)
        r"\bgraph[-\s]?based\s+clustering\b",
        r"\bcommunity\s+detection\b",
        r"\blouvain\b",
        r"\bleiden\b",
        r"\bshared\s+nearest\s+neighbor\b|\bsnn\b",

        # Domain/tool proxies (optional but improves recall in omics-heavy PMC)
        r"\bphenograph\b",
        r"\bseurat\b",
        r"\bscanpy\b",
        r"\bsc3\b",

        # Generic mention (keep last; useful but broad)
        r"\bcluster(?:ing|ed|s)?\b",
    ],

    "params": [
        # k-means / partitioning
        r"\bn[_\s-]?clusters\b",
        r"\bnumber\s+of\s+clusters\b",
        r"\b(?:k|K)\s*=\s*\d+\b",
        r"\binit\b\s*=\s*(?:k[-\s]?means\+\+|random)\b",
        r"\bn[_\s-]?init\b\s*=\s*\d+\b",
        r"\bmax[_\s-]?iter(?:ations)?\b\s*=\s*\d+\b",
        r"\btol(?:erance)?\b\s*=\s*[\deE\.\-]+\b",

        # Density-based
        r"\beps\b\s*=\s*[\deE\.\-]+\b",
        r"\bmin[_\s-]?samples\b\s*=\s*\d+\b",
        r"\bmin[_\s-]?cluster[_\s-]?size\b\s*=\s*\d+\b",
        r"\bxi\b\s*=\s*[\deE\.\-]+\b",  # OPTICS
        r"\bcluster[_\s-]?method\b\s*=\s*\w+\b",  # OPTICS/HDBSCAN variants

        # Hierarchical
        r"\blinkage\b\s*=\s*(?:ward|complete|average|single)\b",
        r"\b(?:distance|affinity)\s+metric\b",
        r"\bmetric\b\s*=\s*(?:euclidean|cosine|manhattan|cityblock|chebyshev|minkowski|correlation)\b",
        r"\bward\b\s+linkage\b",
        r"\bcut(?:ting)?\s+the\s+dendrogram\b|\bcut\s+height\b",

        # GMM / mixture
        r"\bn[_\s-]?components\b\s*=\s*\d+\b",
        r"\bcovariance[_\s-]?type\b\s*=\s*(?:full|tied|diag|spherical)\b",
        r"\bregulari[sz]ation\b\s*=\s*[\deE\.\-]+\b",

        # Spectral
        r"\bn[_\s-]?neighbors\b\s*=\s*\d+\b",
        r"\bgamma\b\s*=\s*[\deE\.\-]+\b",

        # Affinity propagation / mean shift
        r"\bdamping\b\s*=\s*[\deE\.\-]+\b",
        r"\bpreference\b\s*=\s*[\deE\.\-]+\b",
        r"\bbandwidth\b\s*=\s*[\deE\.\-]+\b",

        # Graph/community detection
        r"\bresolution\b\s*=\s*[\deE\.\-]+\b",
        r"\bmodularity\b",
        r"\bkNN\b|\bk\s*nearest\s+neighbors\b",
        r"\bneighbors\b\s*=\s*\d+\b",

        # Keep embeddings separate from clustering if you care about purity of "params";
        # included because many papers report clustering in embedding space.
        r"\bUMAP\b.*\b(n[_\s-]?neighbors|min[_\s-]?dist)\b",
        r"\bt-?SNE\b.*\b(perplexity|learning\s*rate)\b",
    ],

    "justification": [
        # Selecting number of clusters / resolution / thresholds
        r"\belbow\b|\bknee\b",
        r"\bgap\s+statistic\b",
        r"\bsilhouette(?:\s+analysis|\s+score)?\b",
        r"\b(?:chosen|selected|determined|set)\s+(?:based\s+on|according\s+to)\b",
        r"\bwe\s+(?:chose|selected|set|determined)\b",
        r"\bwas\s+(?:chosen|selected|determined|set)\b",
        r"\bto\s+determine\s+(?:the\s+)?number\s+of\s+clusters\b",
        r"\boptimal\s+(?:k|number\s+of\s+clusters|resolution)\b",
        r"\bempiric(?:al|ally)\b",
        r"\bconsistent\s+with\s+(?:previous|prior)\s+(?:work|studies)\b",
        r"\bfollowing\s+(?:previous|prior)\s+(?:work|studies|protocol)\b",
        r"\bas\s+described\s+(?:previously|elsewhere)\b",
    ],

    "evaluation": [
        # Internal validation
        r"\bsilhouette(?:\s+score|\s+coefficient)?\b",
        r"\bdavies[-\s]?bouldin\b",
        r"\bcalinski[-\s]?harabasz\b",
        r"\bwithin[-\s]?cluster\s+sum\s+of\s+squares\b|\bWCSS\b",
        r"\bbetween[-\s]?cluster\s+variance\b",
        r"\bcluster\s+stability\b|\bstability\s+analysis\b",
        r"\bbootstrap(?:ping)?\b.*\bcluster\b",

        # External validation / agreement
        r"\badjusted\s+rand\b|\bARI\b",
        r"\bnormalized\s+mutual\s+information\b|\bNMI\b",
        r"\badjusted\s+mutual\s+information\b|\bAMI\b",
        r"\bfowlkes[-\s]?mallows\b|\bFMI\b",
        r"\bjaccard\b",
        r"\bpurity\b",

        # Model-based clustering diagnostics
        r"\bBIC\b|\bAIC\b",
        r"\blog[-\s]?likelihood\b",

        # Graph clustering diagnostics
        r"\bmodularity\b",
        r"\bconductance\b",
    ],

    "tuning": [
        # Explicit tuning language
        r"\bhyper[-\s]?parameter\s+(?:tuning|optimization|optimisation)\b",
        r"\bparameter\s+sweep\b",
        r"\bgrid\s+search\b",
        r"\brandom(?:ized)?\s+search\b",
        r"\bbayesian\s+optimization\b|\bbayesian\s+optimisation\b",
        r"\boptuna\b|\bhyperopt\b|\bskopt\b",
        r"\bcross[-\s]?validation\b|\bCV\b",
        r"\bnested\s+cross[-\s]?validation\b",
        r"\b(?:tuned|optimized|optimised|selected)\s+using\b",
        r"\bwe\s+(?:tuned|optimized|optimised)\b",
    ],
}

def load_patterns_json(path: Path) -> Dict[str, List[str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("patterns JSON must be an object/dict of lists")
    out: Dict[str, List[str]] = {}
    for k, v in obj.items():
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValueError(f"patterns[{k}] must be a list of strings")
        out[k] = v
    return out


def any_match(text: str, pats: List[re.Pattern]) -> bool:
    return any(p.search(text) is not None for p in pats)


def all_matches(text: str, pats: List[re.Pattern], limit_per_pattern: int = 20, limit_total: int = 200) -> List[str]:
    hits: List[str] = []
    for p in pats:
        c = 0
        for m in p.finditer(text):
            hits.append(m.group(0))
            c += 1
            if c >= limit_per_pattern or len(hits) >= limit_total:
                break
        if len(hits) >= limit_total:
            break
    # dedupe preserve order
    seen = set()
    out = []
    for h in hits:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            out.append(h)
        if len(out) >= limit_total:
            break
    return out



def analyze_text(text: str, bank: PatternBank) -> Dict[str, object]:
    algo_found = all_matches(text, bank.algorithm)
    params_found = all_matches(text, bank.params)
    justification_found = any_match(text, bank.justification)
    evaluation_found = any_match(text, bank.evaluation)
    tuning_found = any_match(text, bank.tuning)

    # Interpret this as "missing explicit reporting signals" (not definitive misuse).
    missing_params = len(params_found) == 0
    missing_just = not justification_found
    missing_eval = not evaluation_found
    missing_tune = not tuning_found
    missing_reporting_signals = missing_params or missing_just or missing_eval or missing_tune

    return {
        "algorithms_found": ";".join(algo_found),
        "params_found": ";".join(params_found),
        "justification_found": int(justification_found),
        "evaluation_found": int(evaluation_found),
        "tuning_found": int(tuning_found),
        "missing_params": int(missing_params),
        "missing_justification": int(missing_just),
        "missing_evaluation": int(missing_eval),
        "missing_tuning": int(missing_tune),
        "missing_reporting_signals": int(missing_reporting_signals),
    }


def iter_year_dirs(base: Path, recursive: bool = False) -> List[Tuple[str, Path]]:
    """
    Collects year-like subdirs under base. Year is any 4-digit 19xx/20xx in folder name.
    Returns list of (year, path) sorted by year.
    """
    year_pat = re.compile(r"(19\d{2}|20\d{2})")
    candidates: List[Tuple[int, str, Path]] = []

    if recursive:
        dirs = [p for p in base.rglob("*") if p.is_dir()]
    else:
        dirs = [p for p in base.iterdir() if p.is_dir()]

    for d in dirs:
        m = year_pat.search(d.name)
        if not m:
            continue
        year = int(m.group(1))
        candidates.append((year, m.group(1), d))

    candidates.sort(key=lambda x: (x[0], str(x[2])))
    # de-dupe: if multiple dirs share same year, keep all, but stable
    return [(ystr, d) for _, ystr, d in candidates]


def write_tsv(rows: Iterable[Dict[str, object]], out_path: Path, fieldnames: List[str]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
            n += 1
    return n


def summarize_tsv(tsv_path: Path) -> Dict[str, object]:
    """
    Lightweight summary without pandas.
    """
    total = 0
    with_algo = 0
    missing_reporting = 0
    missing_params = 0
    missing_just = 0
    missing_eval = 0
    missing_tune = 0
    parse_errors = 0

    with tsv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            total += 1
            if row.get("error"):
                parse_errors += 1
            if (row.get("algorithms_found") or "").strip():
                with_algo += 1
            if row.get("missing_reporting_signals") == "1":
                missing_reporting += 1
            if row.get("missing_params") == "1":
                missing_params += 1
            if row.get("missing_justification") == "1":
                missing_just += 1
            if row.get("missing_evaluation") == "1":
                missing_eval += 1
            if row.get("missing_tuning") == "1":
                missing_tune += 1

    def pct(x: int, denom: int) -> float:
        return (x / denom * 100.0) if denom else 0.0

    return {
        "total_articles": total,
        "articles_with_any_algorithm_match": with_algo,
        "pct_with_any_algorithm_match": pct(with_algo, total),
        "articles_with_missing_reporting_signals": missing_reporting,
        "pct_missing_reporting_signals_among_all": pct(missing_reporting, total),
        "missing_params": missing_params,
        "missing_justification": missing_just,
        "missing_evaluation": missing_eval,
        "missing_tuning": missing_tune,
        "parse_errors": parse_errors,
    }


def plot_yearly_summary(summary_by_year: Dict[str, Dict[str, object]], out_dir: Path) -> None:
    if plt is None:
        raise RuntimeError("matplotlib not available; install matplotlib or run without --plots")

    years = sorted(summary_by_year.keys(), key=lambda y: int(y))
    total = [summary_by_year[y]["total_articles"] for y in years]
    pct_algo = [summary_by_year[y]["pct_with_any_algorithm_match"] for y in years]
    pct_missing = [summary_by_year[y]["pct_missing_reporting_signals_among_all"] for y in years]

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(years, total, marker="o")
    plt.xticks(rotation=45)
    plt.title("Total articles processed per year")
    plt.tight_layout()
    plt.savefig(out_dir / "total_articles_per_year.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(years, pct_algo, marker="o")
    plt.xticks(rotation=45)
    plt.title("Percent with any clustering/algorithm signal per year")
    plt.ylabel("Percent")
    plt.tight_layout()
    plt.savefig(out_dir / "pct_algorithm_signal_per_year.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(years, pct_missing, marker="o")
    plt.xticks(rotation=45)
    plt.title("Percent with missing reporting signals per year")
    plt.ylabel("Percent")
    plt.tight_layout()
    plt.savefig(out_dir / "pct_missing_reporting_signals_per_year.png", dpi=200)
    plt.close()


def process_year_dir(
    year: str,
    year_dir: Path,
    out_dir: Path,
    bank: PatternBank,
) -> Tuple[Path, Dict[str, object]]:
    xml_files = sorted(year_dir.glob("*.xml"))
    if not xml_files:
        # Still write an empty TSV with header for consistency
        out_tsv = out_dir / f"results_{year}.tsv"
        fields = [
            "pmc_id",
            "title",
            "abstract",
            "algorithms_found",
            "params_found",
            "justification_found",
            "evaluation_found",
            "tuning_found",
            "missing_params",
            "missing_justification",
            "missing_evaluation",
            "missing_tuning",
            "missing_reporting_signals",
            "error",
            "source_file",
        ]
        write_tsv([], out_tsv, fields)
        summary = {
            "year": year,
            "dir": str(year_dir),
            "total_articles": 0,
            "note": "no .xml files found",
        }
        return out_tsv, summary

    out_tsv = out_dir / f"results_{year}.tsv"
    fields = [
        "pmc_id",
        "title",
        "abstract",
        "algorithms_found",
        "params_found",
        "justification_found",
        "evaluation_found",
        "tuning_found",
        "missing_params",
        "missing_justification",
        "missing_evaluation",
        "missing_tuning",
        "missing_reporting_signals",
        "error",
        "source_file",
    ]

    def row_iter() -> Iterable[Dict[str, object]]:
        for xml_path in tqdm(xml_files, desc=f"Year {year} ({year_dir.name})"):
            pmc_id = xml_path.stem
            try:
                title, abstract, body_text = extract_full_text_jats(xml_path)
                analysis = analyze_text(body_text, bank)
                yield {
                    "pmc_id": pmc_id,
                    "title": title,
                    "abstract": abstract,
                    **analysis,
                    "error": "",
                    "source_file": str(xml_path),
                }
            except Exception as e:
                # Always emit a row to avoid silent drop
                yield {
                    "pmc_id": pmc_id,
                    "title": "",
                    "abstract": "",
                    "algorithms_found": "",
                    "params_found": "",
                    "justification_found": 0,
                    "evaluation_found": 0,
                    "tuning_found": 0,
                    "missing_params": 1,
                    "missing_justification": 1,
                    "missing_evaluation": 1,
                    "missing_tuning": 1,
                    "missing_reporting_signals": 1,
                    "error": str(e),
                    "source_file": str(xml_path),
                }

    n = write_tsv(row_iter(), out_tsv, fields)
    summary = summarize_tsv(out_tsv)
    summary["year"] = year
    summary["dir"] = str(year_dir)
    summary["results_tsv"] = str(out_tsv)
    summary["rows_written"] = n
    return out_tsv, summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="PMC/JATS XML text-mining per-year: extracts text, mines clustering reporting signals, writes per-year TSV + summary JSON."
    )
    ap.add_argument("-i", "--input-dir", required=True, help="Input directory containing year subfolders (e.g., pmc2000, pmc_articles_2001, etc.)")
    ap.add_argument("-o", "--output-dir", required=True, help="Output directory for per-year results + summaries")
    ap.add_argument("--patterns-json", default=None, help="Optional JSON file with regex lists: keys = algorithm/params/justification/evaluation/tuning")
    ap.add_argument("--recursive", action="store_true", help="Search for year folders recursively under input-dir")
    ap.add_argument("--plots", action="store_true", help="Generate simple yearly plots (requires matplotlib)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[ERROR] input-dir not found or not a directory: {in_dir}", file=sys.stderr)
        return 2

    patterns = DEFAULT_PATTERNS
    if args.patterns_json:
        p = Path(args.patterns_json).resolve()
        if not p.exists():
            print(f"[ERROR] patterns-json not found: {p}", file=sys.stderr)
            return 2
        patterns = load_patterns_json(p)

    bank = compile_patterns(patterns)

    year_dirs = iter_year_dirs(in_dir, recursive=args.recursive)
    if not year_dirs:
        print(f"[ERROR] No year-like subfolders found under: {in_dir}", file=sys.stderr)
        print("        Folder name must include a 4-digit year like 2000..2099.", file=sys.stderr)
        return 3

    out_dir.mkdir(parents=True, exist_ok=True)

    summary_by_year: Dict[str, Dict[str, object]] = {}
    for year, ydir in year_dirs:
        _, summ = process_year_dir(year, ydir, out_dir, bank)
        summary_by_year[year] = summ
        # per-year summary JSON
        (out_dir / f"summary_{year}.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")

    (out_dir / "summary_all_years.json").write_text(json.dumps(summary_by_year, indent=2), encoding="utf-8")

    if args.plots:
        plot_yearly_summary(summary_by_year, out_dir)

    print(f"[OK] Wrote per-year TSV + summaries to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
