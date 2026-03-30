# total_articles per year, where they mentioned clustering
# of the total algorithms per year, which of them has articles_with_any_algorithm_match
# then for the same year: pct_with_any_algorithm_match
# articles_with_missing_reporting_signals --> pct_missing_reporting_signals_among_all
# missing_params
# missing_justification
# missing_evaluation
# missing_tuning

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

DEFAULT_PATTERNS: Dict[str, List[str]] = {
    "algorithm": [
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
        r"\bgraph[-\s]?based\s+clustering\b",
        r"\bcommunity\s+detection\b",
        r"\blouvain\b",
        r"\bleiden\b",
        r"\bshared\s+nearest\s+neighbor\b|\bsnn\b",
        r"\bphenograph\b",
        r"\bseurat\b",
        r"\bscanpy\b",
        r"\bsc3\b",
        r"\bcluster(?:ing|ed|s)?\b",
    ],
    "params": [
        r"\bn[_\s-]?clusters\b",
        r"\bnumber\s+of\s+clusters\b",
        r"\b(?:k|K)\s*=\s*\d+\b",
        r"\binit\b\s*=\s*(?:k[-\s]?means\+\+|random)\b",
        r"\bn[_\s-]?init\b\s*=\s*\d+\b",
        r"\bmax[_\s-]?iter(?:ations)?\b\s*=\s*\d+\b",
        r"\btol(?:erance)?\b\s*=\s*[\deE\.\-]+\b",
        r"\beps\b\s*=\s*[\deE\.\-]+\b",
        r"\bmin[_\s-]?samples\b\s*=\s*\d+\b",
        r"\bmin[_\s-]?cluster[_\s-]?size\b\s*=\s*\d+\b",
        r"\bxi\b\s*=\s*[\deE\.\-]+\b",
        r"\bcluster[_\s-]?method\b\s*=\s*\w+\b",
        r"\blinkage\b\s*=\s*(?:ward|complete|average|single)\b",
        r"\b(?:distance|affinity)\s+metric\b",
        r"\bmetric\b\s*=\s*(?:euclidean|cosine|manhattan|cityblock|chebyshev|minkowski|correlation)\b",
        r"\bward\b\s+linkage\b",
        r"\bcut(?:ting)?\s+the\s+dendrogram\b|\bcut\s+height\b",
        r"\bn[_\s-]?components\b\s*=\s*\d+\b",
        r"\bcovariance[_\s-]?type\b\s*=\s*(?:full|tied|diag|spherical)\b",
        r"\bregulari[sz]ation\b\s*=\s*[\deE\.\-]+\b",
        r"\bn[_\s-]?neighbors\b\s*=\s*\d+\b",
        r"\bgamma\b\s*=\s*[\deE\.\-]+\b",
        r"\bdamping\b\s*=\s*[\deE\.\-]+\b",
        r"\bpreference\b\s*=\s*[\deE\.\-]+\b",
        r"\bbandwidth\b\s*=\s*[\deE\.\-]+\b",
        r"\bresolution\b\s*=\s*[\deE\.\-]+\b",
        r"\bmodularity\b",
        r"\bkNN\b|\bk\s*nearest\s+neighbors\b",
        r"\bneighbors\b\s*=\s*\d+\b",
        r"\bUMAP\b.*\b(n[_\s-]?neighbors|min[_\s-]?dist)\b",
        r"\bt-?SNE\b.*\b(perplexity|learning\s*rate)\b",
    ],
    "justification": [
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
        r"\bsilhouette(?:\s+score|\s+coefficient)?\b",
        r"\bdavies[-\s]?bouldin\b",
        r"\bcalinski[-\s]?harabasz\b",
        r"\bwithin[-\s]?cluster\s+sum\s+of\s+squares\b|\bWCSS\b",
        r"\bbetween[-\s]?cluster\s+variance\b",
        r"\bcluster\s+stability\b|\bstability\s+analysis\b",
        r"\bbootstrap(?:ping)?\b.*\bcluster\b",
        r"\badjusted\s+rand\b|\bARI\b",
        r"\bnormalized\s+mutual\s+information\b|\bNMI\b",
        r"\badjusted\s+mutual\s+information\b|\bAMI\b",
        r"\bfowlkes[-\s]?mallows\b|\bFMI\b",
        r"\bjaccard\b",
        r"\bpurity\b",
        r"\bBIC\b|\bAIC\b",
        r"\blog[-\s]?likelihood\b",
        r"\bmodularity\b",
        r"\bconductance\b",
    ],
    "tuning": [
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

def load_df(json_path: Path) -> pd.DataFrame:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for k, v in data.items():
        r = dict(v)
        r["year"] = int(r.get("year", k))
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    for c in [
        "total_articles",
        "articles_with_any_algorithm_match",
        "articles_with_missing_reporting_signals",
        "missing_params",
        "missing_justification",
        "missing_evaluation",
        "missing_tuning",
        "parse_errors",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["pct_with_any_algorithm_match", "pct_missing_reporting_signals_among_all"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["articles_without_algorithm_match"] = df["total_articles"] - df["articles_with_any_algorithm_match"]
    df["pct_without_algorithm_match"] = 100.0 - df["pct_with_any_algorithm_match"]
    df["rate_missing_params"] = df["missing_params"] / df["total_articles"] * 100.0
    df["rate_missing_justification"] = df["missing_justification"] / df["total_articles"] * 100.0
    df["rate_missing_evaluation"] = df["missing_evaluation"] / df["total_articles"] * 100.0
    df["rate_missing_tuning"] = df["missing_tuning"] / df["total_articles"] * 100.0
    miss_sum = df[["missing_params", "missing_justification", "missing_evaluation", "missing_tuning"]].sum(axis=1).replace(0, np.nan)
    df["comp_missing_params"] = df["missing_params"] / miss_sum * 100.0
    df["comp_missing_justification"] = df["missing_justification"] / miss_sum * 100.0
    df["comp_missing_evaluation"] = df["missing_evaluation"] / miss_sum * 100.0
    df["comp_missing_tuning"] = df["missing_tuning"] / miss_sum * 100.0
    return df

def html_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;"))

def kpi_cards(df: pd.DataFrame) -> str:
    last = df.iloc[-1].to_dict()
    y0, y1 = int(df["year"].min()), int(df["year"].max())
    def fmt_int(x):
        return f"{int(x):,}" if pd.notna(x) else "NA"
    def fmt_pct(x):
        return f"{x:.2f}%" if pd.notna(x) else "NA"
    cards = [
        ("Coverage window", f"{y0}–{y1}"),
        (f"Total articles ({y1})", fmt_int(last.get("total_articles"))),
        (f"Any algorithm match ({y1})", fmt_pct(last.get("pct_with_any_algorithm_match"))),
        (f"Missing reporting signals ({y1})", fmt_pct(last.get("pct_missing_reporting_signals_among_all"))),
        (f"Parse errors ({y1})", fmt_int(last.get("parse_errors"))),
    ]
    html = []
    for title, value in cards:
        html.append(
            f"""
            <div class="card">
              <div class="card-title">{title}</div>
              <div class="card-value">{value}</div>
            </div>
            """
        )
    return "\n".join(html)

def make_main_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        subplot_titles=[
            "Algorithm-match counts",
            "Algorithm-match rate (%)",
            "Missing reporting signals among all (%)",
            "Missing reporting categories (rates, % of all articles)",
            "Missing reporting categories ",
            "Parse errors "
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    x = df["year"]

    fig.add_trace(
        go.Scatter(
            x=x, y=df["total_articles"], mode="lines+markers",
            name="Total articles",
            hovertemplate="Year=%{x}<br>Total=%{y:,}<extra></extra>"
        ),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df["articles_with_any_algorithm_match"], mode="lines+markers",
            name="Any algorithm match ",
            hovertemplate="Year=%{x}<br>Matched=%{y:,}<extra></extra>"
        ),
        row=1, col=1, secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=df["pct_with_any_algorithm_match"], mode="lines+markers",
            name="Any algorithm match (%)",
            hovertemplate="Year=%{x}<br>% Match=%{y:.2f}%<extra></extra>"
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=df["pct_missing_reporting_signals_among_all"], mode="lines+markers",
            name="Missing reporting signals (%)",
            hovertemplate="Year=%{x}<br>% Missing=%{y:.2f}%<extra></extra>"
        ),
        row=2, col=1
    )

    for name, col in [
        ("Missing params (%)", "rate_missing_params"),
        ("Missing justification (%)", "rate_missing_justification"),
        ("Missing evaluation (%)", "rate_missing_evaluation"),
        ("Missing tuning (%)", "rate_missing_tuning"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x, y=df[col], mode="lines",
                name=name,
                hovertemplate="Year=%{x}<br>" + name + "=%{y:.2f}%<extra></extra>"
            ),
            row=2, col=2
        )

    for name, col in [
        ("Missing params ", "missing_params"),
        ("Missing justification ", "missing_justification"),
        ("Missing evaluation ", "missing_evaluation"),
        ("Missing tuning ", "missing_tuning"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x, y=df[col], mode="lines",
                name=name,
                hovertemplate="Year=%{x}<br>" + name + "=%{y:,}<extra></extra>"
            ),
            row=3, col=1
        )

    if df["parse_errors"].fillna(0).eq(0).all():
        fig.add_trace(
            go.Scatter(
                x=x, y=df["parse_errors"].fillna(0),
                mode="lines",
                name="Parse errors",
                hovertemplate="Year=%{x}<br>Errors=%{y:,}<extra></extra>"
            ),
            row=3, col=2
        )
    else:
        fig.add_trace(
            go.Bar(
                x=x, y=df["parse_errors"],
                name="Parse errors",
                hovertemplate="Year=%{x}<br>Errors=%{y:,}<extra></extra>"
            ),
            row=3, col=2
        )

    fig.update_xaxes(rangeslider_visible=False, row=3, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=3, col=2)

    fig.update_yaxes(type="log", row=1, col=1, title_text="Total (log)")
    fig.update_yaxes(row=1, col=1, secondary_y=True, title_text="")
    fig.update_yaxes(row=1, col=2, title_text="Percent")
    fig.update_yaxes(row=2, col=1, title_text="Percent")
    fig.update_yaxes(row=2, col=2, title_text="Percent")
    fig.update_yaxes(row=3, col=1, title_text="Count")
    fig.update_yaxes(row=3, col=2, title_text="Count")

    fig.update_layout(
        template="plotly_white",
        height=1100,
        margin=dict(l=60, r=40, t=70, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        hovermode="x unified"
    )
    return fig

def make_composition_figure(df: pd.DataFrame) -> go.Figure:
    x = df["year"]
    fig = go.Figure()
    traces = [
        ("Missing params share (%)", "comp_missing_params"),
        ("Missing justification share (%)", "comp_missing_justification"),
        ("Missing evaluation share (%)", "comp_missing_evaluation"),
        ("Missing tuning share (%)", "comp_missing_tuning"),
    ]
    for name, col in traces:
        fig.add_trace(
            go.Scatter(
                x=x, y=df[col], mode="lines",
                stackgroup="one",
                name=name,
                hovertemplate="Year=%{x}<br>" + name + "=%{y:.2f}%<extra></extra>"
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=60, r=40, t=60, b=60),
        title="",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
    )
    fig.update_yaxes(range=[0, 100], title_text="Share (%)")
    return fig

def metric_definitions_table(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    y = int(last["year"])
    value_map = {
        "total_articles": f"{int(last['total_articles']):,}",
        "articles_with_any_algorithm_match": f"{int(last['articles_with_any_algorithm_match']):,}",
        "pct_with_any_algorithm_match": f"{last['pct_with_any_algorithm_match']:.2f}%",
        "articles_with_missing_reporting_signals": f"{int(last['articles_with_missing_reporting_signals']):,}",
        "pct_missing_reporting_signals_among_all": f"{last['pct_missing_reporting_signals_among_all']:.2f}%",
        "missing_params": f"{int(last['missing_params']):,}",
        "missing_justification": f"{int(last['missing_justification']):,}",
        "missing_evaluation": f"{int(last['missing_evaluation']):,}",
        "missing_tuning": f"{int(last['missing_tuning']):,}",
        "parse_errors": f"{int(last['parse_errors']):,}",
    }
    rows = [
        ("total_articles", value_map["total_articles"], "Count of processed XML files assigned to that year."),
        ("articles_with_any_algorithm_match", value_map["articles_with_any_algorithm_match"], "Articles with algorithms_found non-empty."),
        ("pct_with_any_algorithm_match", value_map["pct_with_any_algorithm_match"], "100 × articles_with_any_algorithm_match / total_articles."),
        ("articles_with_missing_reporting_signals", value_map["articles_with_missing_reporting_signals"], "Articles where missing_reporting_signals=1."),
        ("pct_missing_reporting_signals_among_all", value_map["pct_missing_reporting_signals_among_all"], "100 × articles_with_missing_reporting_signals / total_articles."),
        ("missing_params", value_map["missing_params"], "Articles where missing_params=1 (params_found empty)."),
        ("missing_justification", value_map["missing_justification"], "Articles where missing_justification=1 (justification_found=0)."),
        ("missing_evaluation", value_map["missing_evaluation"], "Articles where missing_evaluation=1 (evaluation_found=0)."),
        ("missing_tuning", value_map["missing_tuning"], "Articles where missing_tuning=1 (tuning_found=0)."),
        ("parse_errors", value_map["parse_errors"], "Rows with non-empty error field; still counted in totals."),
    ]
    trs = "\n".join([f"<tr><td><span class='mono'>{k}</span></td><td>{v}</td><td>{d}</td></tr>" for k, v, d in rows])
    return f"""
    <div class="panel">
      <div class="panel-title">Snapshot ({y}) + metric definitions</div>
      <table class="tbl">
        <thead><tr><th>Metric</th><th>Value</th><th>Definition</th></tr></thead>
        <tbody>{trs}</tbody>
      </table>
      <div class="note">All signals are regex-based extractions from normalized body text; “missing” means “no regex match in that category”.</div>
    </div>
    """

def regex_table(key: str) -> str:
    pats = DEFAULT_PATTERNS.get(key, [])
    if not pats:
        return "<div class='note'>No regex defined for this category.</div>"
    rows = "\n".join([f"<tr><td class='mono'>{i}</td><td class='mono'>{html_escape(p)}</td></tr>" for i, p in enumerate(pats, 1)])
    return f"""
    <table class="tbl">
      <thead><tr><th>#</th><th>Regex (verbatim)</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """

def field_meanings_interactive() -> str:
    blocks = []

    blocks.append(
        f"""
        <details class="details">
          <summary><span class="mono">algorithms_found</span><span class="muted">distinct hits from DEFAULT_PATTERNS["algorithm"]</span></summary>
          <div class="details-body">
            <div class="note">
              The pipeline searches the article body text against every regex in <span class="mono">DEFAULT_PATTERNS["algorithm"]</span> with case-insensitive matching.
              Each regex contributes all non-overlapping matches it finds. Matched substrings are de-duplicated per article (case-folded) and emitted as a semicolon-joined list.
              If the list is non-empty, the article contributes to <span class="mono">articles_with_any_algorithm_match</span> in the yearly summary.
            </div>
            {regex_table("algorithm")}
          </div>
        </details>
        """
    )

    blocks.append(
        f"""
        <details class="details">
          <summary><span class="mono">params_found</span><span class="muted">distinct hits from DEFAULT_PATTERNS["params"]</span></summary>
          <div class="details-body">
            <div class="note">
              Same matching logic as <span class="mono">algorithms_found</span>, but using <span class="mono">DEFAULT_PATTERNS["params"]</span>.
              This bank targets explicit parameter mentions (k, n_clusters, eps, min_samples, resolution, linkage, etc.).
              If <span class="mono">params_found</span> is empty, then <span class="mono">missing_params=1</span>.
            </div>
            {regex_table("params")}
          </div>
        </details>
        """
    )

    blocks.append(
        f"""
        <details class="details">
          <summary><span class="mono">justification_found</span><span class="muted">binary: any hit in DEFAULT_PATTERNS["justification"]</span></summary>
          <div class="details-body">
            <div class="note">
              Set to <span class="mono">1</span> if at least one regex in <span class="mono">DEFAULT_PATTERNS["justification"]</span> matches the body text; otherwise <span class="mono">0</span>.
              This captures explicit rationale for choosing cluster count/resolution/thresholds (elbow/knee, gap statistic, “we chose… based on…”, etc.).
              If the value is <span class="mono">0</span>, then <span class="mono">missing_justification=1</span>.
            </div>
            {regex_table("justification")}
          </div>
        </details>
        """
    )

    blocks.append(
        f"""
        <details class="details">
          <summary><span class="mono">evaluation_found</span><span class="muted">binary: any hit in DEFAULT_PATTERNS["evaluation"]</span></summary>
          <div class="details-body">
            <div class="note">
              Set to <span class="mono">1</span> if at least one regex in <span class="mono">DEFAULT_PATTERNS["evaluation"]</span> matches the body text; otherwise <span class="mono">0</span>.
              This captures explicit cluster validation or model diagnostics (silhouette, DBI, CH, ARI/NMI, BIC/AIC, stability, modularity, etc.).
              If the value is <span class="mono">0</span>, then <span class="mono">missing_evaluation=1</span>.
            </div>
            {regex_table("evaluation")}
          </div>
        </details>
        """
    )

    blocks.append(
        f"""
        <details class="details">
          <summary><span class="mono">tuning_found</span><span class="muted">binary: any hit in DEFAULT_PATTERNS["tuning"]</span></summary>
          <div class="details-body">
            <div class="note">
              Set to <span class="mono">1</span> if at least one regex in <span class="mono">DEFAULT_PATTERNS["tuning"]</span> matches the body text; otherwise <span class="mono">0</span>.
              This captures explicit hyperparameter search language (grid/random search, Bayesian optimization, Optuna/Hyperopt, CV, etc.).
              If the value is <span class="mono">0</span>, then <span class="mono">missing_tuning=1</span>.
            </div>
            {regex_table("tuning")}
          </div>
        </details>
        """
    )

    blocks.append(
        f"""
        <details class="details">
          <summary><span class="mono">missing_params</span><span class="muted">derived: params_found empty</span></summary>
          <div class="details-body">
            <div class="note">Computed as <span class="mono">int(len(params_found)==0)</span>.</div>
          </div>
        </details>
        """
    )

    blocks.append(
        f"""
        <details class="details">
          <summary><span class="mono">missing_justification</span><span class="muted">derived: justification_found==0</span></summary>
          <div class="details-body">
            <div class="note">Computed as <span class="mono">int(justification_found==0)</span>.</div>
          </div>
        </details>
        """
    )

    blocks.append(
        f"""
        <details class="details">
          <summary><span class="mono">missing_evaluation</span><span class="muted">derived: evaluation_found==0</span></summary>
          <div class="details-body">
            <div class="note">Computed as <span class="mono">int(evaluation_found==0)</span>.</div>
          </div>
        </details>
        """
    )

    blocks.append(
        f"""
        <details class="details">
          <summary><span class="mono">missing_tuning</span><span class="muted">derived: tuning_found==0</span></summary>
          <div class="details-body">
            <div class="note">Computed as <span class="mono">int(tuning_found==0)</span>.</div>
          </div>
        </details>
        """
    )

    blocks.append(
        f"""
        <details class="details">
          <summary><span class="mono">missing_reporting_signals</span><span class="muted">derived: OR of missing flags</span></summary>
          <div class="details-body">
            <div class="note">Computed as <span class="mono">int(missing_params or missing_justification or missing_evaluation or missing_tuning)</span>.</div>
          </div>
        </details>
        """
    )

    blocks.append(
        f"""
        <details class="details">
          <summary><span class="mono">parse_errors</span><span class="muted">row status: extraction failure</span></summary>
          <div class="details-body">
            <div class="note">
              If XML parsing/extraction/analysis raises an exception, the row is still emitted with an <span class="mono">error</span> field set.
              Downstream, year-level <span class="mono">parse_errors</span> counts such rows.
            </div>
          </div>
        </details>
        """
    )

    return f"""
    <div class="panel">
      <div class="panel-title">Field meanings (article-level) — click to expand</div>
      <div class="note">Each field includes the exact operational definition and the regex bank used (verbatim).</div>
      {''.join(blocks)}
    </div>
    """

def build_html(df: pd.DataFrame, fig_main: go.Figure, fig_comp: go.Figure, title: str) -> str:
    pio_html_main = pio.to_html(fig_main, include_plotlyjs="cdn", full_html=False, config={"displaylogo": False, "responsive": True})
    pio_html_comp = pio.to_html(fig_comp, include_plotlyjs=False, full_html=False, config={"displaylogo": False, "responsive": True})
    css = """
    <style>
      :root { --bg:#0b0f19; --panel:#111827; --ink:#e5e7eb; --muted:#9ca3af; --border:#1f2937; --chip:#0f172a; }
      body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: var(--bg); color: var(--ink); }
      .wrap { max-width: 1200px; margin: 0 auto; padding: 22px 18px 40px; }
      .title { font-size: 22px; font-weight: 700; letter-spacing: 0.2px; margin: 0 0 6px; }
      .subtitle { color: var(--muted); margin: 0 0 18px; font-size: 13px; line-height: 1.35; }
      .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 14px 0 18px; }
      .card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 12px 12px 10px; }
      .card-title { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
      .card-value { font-size: 20px; font-weight: 700; }
      .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 14px; margin-top: 14px; }
      .panel-title { font-weight: 700; margin: 0 0 10px; }
      .tbl { width:100%; border-collapse: collapse; font-size: 13px; }
      .tbl th, .tbl td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
      .tbl th { color: var(--muted); font-weight: 600; }
      .note { color: var(--muted); font-size: 12px; margin-top: 10px; line-height: 1.35; }
      .plotwrap { background: #ffffff; border-radius: 14px; overflow: hidden; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; }
      details.details { border: 1px solid var(--border); border-radius: 14px; background: var(--chip); margin-top: 10px; }
      details.details summary { cursor: pointer; padding: 10px 12px; list-style: none; display:flex; gap:10px; align-items:center; }
      details.details summary::-webkit-details-marker { display:none; }
      details.details[open] summary { border-bottom: 1px solid var(--border); }
      .details-body { padding: 10px 12px 12px; }
      .muted { color: var(--muted); font-size: 12px; }
      @media (max-width: 980px) { .grid { grid-template-columns: 1fr 1fr; } }
      @media (max-width: 520px) { .grid { grid-template-columns: 1fr; } }
    </style>
    """
    body = f"""
    <div class="wrap">
      <div class="title">{title}</div>
      <div class="subtitle">Year-level summary</div>
      <div class="grid">{kpi_cards(df)}</div>

      <div class="panel">
        <div class="panel-title">Trends</div>
        <div class="plotwrap">{pio_html_main}</div>
        <div class="note">Use the range slider on the bottom panels to focus on specific eras; hover to read exact values.</div>
      </div>

      <div class="panel">
        <div class="panel-title">Composition of missing reporting categories (stacked share, % of all missing flags)</div>
        <div class="plotwrap">{pio_html_comp}</div>
        <div class="note">Stacked shares normalize the four missing categories by their per-year sum to show dominance patterns over time.</div>
      </div>

      <div class="panel">
        <div class="panel-title">Methods and definitions</div>
        <div class="note">Operational definitions used by the text-mining pipeline, with the exact regex banks embedded.</div>
      </div>

      {metric_definitions_table(df)}
      {field_meanings_interactive()}
    </div>
    """
    return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>{css}</head><body>{body}</body></html>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="/mnt/data/summary_all_years.json")
    ap.add_argument("-o", "--output", default="dashboard_summary_all_years.html")
    ap.add_argument("--title", default="Summary of Datamining (2000-2025)")
    args = ap.parse_args()

    df = load_df(Path(args.input))
    fig_main = make_main_figure(df)
    fig_comp = make_composition_figure(df)
    html = build_html(df, fig_main, fig_comp, args.title)
    Path(args.output).write_text(html, encoding="utf-8")

if __name__ == "__main__":
    main()
