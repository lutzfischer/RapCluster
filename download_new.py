#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import socket
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from Bio import Entrez
from tqdm import tqdm
import xml.etree.ElementTree as ET


Entrez.email = "ahmad.lutfi@fu-berlin.de"
Entrez.api_key = "15dc7c27a72ffe2edb0909fd2af7dbfe4b0a"

YEARS = range(2025, 2026)

BATCH_SIZE = 200        # 100–500 typical; higher = fewer requests
WORKERS = 2             # keep low; higher often triggers throttling
BATCH_THROTTLE = 0.35   # sleep per batch request (not per paper)
MAX_RETRIES = 6
BASE_BACKOFF = 0.5
TIMEOUT_SEC = 30



def setup_logging(log_dir="logs", run_name=None, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"pmc_download_{ts}"
    log_path = os.path.join(log_dir, f"{run_name}.log")

    logger = logging.getLogger("pmc_downloader")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger, log_path


def build_query_for_year(year: int) -> str:
    # Date-range dp filter is more reliable than "{year}"[dp]
    return (
        '("clustering" OR "k-means" OR "kmeans" OR "hierarchical clustering" OR "agglomerative clustering" '
        'OR "DBSCAN" OR "HDBSCAN" OR "OPTICS" OR "mean shift" OR "spectral clustering" '
        'OR "Gaussian mixture model" OR "affinity propagation" OR "Birch clustering") '
        'AND open access[filter] '
        f'AND ("{year}/01/01"[dp] : "{year}/12/31"[dp])'
    )



def esearch_history(term: str, logger):
    """
    ESearch with usehistory, returns (count, WebEnv, QueryKey)
    """
    h = Entrez.esearch(db="pmc", term=term, usehistory="y", retmax=0)
    r = Entrez.read(h)
    count = int(r["Count"])
    webenv = r["WebEnv"]
    qk = r["QueryKey"]
    logger.info(f"SEARCH count={count}")
    return count, webenv, qk


def efetch_batch(webenv: str, query_key: str, retstart: int, retmax: int, logger,
                 max_retries=MAX_RETRIES, base_backoff=BASE_BACKOFF, timeout_sec=TIMEOUT_SEC):
    """
    Fetch a batch of articles via Entrez history params.
    Retries with exponential backoff.
    """
    socket.setdefaulttimeout(timeout_sec)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            h = Entrez.efetch(
                db="pmc",
                query_key=query_key,
                WebEnv=webenv,
                retstart=retstart,
                retmax=retmax,
                rettype="full",
                retmode="xml",
            )
            data = h.read()
            return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
        except Exception as e:
            last_err = e
            sleep_s = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                f"RETRY {attempt}/{max_retries} retstart={retstart} retmax={retmax} "
                f"err={type(e).__name__}: {e} sleep={sleep_s:.2f}s"
            )
            time.sleep(sleep_s)

    raise last_err



def _find_pmcid(article_elem):
    """
    Find PMC id inside an article, if present.
    """
    # Typical: <article-id pub-id-type="pmc">1234567</article-id>
    for aid in article_elem.findall(".//article-id"):
        if aid.attrib.get("pub-id-type") == "pmc" and aid.text:
            return aid.text.strip()
    return None


def _top_level_articles(root):
    """
    Return only the top-level <article> nodes.
    Most common wrapper: <pmc-articleset><article>...</article>...</pmc-articleset>
    """
    arts = root.findall("./article")
    if arts:
        return arts

    # Fallback: sometimes there is a namespace or different wrapper; try shallow search
    # (still avoiding .//article which grabs nested)
    for child in list(root):
        if child.tag.endswith("article"):
            arts.append(child)
    return arts


def split_and_write_articles(xml_text: str, outdir: str, logger, batch_retstart: int) -> int:
    """
    Parse a batch XML and write each *top-level* article as its own XML file.
    Uses unique fallback filenames to avoid collisions across batches.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"PARSE_ERROR retstart={batch_retstart}: {e}")
        return 0

    articles = _top_level_articles(root)
    if not articles:
        logger.warning(f"No top-level <article> found retstart={batch_retstart}")
        return 0

    written = 0
    for idx, art in enumerate(articles):
        pmcid = _find_pmcid(art)
        if pmcid:
            fname = f"PMC{pmcid}.xml" if not pmcid.startswith("PMC") else f"{pmcid}.xml"
        else:
            fname = f"article_{batch_retstart + idx}.xml"

        outpath = os.path.join(outdir, fname)

        if os.path.exists(outpath):
            continue

        wrapper = ET.Element(root.tag)
        wrapper.append(art)

        ET.ElementTree(wrapper).write(outpath, encoding="utf-8", xml_declaration=True)
        written += 1

    return written


def download_year(query: str, outdir: str, logger,
                  batch_size=BATCH_SIZE, workers=WORKERS, batch_throttle=BATCH_THROTTLE):
    os.makedirs(outdir, exist_ok=True)

    expected, webenv, qk = esearch_history(query, logger)
    if expected == 0:
        logger.info(f"SUMMARY {outdir}: expected=0 written=0 failed_batches=0")
        return 0

    starts = list(range(0, expected, batch_size))

    def job(retstart):
        xml = efetch_batch(webenv, qk, retstart, batch_size, logger)
        n = split_and_write_articles(xml, outdir, logger, batch_retstart=retstart)
        time.sleep(batch_throttle)
        return n

    written_total = 0
    failed_batches = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, s) for s in starts]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{outdir} (batches)"):
            try:
                written_total += fut.result()
            except Exception as e:
                failed_batches += 1
                logger.error(f"BATCH_FAILED err={type(e).__name__}: {e}")

    logger.info(f"SUMMARY {outdir}: expected={expected} written={written_total} failed_batches={failed_batches}")
    return written_total



def main():
    logger, log_path = setup_logging()
    logger.info(f"Logging to: {log_path}")
    logger.info(
        f"CONFIG batch_size={BATCH_SIZE} workers={WORKERS} batch_throttle={BATCH_THROTTLE}s "
        f"max_retries={MAX_RETRIES} timeout={TIMEOUT_SEC}s"
    )

    for year in YEARS:
        logger.info(f"YEAR {year} start")
        q = build_query_for_year(year)
        outdir = f"pmc_articles_{year}"
        download_year(q, outdir, logger)
        logger.info(f"YEAR {year} done")

    logger.info("DONE")


if __name__ == "__main__":
    main()

