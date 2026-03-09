import re
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

WS_RE = re.compile(r"\s+")

def norm_ws(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip())

def nsmap_from_root(root):
    if root.tag.startswith("{") and "}" in root.tag:
        uri = root.tag.split("}", 1)[0][1:]
        return {"ns": uri}
    return None

def element_text(el) -> str:
    if el is None:
        return ""
    return norm_ws("".join(el.itertext()))

def find_first(root, path_no_ns: str, path_ns: str, ns):
    if ns:
        el = root.find(path_ns, ns)
        if el is not None:
            return el
    return root.find(path_no_ns)

def find_all(root, path_no_ns: str, path_ns: str, ns):
    if ns:
        els = root.findall(path_ns, ns)
        if els:
            return els
    return root.findall(path_no_ns)

def extract_pmc_fields(xml_path: Path):
    root = ET.parse(str(xml_path)).getroot()
    ns = nsmap_from_root(root)

    title_el = find_first(root, ".//article-title", ".//ns:article-title", ns)
    title = element_text(title_el)

    abs_els = find_all(root, ".//abstract", ".//ns:abstract", ns)
    abstracts = [element_text(a) for a in abs_els]
    abstracts = [a for a in abstracts if a]
    abstract = norm_ws("\n\n".join(abstracts))

    body_el = find_first(root, ".//body", ".//ns:body", ns)
    body_text = element_text(body_el)

    return title, abstract, body_text

def process_year_dir(d: Path):
    xmls = sorted(d.glob("*.xml"))
    if not xmls:
        print(f"[SKIP] {d} (no .xml files found)")
        return []

    rows = []
    for xml_path in tqdm(xmls, desc=f"Processing {d.name}"):
        try:
            title, abstract, body_text = extract_pmc_fields(xml_path)
            err = ""
        except Exception as e:
            title, abstract, body_text, err = "", "", "", str(e)

        rows.append({
            "pmc_id": xml_path.stem,
            "title": title,
            "abstract": abstract,
            "body_text": body_text,
            "error": err,
        })
    return rows

def write_csv(rows, out_path: Path):
    fieldnames = ["pmc_id", "title", "abstract", "body_text", "error"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def iter_year_dirs(base: Path):
    """
    Find subfolders in current directory that contain a 4-digit year anywhere in the name.
    Examples it matches:
      pmc2000
      pmc_articles_2000
      pmc_articles_2023
      2005
    """
    year_pat = re.compile(r"(19\d{2}|20\d{2})")  # 1900-2099
    candidates = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        m = year_pat.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    for _, d in sorted(candidates):
        yield d

if __name__ == "__main__":
    base = Path(".").resolve()
    print(f"[INFO] Running in: {base}")

    year_dirs = list(iter_year_dirs(base))
    if not year_dirs:
        print("[ERROR] No year-like folders found in current directory.")
        print("        Put this script in the parent folder containing your year directories,")
        print("        then run: python extract_pmc.py")
        raise SystemExit(1)

    for d in year_dirs:
        year = re.search(r"(19\d{2}|20\d{2})", d.name).group(1)
        out_file = base / f"pmc_text_{year}.csv"
        print(f"[INFO] Folder: {d.name} -> {out_file.name}")

        rows = process_year_dir(d)
        if rows:
            write_csv(rows, out_file)
            print(f"[OK] Saved: {out_file} (rows={len(rows)})")
