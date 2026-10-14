"""
src/fetch_data.py -- download FinanceBench into a local cache, phase by phase.

The PDFs are only needed for one-time ingestion (parse -> chunk -> embed), so we
download just what the current phase needs and never re-download. Everything
downstream (retrieval, evals, the app) runs off the vector store.

    uv run python -m src.fetch_data meta      -> question + document metadata only
    uv run python -m src.fetch_data phase1    -> metadata + PDFs listed for phase1
    uv run python -m src.fetch_data phase3    -> ...all 368 PDFs

Which PDFs belong to a phase is defined in config/corpus.yaml, by COMPANY KEY:
the filename prefix before the period, e.g. "3M" matches 3M_2015_10K.pdf,
3M_2023Q2_10Q.pdf, ... A phase can also be the string "all".

FinanceBench ships no license file, so the cache lives in financebench/ (git-
ignored) and is never committed -- we link to the source repo instead.
"""

import json
import sys
import urllib.request
from pathlib import Path

import yaml

REPO = "patronus-ai/financebench"
BRANCH = "main"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
API_LISTING = f"https://api.github.com/repos/{REPO}/contents/pdfs"

CACHE = Path("financebench")
MANIFEST = CACHE / "pdf_manifest.json"   # cached list of every PDF filename in the repo
CORPUS_CONFIG = Path("config/corpus.yaml")

METADATA_FILES = [
    "data/financebench_open_source.jsonl",
    "data/financebench_document_information.jsonl",
]


# 1. DOWNLOAD ---- one file into the cache, skipping anything already there
def fetch(rel_path):
    dest = CACHE / rel_path
    if dest.exists() and dest.stat().st_size > 0:
        return dest, False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")      # no half-written files on crash
    urllib.request.urlretrieve(f"{RAW}/{rel_path}", tmp)
    tmp.replace(dest)
    return dest, True


def fetch_metadata():
    for rel in METADATA_FILES:
        path, new = fetch(rel)
        print(f"{'downloaded' if new else 'cached    '}  {path}")


# 2. LIST ---- every PDF in the repo (GitHub API, cached so we ask once)
def list_all_pdfs():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    with urllib.request.urlopen(API_LISTING) as resp:
        entries = json.load(resp)
    names = sorted(e["name"] for e in entries if e["name"].endswith(".pdf"))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(names, indent=1))
    return names


# filename quirks in the source repo: one typo, one casing variant
KEY_ALIASES = {"ACTIVSIONBLIZZARD": "ACTIVISIONBLIZZARD"}


def company_key(pdf_name):
    # "3M_2018_10K.pdf" -> "3M";  "Pfizer_2023Q2_10Q.pdf" -> "PFIZER"
    key = pdf_name.split("_")[0].upper()
    return KEY_ALIASES.get(key, key)


# 3. SELECT ---- the PDFs a phase asks for
def pdfs_for_phase(phase):
    config = yaml.safe_load(CORPUS_CONFIG.read_text())
    if phase not in config:
        raise SystemExit(f"unknown phase {phase!r}; defined: {list(config)}")
    wanted = config[phase]
    all_pdfs = list_all_pdfs()
    if wanted == "all":
        return all_pdfs
    wanted = set(wanted)
    selected = [p for p in all_pdfs if company_key(p) in wanted]
    missing = wanted - {company_key(p) for p in selected}
    if missing:
        raise SystemExit(f"no PDFs found for company keys: {sorted(missing)}")
    return selected


def fetch_phase(phase):
    pdfs = pdfs_for_phase(phase)
    print(f"{phase}: {len(pdfs)} PDFs")
    for i, name in enumerate(pdfs, 1):
        path, new = fetch(f"pdfs/{name}")
        with open(path, "rb") as f:
            if f.read(5) != b"%PDF-":
                raise SystemExit(f"{path} is not a PDF (LFS pointer or error page?)")
        print(f"  [{i}/{len(pdfs)}] {'downloaded' if new else 'cached    '}  {name}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "meta"
    fetch_metadata()
    if target != "meta":
        fetch_phase(target)
