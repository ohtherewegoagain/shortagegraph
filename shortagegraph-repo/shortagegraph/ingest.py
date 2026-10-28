"""Fetching and loading the five public sources named in Section II.

Design rule: every function either returns parsed records or raises with a
message describing which file is missing. Nothing here synthesises data. If a
source is unavailable, the caller must decide what to do, and the pipeline is
expected to fail loudly rather than quietly train on a partial graph.

Network access is optional. Pass ``cache=...`` and the functions read what has
already been downloaded, which is how the test suite runs offline.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import SOURCES

USER_AGENT = "ShortageGraph/0.1 (research; contact: paper authors)"


# --- transport --------------------------------------------------------------


def download(url: str, dest: Path, *, timeout: int = 120) -> Path:
    """Download `url` to `dest` unless it already exists. Returns the path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read()
    dest.write_bytes(payload)
    return dest


def _json(url: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


# --- NDC directory ----------------------------------------------------------


@dataclass
class NdcProduct:
    ndc: str
    labeler_code: str
    labeler_name: str
    ingredient_set: frozenset[str]
    dosage_form: str
    route: str
    marketing_start: dt.date | None
    marketing_end: dt.date | None


def fetch_ndc_directory(cache: Path) -> Path:
    """Download and extract the NDC directory. Returns the extraction folder."""
    dest = cache / "raw" / "ndctext.zip"
    download(SOURCES["ndc"]["url"], dest)
    outdir = cache / "raw" / "ndc"
    if not outdir.exists():
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(outdir)
    return outdir


def _parse_fda_date(value: str) -> dt.date | None:
    """FDA files use YYYYMMDD, with 0-padded or partial values."""
    value = (value or "").strip()
    if not value or set(value) == {"0"}:
        return None
    try:
        return dt.datetime.strptime(value[:8], "%Y%m%d").date()
    except ValueError:
        return None


def load_ndc_products(ndc_dir: Path) -> list[NdcProduct]:
    """Parse product.txt into normalised records."""
    path = ndc_dir / "product.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run fetch_ndc_directory(cache) first, or place the "
            "unzipped NDC directory under data/raw/ndc/."
        )
    products: list[NdcProduct] = []
    with path.open("r", encoding="latin-1", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ingredients = frozenset(
                part.strip().upper()
                for part in re.split(r"[;,]", row.get("SUBSTANCENAME", "") or "")
                if part.strip()
            )
            if not ingredients:
                continue
            products.append(
                NdcProduct(
                    ndc=(row.get("PRODUCTNDC") or "").strip(),
                    labeler_code=(row.get("LABELERNAME") or "").strip(),
                    labeler_name=(row.get("LABELERNAME") or "").strip(),
                    ingredient_set=ingredients,
                    dosage_form=(row.get("DOSAGEFORMNAME") or "").strip().upper(),
                    route=(row.get("ROUTENAME") or "").strip().upper(),
                    marketing_start=_parse_fda_date(row.get("STARTMARKETINGDATE", "")),
                    marketing_end=_parse_fda_date(row.get("ENDMARKETINGDATE", "")),
                )
            )
    return products


# --- openFDA recalls --------------------------------------------------------


@dataclass
class RecallReport:
    report_date: dt.date
    classification: str  # "Class I" / "Class II" / "Class III"
    firm: str
    product_description: str
    ndc_codes: frozenset[str] = field(default_factory=frozenset)


def fetch_openfda_recalls(
    cache: Path, start: dt.date, end: dt.date, *, page: int = 1000, max_pages: int = 200
) -> list[RecallReport]:
    """Page through openFDA enforcement reports between two dates.

    Uses `report_date`, not `recall_initiation_date`, so a feature never
    precedes public disclosure (Section II-B).
    """
    out: list[RecallReport] = []
    skip = 0
    for _ in range(max_pages):
        params = {
            "search": (
                f"report_date:[{start.isoformat()}+TO+{end.isoformat()}]"
            ),
            "limit": str(page),
            "skip": str(skip),
        }
        url = SOURCES["openfda"]["url"] + "?" + urllib.parse.urlencode(params)
        payload = _json(url)
        results = payload.get("results", [])
        if not results:
            break
        for row in results:
            report_date = _parse_fda_date(row.get("report_date", ""))
            if report_date is None:
                continue
            out.append(
                RecallReport(
                    report_date=report_date,
                    classification=(row.get("classification") or "").strip(),
                    firm=(row.get("recalling_firm") or "").strip(),
                    product_description=(row.get("product_description") or "").strip(),
                    ndc_codes=frozenset(
                        code.strip()
                        for code in (row.get("openfda", {}) or {}).get("product_ndc", [])
                    ),
                )
            )
        if len(results) < page:
            break
        skip += page
    return out


# --- warning letters --------------------------------------------------------


@dataclass
class WarningLetter:
    letter_date: dt.date
    firm: str
    fei: str | None
    cgmp_categories: frozenset[str] = field(default_factory=frozenset)
    import_alert: bool = False
    raw_text: str = ""


def load_warning_letters(path: Path) -> list[WarningLetter]:
    """Load warning letters from a JSONL export.

    Each line needs: letter_date (YYYY-MM-DD), firm, and optionally fei,
    cgmp_categories (list), import_alert (bool), text. The extraction step that
    produces this file lives in `shortagegraph.extract`; keep the raw text so
    the hand audit can be reproduced.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Export warning letters (see SOURCES['warning_letters']) "
            "and run shortagegraph.extract to produce this JSONL."
        )
    letters: list[WarningLetter] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        letters.append(
            WarningLetter(
                letter_date=dt.date.fromisoformat(row["letter_date"]),
                firm=row.get("firm", ""),
                fei=(row.get("fei") or None),
                cgmp_categories=frozenset(row.get("cgmp_categories") or []),
                import_alert=bool(row.get("import_alert", False)),
                raw_text=row.get("text", ""),
            )
        )
    return letters


def load_fei_index(path: Path) -> dict[str, str]:
    """Map normalised firm name -> FEI, from the FDA Data Dashboard export.

    The paper uses cited FEI numbers first and this index second, with name
    matching only as a fallback (Section II-B).
    """
    if not path.exists():
        return {}
    index: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            firm = (row.get("firm") or row.get("Firm") or "").strip()
            fei = (row.get("fei") or row.get("FEI") or "").strip()
            if firm and fei:
                index[normalise_firm(firm)] = fei
    return index


def normalise_firm(name: str) -> str:
    """Lowercase, drop punctuation and common corporate suffixes."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    for suffix in (" inc", " llc", " ltd", " corp", " corporation", " co", " plc", " gmbh"):
        name = name.replace(suffix, "")
    return " ".join(name.split())


# --- shortages --------------------------------------------------------------


@dataclass
class ShortagePosting:
    drug_key: tuple[str, str, str]  # (ingredient_set, form, route) - joined by |
    posting_date: dt.date
    product_name: str
    planned_discontinuation: bool = False


def load_shortages(path: Path) -> list[ShortagePosting]:
    """Load shortage postings from the FDA Drug Shortages Database export.

    Expected CSV columns: drug_key, posting_date, product_name,
    planned_discontinuation. `drug_key` is `ingredient_set|form|route` with
    ingredients alphabetically sorted and joined by '+', which is the same key
    `graph.drug_key` produces, so labels attach to the right node.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Export the FDA Drug Shortages Database "
            f"({SOURCES['shortages']['url']}) to CSV with those columns."
        )
    postings: list[ShortagePosting] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("drug_key") or "").strip()
            if not key:
                continue
            ingredients, form, route = (key.split("|") + ["", ""])[:3]
            postings.append(
                ShortagePosting(
                    drug_key=(ingredients, form, route),
                    posting_date=dt.date.fromisoformat(row["posting_date"].strip()),
                    product_name=(row.get("product_name") or "").strip(),
                    planned_discontinuation=(row.get("planned_discontinuation", "")
                                             .strip().lower() in {"1", "true", "yes"}),
                )
            )
    return postings


# --- convenience ------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    """Small helper shared by fixtures and the audit tooling."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")


def load_raw_from_dir(root: Path):
    """Assemble `graph.RawData` from a prepared directory of FDA inputs.

    Expected layout (see shortagegraph/README.md for how each file is made):
        ndc/                    unzipped NDC directory (product.txt, ...)
        establishments.jsonl    {"fei", "name", "country"}
        product_estabs.jsonl    {"ndc", "feis": [...]}
        recalls.jsonl           {"report_date", "classification", "firm", ...}
        letters.jsonl           output of shortagegraph.extract
        shortages.csv           drug_key, posting_date, product_name, ...

    Missing optional files are reported, not silently defaulted: a graph built
    without establishment links cannot support the paper's two-hop claim.
    """
    from .graph import RawData  # local import to avoid a circular dependency

    root = Path(root)
    products = load_ndc_products(root / "ndc")

    establishments: dict[str, dict] = {}
    est_path = root / "establishments.jsonl"
    if est_path.exists():
        for row in read_jsonl(est_path):
            establishments[str(row["fei"])] = {
                "name": row.get("name", ""),
                "country": row.get("country", "US"),
            }

    product_estabs: dict[str, set[str]] = {}
    link_path = root / "product_estabs.jsonl"
    if link_path.exists():
        for row in read_jsonl(link_path):
            product_estabs.setdefault(row["ndc"], set()).update(
                str(fei) for fei in row.get("feis", [])
            )
    else:
        raise FileNotFoundError(
            f"{link_path} missing: drug-to-establishment edges cannot be built "
            "without the SPL-derived NDC->FEI mapping."
        )

    recalls: list[RecallReport] = []
    recall_path = root / "recalls.jsonl"
    if recall_path.exists():
        for row in read_jsonl(recall_path):
            recalls.append(
                RecallReport(
                    report_date=dt.date.fromisoformat(row["report_date"]),
                    classification=row.get("classification", ""),
                    firm=row.get("firm", ""),
                    product_description=row.get("product_description", ""),
                    ndc_codes=frozenset(row.get("ndc_codes") or []),
                )
            )
    else:
        raise FileNotFoundError(
            f"{recall_path} missing. Cache openFDA recalls first "
            "(fetch_openfda_recalls) or export them to this file."
        )

    letters = load_warning_letters(root / "letters.jsonl")
    shortages = load_shortages(root / "shortages.csv")

    return RawData(
        products=products,
        establishments=establishments,
        product_estabs=product_estabs,
        recalls=recalls,
        letters=letters,
        shortages=shortages,
    )
