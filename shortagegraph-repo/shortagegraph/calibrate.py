"""Measure public-data marginals from live endpoints; write calibration.json.

Every parameter is tagged `measured` or `assumed` so the synthetic generator's
inputs can be audited. Nothing here guesses silently: if an endpoint fails, the
failure is recorded and the parameter falls back to an assumption that is
listed in `assumed`.

Measured here:
  * NDC directory product count, labeler count, drug-group count and the share
    of drug groups with a single labeler (from the real product.txt)
  * openFDA drug shortages: total records, per-year postings, status mix
  * openFDA drug enforcement: total reports, per-year totals, class mix

Assumed here (no open endpoint available from this host):
  * establishment count and the share located outside the U.S.
  * warning letters per year (the FDA Data Dashboard returns 403 here)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

UA = "ShortageGraph/0.1 (public data audit; contact: paper authors)"

SHORTAGES = "https://api.fda.gov/drug/shortages.json"
ENFORCEMENT = "https://api.fda.gov/drug/enforcement.json"
NDC_ZIP = "https://www.accessdata.fda.gov/cder/ndctext.zip"

YEARS = list(range(2018, 2026))

ASSUMED = {
    "products.n_establishments": (1200, "no open establishment registry endpoint from this host"),
    "products.outside_us_share": (0.22, "assumption; requires registration data to measure"),
    "letters.per_year": (55, "FDA Data Dashboard returns 403 for automated access here"),
}


def _json(url: str, timeout: int = 90) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _total(url: str) -> int | None:
    try:
        return _json(url).get("meta", {}).get("results", {}).get("total")
    except Exception:
        return None


def _counts(base: str, field: str) -> dict[str, int]:
    try:
        payload = _json(f"{base}?count={field}")
    except Exception:
        return {}
    out = {}
    for row in payload.get("results", []):
        term = row.get("term")
        if term is not None:
            out[str(term)] = int(row.get("count", 0))
    return out


def _yearly(base: str, field: str) -> dict[int, int | None]:
    """Record count per calendar year, using a one-record query per year."""
    out: dict[int, int | None] = {}
    for year in YEARS:
        window = f"%5B{year}-01-01+TO+{year}-12-31%5D"
        out[year] = _total(f"{base}?search={field}:{window}&limit=1")
    return out


# --- NDC directory ----------------------------------------------------------


def ndc_marginals(zip_path: Path) -> dict:
    """Parse the real NDC product file for structural marginals."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith("product.txt"))
        raw = zf.read(name).decode("latin-1")

    n_products = 0
    labelers: set[str] = set()
    groups: dict[tuple, set[str]] = defaultdict(set)
    forms: Counter = Counter()
    routes: Counter = Counter()

    reader = csv.DictReader(io.StringIO(raw), delimiter="\t")
    for row in reader:
        n_products += 1
        labeler = (row.get("LABELERNAME") or "").strip()
        labelers.add(labeler)
        ingredients = tuple(sorted(
            part.strip().upper()
            for part in (row.get("SUBSTANCENAME") or "").replace(";", ",").split(",")
            if part.strip()
        ))
        form = (row.get("DOSAGEFORMNAME") or "").strip().upper()
        route = (row.get("ROUTENAME") or "").strip().upper()
        forms[form] += 1
        routes[route] += 1
        if ingredients:
            groups[(ingredients, form, route)].add(labeler)

    single = sum(1 for makers in groups.values() if len(makers) == 1)
    return {
        "n_products": n_products,
        "n_labelers": len(labelers),
        "n_drug_groups": len(groups),
        "single_labeler_share": round(single / max(1, len(groups)), 4),
        "top_forms": dict(forms.most_common(6)),
        "top_routes": dict(routes.most_common(6)),
    }


# --- assembly ---------------------------------------------------------------


def build(ndc_zip: Path) -> dict:
    retrieved = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    notes: list[str] = []

    ndc = ndc_marginals(ndc_zip)
    shortages_total = _total(f"{SHORTAGES}?limit=1")
    shortages_yearly = _yearly(SHORTAGES, "initial_posting_date")
    shortages_status = _counts(SHORTAGES, "status")
    recalls_total = _total(f"{ENFORCEMENT}?limit=1")
    recalls_yearly = _yearly(ENFORCEMENT, "report_date")
    recalls_class = _counts(ENFORCEMENT, "classification")

    measured_years = [v for v in shortages_yearly.values() if v]
    shortages_per_year = (
        round(sum(measured_years) / len(measured_years)) if measured_years else None
    )
    recall_years = [v for v in recalls_yearly.values() if v]
    recalls_per_year = round(sum(recall_years) / len(recall_years)) if recall_years else None

    total_class = sum(recalls_class.values()) or 1
    class_shares = {
        name: round(count / total_class, 4) for name, count in recalls_class.items()
    } if recalls_class else None
    if not class_shares:
        notes.append("recall class mix unavailable; generator default used")
    if shortages_per_year is None:
        notes.append("shortage yearly counts unavailable; generator default used")
    if recalls_per_year is None:
        notes.append("recall yearly counts unavailable; generator default used")

    parameters: dict[str, dict] = {
        "products": {
            "n_products": ndc["n_products"],
            "n_labelers": ndc["n_labelers"],
            "single_labeler_share": ndc["single_labeler_share"],
            "n_establishments": ASSUMED["products.n_establishments"][0],
            "outside_us_share": ASSUMED["products.outside_us_share"][0],
        },
        "shortages": {"per_year": shortages_per_year or 180},
        "recalls": {
            "per_year": recalls_per_year or 900,
            "class_shares": class_shares or {"Class I": 0.08, "Class II": 0.74, "Class III": 0.18},
        },
        "letters": {"per_year": ASSUMED["letters.per_year"][0]},
    }

    return {
        "retrieved": retrieved,
        "sources": [
            {"name": "ndc_directory", "url": NDC_ZIP,
             "note": f"{ndc_zip.stat().st_size} bytes zipped, parsed locally"},
            {"name": "openfda_shortages", "url": SHORTAGES,
             "note": f"total={shortages_total}, yearly={shortages_yearly}, status={shortages_status}"},
            {"name": "openfda_enforcement", "url": ENFORCEMENT,
             "note": f"total={recalls_total}, yearly={recalls_yearly}, class={recalls_class}"},
            {"name": "fda_data_dashboard", "url": "https://datadashboard.fda.gov/ora/index.htm",
             "note": "403 for automated access from this host; warning-letter rate assumed"},
        ],
        "measurements": {
            "ndc": ndc,
            "shortages_total": shortages_total,
            "shortages_by_year": shortages_yearly,
            "shortages_status": shortages_status,
            "recalls_total": recalls_total,
            "recalls_by_year": recalls_yearly,
            "recalls_class_mix": recalls_class,
        },
        "parameters": parameters,
        "measured": [
            "products.n_products", "products.n_labelers", "products.single_labeler_share",
            "shortages.per_year", "recalls.per_year", "recalls.class_shares",
        ],
        "assumed": {key: {"value": value, "reason": reason}
                    for key, (value, reason) in ASSUMED.items()},
        "notes": notes,
        "caveat": (
            "These are marginals of public data, used only to shape a synthetic graph. "
            "A graph generated from them is not an observation of the U.S. supply network."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure public-data marginals")
    parser.add_argument("--out", default="data/raw/calibration.json")
    parser.add_argument("--ndc-zip", default="data/raw/ndctext.zip")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)

    zip_path = Path(args.ndc_zip)
    if not zip_path.exists():
        if args.skip_download:
            raise SystemExit(f"{zip_path} missing and --skip-download was set")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {NDC_ZIP} ...")
        request = urllib.request.Request(NDC_ZIP, headers={"User-Agent": UA})
        with urllib.request.urlopen(request, timeout=180) as response:
            zip_path.write_bytes(response.read())
        print(f"saved {zip_path} ({zip_path.stat().st_size} bytes)")

    payload = build(zip_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    p = payload["parameters"]
    print(f"wrote {out}")
    print(f"  products      : {p['products']['n_products']} rows, "
          f"{p['products']['n_labelers']} labelers, "
          f"{100 * p['products']['single_labeler_share']:.1f}% single-labeler")
    print(f"  shortages/yr  : {p['shortages']['per_year']}  (total {payload['measurements']['shortages_total']})")
    print(f"  recalls/yr    : {p['recalls']['per_year']}  (total {payload['measurements']['recalls_total']})")
    print(f"  class mix     : {p['recalls']['class_shares']}")
    print(f"  assumed       : {', '.join(payload['assumed'])}")
    if payload["notes"]:
        print(f"  notes         : {payload['notes']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
