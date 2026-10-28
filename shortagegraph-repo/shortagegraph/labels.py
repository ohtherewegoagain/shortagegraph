"""The label source, and the audit that says whether it can be trusted.

The paper labels a drug by the week of its *first* posting on the FDA shortage
list. That definition is only sound if the posting history is complete for the
window being labelled: if the source is missing early postings, a drug looks
"never listed" when it was already in shortage, and the model is trained on
corrupted negatives.

That is the failure mode this module exists to catch. It:

  1. pulls every shortage record openFDA serves, caching the raw payload;
  2. converts records into `ShortagePosting`s keyed the same way as graph nodes;
  3. reports per-year coverage and the share of records with usable dates;
  4. reports the match rate against a real graph's drug keys, because a silent
     key mismatch yields zero positives and a model that still "trains";
  5. refuses to certify a window whose coverage is below a stated floor.

Run it as a CLI to see the current state of the public source:

    python -m shortagegraph.labels --audit
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import SPLITS
from .graph import drug_key
from .ingest import ShortagePosting, normalise_firm

API = "https://api.fda.gov/drug/shortages.json"
UA = "ShortageGraph/0.1 (public data audit; contact: paper authors)"

# Dosage-form fallbacks for route, used only when openFDA gives none. Anything
# not listed stays empty rather than being guessed, because a guessed route
# silently changes the node key and detaches every label.
FORM_TO_ROUTE = {
    "INJECTION": "INTRAVENOUS",
    "INJECTABLE": "INTRAVENOUS",
    "INJECTION, SOLUTION": "INTRAVENOUS",
    "TABLET": "ORAL",
    "TABLET, FILM COATED": "ORAL",
    "CAPSULE": "ORAL",
    "SOLUTION": "ORAL",
    "SUSPENSION": "ORAL",
}


def parse_date(value) -> dt.date | None:
    if not value:
        return None
    text = str(value)[:10]
    # Enforcement reports use compact YYYYMMDD; shortage records use
    # MM/DD/YYYY. Both appear in the same study, so both must parse here.
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


# --- fetching ---------------------------------------------------------------


def fetch_all(cache: Path, *, page_size: int = 1000, max_pages: int = 60) -> list[dict]:
    """Page the whole shortages endpoint and cache the raw payload."""
    cached = cache / "shortages_all.json"
    if cached.exists() and cached.stat().st_size > 0:
        return json.loads(cached.read_text(encoding="utf-8"))

    records: list[dict] = []
    skip = 0
    total = None
    for _ in range(max_pages):
        url = f"{API}?limit={page_size}&skip={skip}"
        request = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        total = payload.get("meta", {}).get("results", {}).get("total", total)
        batch = payload.get("results", [])
        if not batch:
            break
        records.extend(batch)
        if total is not None and len(records) >= total:
            break
        skip += page_size

    cache.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(records), encoding="utf-8")
    return records


# --- conversion -------------------------------------------------------------


def _ingredients(record: dict) -> tuple[str, ...]:
    openfda = record.get("openfda") or {}
    names: list[str] = []
    for key in ("substance_name", "generic_name"):
        values = openfda.get(key) or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            for part in str(value).replace(";", ",").split(","):
                part = part.strip().upper()
                if part and part not in names:
                    names.append(part)
        if names:
            break
    if not names:
        generic = record.get("generic_name") or ""
        for part in str(generic).replace(";", ",").split(","):
            part = part.strip().upper()
            if part:
                names.append(part)
    return tuple(sorted(names))


def _form(record: dict) -> str:
    return " ".join(str(record.get("dosage_form") or "").upper().split())


def _route(record: dict, form: str) -> str:
    openfda = record.get("openfda") or {}
    routes = openfda.get("route") or []
    if isinstance(routes, str):
        routes = [routes]
    for route in routes:
        text = " ".join(str(route).upper().split())
        if text:
            return text
    return FORM_TO_ROUTE.get(form, "")


def to_postings(records: list[dict]) -> tuple[list[ShortagePosting], dict]:
    """Convert raw records into postings, keeping the ones we had to drop."""
    postings: list[ShortagePosting] = []
    dropped = collections.Counter()

    for record in records:
        date = parse_date(record.get("initial_posting_date"))
        if date is None:
            dropped["no_initial_posting_date"] += 1
            continue
        ingredients = _ingredients(record)
        if not ingredients:
            dropped["no_ingredients"] += 1
            continue
        form = _form(record)
        route = _route(record, form)
        postings.append(
            ShortagePosting(
                drug_key=(drug_key(ingredients, form, route).split("|")[0], form, route),
                posting_date=date,
                product_name=" ".join(str(record.get("generic_name") or "").split()),
                planned_discontinuation=str(record.get("update_type") or "").lower().startswith("discontinu"),
            )
        )

    # A drug can appear once per package; the label is the first posting.
    first: dict[tuple[str, str, str], dt.date] = {}
    for posting in postings:
        key = posting.drug_key
        if key not in first or posting.posting_date < first[key]:
            first[key] = posting.posting_date
    deduped = [
        ShortagePosting(drug_key=key, posting_date=date, product_name="", planned_discontinuation=False)
        for key, date in first.items()
    ]
    deduped.sort(key=lambda p: p.posting_date)
    return deduped, {"dropped": dict(dropped), "records": len(records), "postings": len(postings)}


# --- audits -----------------------------------------------------------------


def coverage(postings: list[ShortagePosting]) -> dict:
    """Per-year posting counts over the study window."""
    years = list(range(SPLITS[0].start.year, SPLITS[-1].end.year + 1))
    counts = collections.Counter(p.posting_date.year for p in postings)
    return {
        "years": years,
        "counts": {str(y): counts.get(y, 0) for y in years},
        "total_in_window": sum(counts.get(y, 0) for y in years),
        "first_posting": min((p.posting_date for p in postings), default=None).isoformat()
        if postings
        else None,
        "last_posting": max((p.posting_date for p in postings), default=None).isoformat()
        if postings
        else None,
    }


def split_coverage(postings: list[ShortagePosting]) -> dict:
    """Onsets per split, so an empty training window is visible immediately."""
    out: dict[str, int] = {}
    for split in SPLITS:
        out[split.name] = sum(
            1 for p in postings if split.start <= p.posting_date <= split.end
        )
    return out


def key_match_rate(postings: list[ShortagePosting], drug_keys: set[str]) -> dict:
    """Share of postings whose key exists in a real graph.

    A low value means labels would attach to nothing: the model would train on
    all negatives and report a degenerate, still-plausible-looking result.
    """
    normalised = {drug_key(k.split("|")[0], k.split("|")[1], k.split("|")[2]) for k in drug_keys}
    matched = 0
    examples: list[str] = []
    for posting in postings:
        key = drug_key(posting.drug_key[0], posting.drug_key[1], posting.drug_key[2])
        if key in normalised:
            matched += 1
        elif len(examples) < 5:
            examples.append(key)
    total = len(postings) or 1
    return {
        "matched": matched,
        "total": len(postings),
        "rate": round(matched / total, 4),
        "unmatched_examples": examples,
    }


@dataclass
class Certification:
    ok: bool
    reasons: list[str]
    coverage: dict
    splits: dict


def certify(
    postings: list[ShortagePosting],
    *,
    min_split_onsets: int = 25,
    min_year_onsets: int = 20,
) -> Certification:
    """Decide whether the label window is fit to train on.

    Two independent failures are checked, because they catch different defects:

      * a split with too few onsets overall (nothing to learn from);
      * any single year with implausibly few onsets. The FDA recorded hundreds
        of shortages a year in this period, so a year with a handful means the
        source has lost those records, not that shortages paused. A live
        snapshot drops resolved entries, which is exactly how a history becomes
        survivorship-biased.
    """
    reasons: list[str] = []
    cov = coverage(postings)
    splits = split_coverage(postings)

    for name, count in splits.items():
        if count < min_split_onsets:
            reasons.append(f"split '{name}' has {count} onsets (< {min_split_onsets})")

    counts = cov["counts"]
    for split in SPLITS:
        years = list(range(split.start.year, split.end.year + 1))
        values = [counts.get(str(y), 0) for y in years]
        thin = {y: v for y, v in zip(years, values) if v < min_year_onsets}
        if thin:
            reasons.append(
                f"split '{split.name}' has years below {min_year_onsets} onsets: {thin}. "
                "A live shortage snapshot loses resolved entries, so a thin year "
                "indicates survivorship bias rather than a quiet year."
            )
    return Certification(ok=not reasons, reasons=reasons, coverage=cov, splits=splits)


def certification_summary(cert: Certification) -> dict:
    """Flat, JSON-serialisable view of a Certification for reports."""
    return {
        "certified": cert.ok,
        "reasons": cert.reasons,
        "coverage": cert.coverage,
        "split_onsets": cert.splits,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the shortage label source")
    parser.add_argument("--cache", default="data/raw")
    parser.add_argument("--audit", action="store_true", help="fetch (or reuse cache) and report")
    parser.add_argument("--out", default="data/raw/label_coverage.json")
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    records = fetch_all(cache)
    postings, drops = to_postings(records)
    cert = certify(postings)

    payload = {
        "fetched_records": drops,
        "coverage": cert.coverage,
        "split_onsets": cert.splits,
        "certified": cert.ok,
        "reasons": cert.reasons,
        "note": (
            "First-posting labels are only valid where this source is complete. "
            "Uncertified windows must not be trained on."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"records: {drops['records']}  postings: {drops['postings']}  deduped: {len(postings)}")
    print(f"dropped: {drops['dropped']}")
    print(f"coverage by year: {cert.coverage['counts']}")
    print(f"onsets per split: {cert.splits}")
    if cert.ok:
        print("certified: YES")
    else:
        print("certified: NO")
        for reason in cert.reasons:
            print(f"  - {reason}")
    print(f"wrote {out}")
    return 0 if cert.ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
