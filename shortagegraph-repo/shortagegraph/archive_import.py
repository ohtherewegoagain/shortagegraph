"""Import a complete shortage posting history from an archived export.

The public endpoint is a live snapshot and fails `labels.certify`. When a
complete history is obtained — an FDA archive export, an ASHP historical file,
or a health system's own records — this is the door it comes in through.

Accepted inputs:

  CSV    columns: posting_date, and either drug_key ("ING+ING|FORM|ROUTE") or
         the trio generic_name, dosage_form, route. Optional: product_name,
         planned_discontinuation.
  JSONL/JSON  one object per record, same field names.

Everything is normalised through `graph.drug_key`, de-duplicated to the first
posting per drug, written as the `shortages.csv` that `ingest.load_shortages`
reads, and put through the same certification gate as the live source. It exits
non-zero when the imported history is still incomplete, because an import that
looks clean but is missing years is precisely the failure the gate exists for.

    python -m shortagegraph.archive_import --source history.csv \
        --out data/raw/shortages.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .graph import drug_key
from .ingest import ShortagePosting
from .labels import certification_summary, certify, parse_date


def _from_row(row: dict) -> ShortagePosting | None:
    date = parse_date(row.get("posting_date") or row.get("date") or row.get("initial_posting_date"))
    if date is None:
        return None

    if row.get("drug_key"):
        joined = str(row["drug_key"]).split("|")[0]
        form = (str(row["drug_key"]).split("|") + [""])[1] if "|" in str(row["drug_key"]) else ""
        route = (str(row["drug_key"]).split("|") + ["", ""])[2] if "|" in str(row["drug_key"]) else ""
        ingredients = joined.replace("|", "+")
    else:
        ingredients = row.get("generic_name") or row.get("ingredients") or ""
        form = row.get("dosage_form") or row.get("form") or ""
        route = row.get("route") or ""

    key = drug_key(ingredients, form, route)
    ingredients_part, form_part, route_part = (key.split("|") + ["", ""])[:3]
    return ShortagePosting(
        drug_key=(ingredients_part, form_part, route_part),
        posting_date=date,
        product_name=str(row.get("product_name") or ""),
        planned_discontinuation=str(row.get("planned_discontinuation") or "").strip().lower()
        in {"1", "true", "yes"},
    )


def load_history(path: Path) -> tuple[list[ShortagePosting], dict]:
    """Read CSV/JSONL/JSON into postings, reporting what was dropped."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("results", [])
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

    postings: list[ShortagePosting] = []
    dropped = 0
    for row in rows:
        posting = _from_row(row)
        if posting is None:
            dropped += 1
        else:
            postings.append(posting)

    first: dict[tuple[str, str, str], ShortagePosting] = {}
    for posting in postings:
        key = posting.drug_key
        if key not in first or posting.posting_date < first[key].posting_date:
            first[key] = posting

    deduped = sorted(first.values(), key=lambda p: p.posting_date)
    return deduped, {
        "rows_read": len(rows),
        "dropped_no_date": dropped,
        "postings": len(deduped),
        "duplicates_collapsed": len(postings) - len(deduped),
    }


def write_shortages_csv(postings: list[ShortagePosting], path: Path) -> int:
    """Write the exact schema `ingest.load_shortages` expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["drug_key", "posting_date", "product_name", "planned_discontinuation"])
        for posting in postings:
            writer.writerow(
                [
                    "|".join(posting.drug_key),
                    posting.posting_date.isoformat(),
                    posting.product_name,
                    int(posting.planned_discontinuation),
                ]
            )
    return len(postings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import an archived shortage history")
    parser.add_argument("--source", required=True, help="CSV, JSONL or JSON export")
    parser.add_argument("--out", default="data/raw/shortages.csv")
    parser.add_argument("--report", default="data/raw/archive_import_report.json")
    parser.add_argument("--allow-uncertified", action="store_true",
                        help="write the CSV even if the history fails certification")
    args = parser.parse_args(argv)

    postings, meta = load_history(Path(args.source))
    cert = certify(postings)

    report = {"source": args.source, "import": meta, **certification_summary(cert)}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"read {meta['rows_read']} rows -> {meta['postings']} postings "
          f"({meta['duplicates_collapsed']} duplicates collapsed, {meta['dropped_no_date']} dropped)")
    print(f"onsets per split: {cert.splits}")
    print(f"coverage by year: {cert.coverage['counts']}")

    if not cert.ok:
        print("certified: NO")
        for reason in cert.reasons:
            print(f"  - {reason}")
        if not args.allow_uncertified:
            print("refusing to write shortages.csv; pass --allow-uncertified to inspect anyway")
            print(f"wrote {report_path}")
            return 2

    written = write_shortages_csv(postings, Path(args.out))
    print(f"certified: {'YES' if cert.ok else 'NO (override)'}")
    print(f"wrote {args.out} ({written} postings)")
    print(f"wrote {report_path}")
    return 0 if cert.ok else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
