"""Real dataset statistics (Table I) from the actual NDC directory.

This is the one part of the paper that can be produced honestly without a
complete shortage history: the supply-network structure. Every number written
here comes from the downloaded `ndctext.zip` and is measured, not generated.

What is measured:
  * active product count and distinct drug nodes per sampled week
  * distinct labeler nodes per sampled week
  * `markets` edges (drug-labeler pairs) and `substitutes` edges per week
  * share of drug nodes with a single labeler

What is NOT measured here, and why:
  * establishment nodes and `made at` edges — they need the NDC->FEI mapping
    parsed from DailyMed SPL labeling, which is a separate ingestion job
  * warning-letter counts — the FDA Data Dashboard returns 403 here
  * first-onset events — depends on the shortage history, which currently fails
    its own audit (see shortagegraph.labels)

Those cells are emitted as null rather than zero, so an absent value stays
visibly absent instead of silently wrong.

    python -m shortagegraph.realdata --out data/raw/real_dataset_stats.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .config import SPLITS
from .graph import drug_key
from .ingest import fetch_ndc_directory, load_ndc_products


def sample_weeks(start: dt.date, end: dt.date, cadence: str = "monthly") -> list[dt.date]:
    """First Monday of each month (or year) inside the study window."""
    weeks: list[dt.date] = []
    cursor = dt.date(start.year, start.month, 1)
    while cursor <= end:
        monday = cursor - dt.timedelta(days=cursor.weekday())
        if monday < start:
            monday = start - dt.timedelta(days=start.weekday())
        if start <= monday <= end:
            weeks.append(monday)
        if cadence == "yearly":
            cursor = dt.date(cursor.year + 1, 1, 1)
        else:
            year, month = (cursor.year + 1, 1) if cursor.month == 12 else (cursor.year, cursor.month + 1)
            cursor = dt.date(year, month, 1)
    return weeks


def measure(products, weeks: list[dt.date]) -> dict:
    """Structural statistics per sampled week."""
    per_week = []
    for week in weeks:
        active = [
            p for p in products
            if (not p.marketing_start or p.marketing_start <= week)
            and (not p.marketing_end or p.marketing_end >= week)
        ]
        groups: dict[str, set[str]] = defaultdict(set)
        for product in active:
            key = drug_key(product.ingredient_set, product.dosage_form, product.route)
            groups[key].add(product.labeler_code)

        single = sum(1 for makers in groups.values() if len(makers) == 1)

        by_ingredients: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for key in groups:
            ingredients = tuple((key.split("|") + [""])[0].split("+"))
            by_ingredients[ingredients].append(key)
        substitutes = 0
        for keys in by_ingredients.values():
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    _, form_a, route_a = (keys[i].split("|") + ["", ""])[:3]
                    _, form_b, route_b = (keys[j].split("|") + ["", ""])[:3]
                    if (form_a, route_a) != (form_b, route_b):
                        substitutes += 2  # both directions, as the builder does

        per_week.append(
            {
                "week": week.isoformat(),
                "active_products": len(active),
                "drug_nodes": len(groups),
                "labeler_nodes": len({p.labeler_code for p in active}),
                "market_edges": len(active),
                "substitute_edges": substitutes,
                "single_labeler_share": round(single / max(1, len(groups)), 4),
            }
        )
    return {"weeks": per_week}


def summarise(series: dict) -> dict:
    weeks = series["weeks"]
    if not weeks:
        return {}

    def med(key: str) -> float:
        return float(statistics.median(w[key] for w in weeks))

    return {
        "sampled_weeks": len(weeks),
        "sampled_from": weeks[0]["week"],
        "sampled_to": weeks[-1]["week"],
        "active_products_median": int(med("active_products")),
        "drug_nodes_median": int(med("drug_nodes")),
        "labeler_nodes_median": int(med("labeler_nodes")),
        "market_edges_median": int(med("market_edges")),
        "substitute_edges_median": int(med("substitute_edges")),
        "single_labeler_pct_median": round(100 * med("single_labeler_share"), 1),
    }


def table_rows(summary: dict) -> dict:
    """Table I rows: a measured value, or None where the input is unavailable."""
    edges = (summary.get("market_edges_median") or 0) + (summary.get("substitute_edges_median") or 0)
    return {
        "Weekly snapshots": summary.get("sampled_weeks"),
        "Drug nodes |VD| (median per snapshot)": summary.get("drug_nodes_median"),
        "Labeler nodes |VL|": summary.get("labeler_nodes_median"),
        "Establishment nodes |VF|": None,
        "Edges (all relations, median per snapshot)": edges or None,
        "Drugs with a single labeler (%)": summary.get("single_labeler_pct_median"),
        "Recall / enforcement reports linked": None,
        "Warning letters linked to an FEI": None,
        "First-onset events (train / val / test)": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure real dataset statistics")
    parser.add_argument("--cache", default="data/raw")
    parser.add_argument("--out", default="data/raw/real_dataset_stats.json")
    parser.add_argument("--cadence", choices=("monthly", "yearly"), default="monthly")
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    ndc_dir = fetch_ndc_directory(cache)
    products = load_ndc_products(ndc_dir)

    weeks = sample_weeks(SPLITS[0].start, SPLITS[-1].end, args.cadence)
    series = measure(products, weeks)
    summary = summarise(series)

    payload = {
        "source": str(ndc_dir),
        "measured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "cadence": args.cadence,
        "products_parsed": len(products),
        "summary": summary,
        "table_rows": table_rows(summary),
        "null_reasons": {
            "Establishment nodes |VF|": "needs the NDC->FEI mapping from DailyMed SPL",
            "Recall / enforcement reports linked": "needs recall-product linkage over the full NDC set",
            "Warning letters linked to an FEI": "FDA Data Dashboard returns 403 to automated clients here",
            "First-onset events (train / val / test)": "shortage history fails its completeness audit",
        },
        "series": series,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"parsed {len(products)} NDC products; sampled {len(weeks)} weeks")
    for row, value in payload["table_rows"].items():
        print(f"  {row:<46} {'NOT AVAILABLE' if value is None else value}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
