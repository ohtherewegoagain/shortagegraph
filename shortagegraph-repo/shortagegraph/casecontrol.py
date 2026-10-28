"""Cross-sectional case-control analysis of the shortage snapshot.

Why this module exists
----------------------
The retrospective forecasting design needs a complete first-onset history, and
`shortagegraph.labels` proved that history is not available: the openFDA mirror
drops resolved shortages, so pre-2021 postings are survivorship-biased. No
amount of modelling repairs that.

This module answers a question that the *available* data can support. At a
single reference date, the shortage list gives a set of drugs in shortage
(cases) and the NDC directory gives every marketed drug (the risk set). For
each we can measure two exposures without needing any history of shortages:

  * concentration — how many distinct labelers market the drug (sole-source
    or not), measured from the NDC directory alone;
  * recall exposure — whether a product of that drug was the subject of an
    FDA enforcement report in a fixed window before the reference date.

Both exposures and the outcome are observed at one time point, so nothing here
depends on the corrupted posting history. The design is a prevalence
case-control (strictly: a cross-sectional association study), and it is
reported as such -- it estimates association, not incidence or causation.

Reference date
--------------
`SNAPSHOT` is the latest `initial_posting_date` in the mirror. That is a
property of the data, not a choice: the mirror's postings stop there, so any
drug first posted later cannot be a case. Callers must keep this date when
describing the study.

    python -m shortagegraph.casecontrol --window-months 24
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .config import SOURCES
from .graph import drug_key
from .ingest import (
    NdcProduct,
    fetch_ndc_directory,
    load_ndc_products,
    normalise_firm,
)
from .labels import FORM_TO_ROUTE, fetch_all as fetch_shortages, parse_date

RECALL_API = SOURCES["openfda"]["url"]
UA = "ShortageGraph/0.1 (public data study; contact: paper authors)"


# --- NDC normalisation ------------------------------------------------------


def ndc_key(code: str) -> str:
    """Normalise an NDC product code to a comparable digit string.

    openFDA returns `0002-1430`; the NDC directory uses the same shape, but
    padding varies across files. Squashing to digits and zero-padding each
    segment to four digits makes the two directly comparable.
    """
    parts = [p for p in re.split(r"[^0-9]+", (code or "").strip()) if p]
    if not parts:
        return ""
    if len(parts) == 1:
        digits = parts[0]
        if len(digits) == 8:
            return digits
        return digits.zfill(8)
    padded = "".join(part.zfill(4) for part in parts[:2])
    return padded


# --- recall cache -----------------------------------------------------------


def fetch_enforcement(
    cache: Path, start: dt.date, end: dt.date, *, page: int = 1000, max_pages: int = 60
) -> list[dict]:
    """Page openFDA enforcement reports over a window and cache the raw rows.

    Cached to `recalls_<start>_<end>.json` so the study is reproducible offline
    and a rerun cannot silently change its controls.
    """
    path = cache / f"recalls_{start.isoformat()}_{end.isoformat()}.json"
    if path.exists() and path.stat().st_size > 0:
        return json.loads(path.read_text(encoding="utf-8"))

    import urllib.request

    rows: list[dict] = []
    skip = 0
    total = None
    for _ in range(max_pages):
        # Built by hand, not urlencode: openFDA's range parser needs a literal
        # `+` between endpoints, and percent-encoding it (`%2B`) is rejected
        # with a 500 parse error.
        url = (
            f"{RECALL_API}?search=report_date:[{start:%Y%m%d}+TO+{end:%Y%m%d}]"
            f"&limit={page}&skip={skip}"
        )
        request = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
        total = (payload.get("meta", {}) or {}).get("results", {}).get("total", total)
        batch = payload.get("results", [])
        if not batch:
            break
        rows.extend(batch)
        if total is not None and len(rows) >= total:
            break
        skip += page

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


# --- the drug universe ------------------------------------------------------


INJECTABLE_ROUTES = ("INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS", "INTRA-ARTERIAL")


@dataclass
class DrugNode:
    key: str
    labelers: set[str]
    ndcs: set[str]
    route: str = ""

    @property
    def sole_source(self) -> bool:
        return len(self.labelers) == 1


def build_universe(products: list[NdcProduct], snapshot: dt.date) -> dict[str, DrugNode]:
    """Every drug marketed on `snapshot`, keyed exactly as the graph keys it."""
    universe: dict[str, DrugNode] = {}
    for product in products:
        if product.marketing_start and product.marketing_start > snapshot:
            continue
        if product.marketing_end and product.marketing_end < snapshot:
            continue
        key = drug_key(product.ingredient_set, product.dosage_form, product.route)
        node = universe.setdefault(
            key, DrugNode(key=key, labelers=set(), ndcs=set(), route=product.route)
        )
        node.labelers.add(product.labeler_code)
        if product.ndc:
            node.ndcs.add(ndc_key(product.ndc))
    return universe


def map_cases(
    records: list[dict], universe: dict[str, DrugNode]
) -> tuple[set[str], dict]:
    """Map shortage records onto universe drug keys, measuring how well it works.

    Primary route is the record's own NDC codes. Only records that fail that
    fall back to ingredient-set matching, and the split is reported, because a
    low NDC match rate would mean the cases are not the same objects as the
    controls.
    """
    by_ndc: dict[str, str] = {}
    for key, node in universe.items():
        for ndc in node.ndcs:
            by_ndc[ndc] = key

    by_ingredients: dict[tuple[frozenset[str], str], list[str]] = collections.defaultdict(list)
    for key in universe:
        parts = (key.split("|") + ["", ""])[:3]
        ingredients = frozenset(parts[0].split("+"))
        by_ingredients[(ingredients, parts[2])].append(key)

    cases: set[str] = set()
    stats = collections.Counter()
    for record in records:
        openfda = record.get("openfda") or {}
        mapped = False
        for code in openfda.get("product_ndc") or []:
            key = by_ndc.get(ndc_key(code))
            if key:
                cases.add(key)
                mapped = True
        if mapped:
            stats["via_ndc"] += 1
            continue

        substances = frozenset(
            s.strip().upper() for s in (openfda.get("substance_name") or []) if s.strip()
        )
        route = ((openfda.get("route") or [""])[0] or "").strip().upper()
        if not route:
            form = (record.get("dosage_form") or "").strip().upper()
            route = FORM_TO_ROUTE.get(form, "")
        matched = by_ingredients.get((substances, route), []) if substances else []
        if matched:
            cases.update(matched)
            stats["via_ingredients"] += 1
        else:
            stats["unmapped"] += 1

    stats["records"] = len(records)
    stats["distinct_cases"] = len(cases)
    return cases, dict(stats)


# --- recall exposure --------------------------------------------------------


def recall_exposure(
    rows: list[dict],
    universe: dict[str, DrugNode],
    *,
    window_start: dt.date,
    window_end: dt.date,
) -> tuple[set[str], set[str], dict]:
    """Drug keys with a recall in the window, by NDC link and by labeler name.

    Two routes are kept separate rather than merged: the NDC link is precise
    but only present for some reports; the firm-name link is broad but
    approximate. Reporting both shows how much each contributes.
    """
    by_ndc: dict[str, str] = {}
    for key, node in universe.items():
        for ndc in node.ndcs:
            by_ndc[ndc] = key

    labeler_to_keys: dict[str, set[str]] = collections.defaultdict(set)
    for key, node in universe.items():
        for labeler in node.labelers:
            labeler_to_keys[normalise_firm(labeler)].add(key)

    exposed_ndc: set[str] = set()
    exposed_firm: set[str] = set()
    stats = collections.Counter()
    for row in rows:
        report_date = parse_date(row.get("report_date"))
        if report_date is None or not (window_start <= report_date <= window_end):
            stats["outside_window"] += 1
            continue
        stats["in_window"] += 1
        codes = (row.get("openfda") or {}).get("product_ndc") or []
        hit = False
        for code in codes:
            key = by_ndc.get(ndc_key(code))
            if key:
                exposed_ndc.add(key)
                hit = True
        if hit:
            stats["linked_by_ndc"] += 1
        firm = normalise_firm(row.get("recalling_firm") or "")
        for key in labeler_to_keys.get(firm, ()):  # exact normalised-name match
            exposed_firm.add(key)
            stats["linked_by_firm"] += 1
    return exposed_ndc, exposed_firm, dict(stats)


# --- statistics -------------------------------------------------------------


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for a 2x2 table, computed from the hypergeometric."""
    n = a + b + c + d
    row1 = a + b
    col1 = a + c
    if n == 0:
        return 1.0

    def prob(x: int) -> float:
        return (
            math.comb(row1, x) * math.comb(n - row1, col1 - x) / math.comb(n, col1)
        )

    observed = prob(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    total = 0.0
    for x in range(lo, hi + 1):
        p = prob(x)
        if p <= observed * (1 + 1e-9):
            total += p
    return min(1.0, total)


def odds_ratio(a: int, b: int, c: int, d: int) -> dict:
    """Odds ratio with a Katz (log-normal) 95% CI and Fisher's exact p.

    A 0.5 continuity correction is applied only when a cell is zero, so tables
    without zero cells are reported uncorrected.
    """
    zero = 0 in (a, b, c, d)
    aa, bb, cc, dd = (x + 0.5 for x in (a, b, c, d)) if zero else (a, b, c, d)
    or_ = (aa * dd) / (bb * cc)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo = math.exp(math.log(or_) - 1.96 * se)
    hi = math.exp(math.log(or_) + 1.96 * se)
    return {
        "table": {"exposed_case": a, "unexposed_case": b, "exposed_control": c, "unexposed_control": d},
        "odds_ratio": round(or_, 3),
        "ci95": [round(lo, 3), round(hi, 3)],
        "fisher_p": float(f"{fisher_two_sided(a, b, c, d):.6g}"),
        "continuity_corrected": zero,
    }


def mantel_haenszel(strata: list[tuple[int, int, int, int]]) -> dict:
    """Pooled odds ratio across strata (e.g. sole-source vs not)."""
    num = den = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n
        den += b * c / n
    if den == 0:
        return {"odds_ratio": None, "note": "one stratum contributes no information"}
    return {"odds_ratio": round(num / den, 3), "strata": len(strata)}


def density_stratum(n_ndcs: int) -> str:
    """Bin a drug by how many NDC products it carries.

    This is the ascertainment variable, not a scientific one. A case is
    identified by matching a shortage record's NDC to the directory, so a drug
    with more listed products is more likely to be captured. Stratifying on it
    shows how much of an association survives once that advantage is removed.
    """
    if n_ndcs <= 1:
        return "1"
    if n_ndcs <= 3:
        return "2-3"
    if n_ndcs <= 7:
        return "4-7"
    if n_ndcs <= 20:
        return "8-20"
    return "21+"


def two_by_two(cases: set[str], controls: set[str], exposed: set[str]) -> tuple[int, int, int, int]:
    a = len(cases & exposed)
    b = len(cases - exposed)
    c = len(controls & exposed)
    d = len(controls - exposed)
    return a, b, c, d


# --- study ------------------------------------------------------------------


def run(cache: Path, *, window_months: int = 24, recall_lag_days: int = 0) -> dict:
    records = fetch_shortages(cache)
    posting_dates = [d for d in (parse_date(r.get("initial_posting_date")) for r in records) if d]
    if not posting_dates:
        raise SystemExit("no usable initial_posting_date in the shortage mirror")
    snapshot = max(posting_dates)

    ndc_dir = fetch_ndc_directory(cache)
    products = load_ndc_products(ndc_dir)
    universe = build_universe(products, snapshot)

    cases, case_stats = map_cases(records, universe)
    controls = set(universe) - cases

    window_start = snapshot - dt.timedelta(days=int(30.44 * window_months))
    rows = fetch_enforcement(cache, window_start, snapshot - dt.timedelta(days=recall_lag_days))
    exposed_ndc, exposed_firm, recall_stats = recall_exposure(
        rows, universe, window_start=window_start, window_end=snapshot - dt.timedelta(days=recall_lag_days)
    )
    exposed_any = exposed_ndc | exposed_firm

    def block(name: str, exposed: set[str]) -> dict:
        table = two_by_two(cases, controls, exposed)
        out = {"exposure": name, **odds_ratio(*table)}
        out["exposed_share_cases"] = round(len(cases & exposed) / max(1, len(cases)), 4)
        out["exposed_share_controls"] = round(len(controls & exposed) / max(1, len(controls)), 4)
        return out

    results = {
        "design": "cross-sectional prevalence case-control at a single reference date",
        "snapshot": snapshot.isoformat(),
        "snapshot_basis": "latest initial_posting_date in the openFDA shortage mirror",
        "window_months": window_months,
        "universe": {
            "products_parsed": len(products),
            "drug_nodes": len(universe),
            "cases": len(cases),
            "controls": len(controls),
        },
        "case_mapping": case_stats,
        "recall_linking": recall_stats,
        "concentration": {
            "sole_source_cases_pct": round(
                100 * sum(1 for k in cases if universe[k].sole_source) / max(1, len(cases)), 1
            ),
            "sole_source_controls_pct": round(
                100 * sum(1 for k in controls if universe[k].sole_source) / max(1, len(controls)), 1
            ),
            "median_labelers_cases": _median([len(universe[k].labelers) for k in cases]),
            "median_labelers_controls": _median([len(universe[k].labelers) for k in controls]),
        },
        "associations": {
            "sole_source": block("sole-source (single labeler)", {k for k in universe if universe[k].sole_source}),
            "recall_ndc_link": block("recall linked by NDC", exposed_ndc),
            "recall_firm_link": block("recall linked by firm name", exposed_firm),
            "recall_any": block("recall, either link", exposed_any),
        },
    }

    # Recall exposure, stratified by concentration: does the association
    # survive among sole-source drugs, or is it just concentration in disguise?
    strata = []
    detail = {}
    for label, wanted in (("sole_source", True), ("multi_labeler", False)):
        keys = {k for k in universe if universe[k].sole_source is wanted}
        sub_cases = cases & keys
        sub_controls = controls & keys
        table = two_by_two(sub_cases, sub_controls, exposed_any)
        detail[label] = {
            "cases": len(sub_cases),
            "controls": len(sub_controls),
            **odds_ratio(*table),
        }
        strata.append(table)
    results["recall_any_stratified"] = {"strata": detail, "mantel_haenszel": mantel_haenszel(strata)}

    # Ascertainment control: cases enter through NDC linkage, so a drug with
    # more listed products is easier to capture and easier to expose. Pooling
    # within product-count strata removes that advantage.
    density: dict[str, dict] = {}
    density_strata = []
    for stratum in ("1", "2-3", "4-7", "8-20", "21+"):
        keys = {k for k in universe if density_stratum(len(universe[k].ndcs)) == stratum}
        sub_cases = cases & keys
        sub_controls = controls & keys
        if not sub_cases or not sub_controls:
            continue
        table = two_by_two(sub_cases, sub_controls, exposed_ndc)
        density[stratum] = {
            "cases": len(sub_cases),
            "controls": len(sub_controls),
            **odds_ratio(*table),
        }
        density_strata.append(table)
    results["ndc_density_stratified"] = {
        "exposure": "recall linked by NDC",
        "strata": density,
        "mantel_haenszel": mantel_haenszel(density_strata),
    }

    # Injectable subgroup: the products whose loss most disrupts acute care, and
    # the class where recall exposure should matter most if the mechanism is
    # manufacturing disruption.
    injectable = {
        k for k in universe if any(tok in universe[k].route for tok in INJECTABLE_ROUTES)
    }
    inj_cases = cases & injectable
    inj_controls = controls & injectable
    results["injectable_subgroup"] = {
        "cases": len(inj_cases),
        "controls": len(inj_controls),
        "sole_source_pct_cases": round(
            100 * sum(1 for k in inj_cases if universe[k].sole_source) / max(1, len(inj_cases)), 1
        ),
        "recall_ndc": {
            **odds_ratio(*two_by_two(inj_cases, inj_controls, exposed_ndc)),
            "exposed_share_cases": round(len(inj_cases & exposed_ndc) / max(1, len(inj_cases)), 4),
            "exposed_share_controls": round(
                len(inj_controls & exposed_ndc) / max(1, len(inj_controls)), 4
            ),
        },
        "recall_any": {
            **odds_ratio(*two_by_two(inj_cases, inj_controls, exposed_any)),
            "exposed_share_cases": round(len(inj_cases & exposed_any) / max(1, len(inj_cases)), 4),
            "exposed_share_controls": round(
                len(inj_controls & exposed_any) / max(1, len(inj_controls)), 4
            ),
        },
    }

    return results


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2


# --- reporting --------------------------------------------------------------


def to_markdown(results: dict) -> str:
    u = results["universe"]
    lines = [
        "# Shortage snapshot case-control",
        "",
        f"Reference date: **{results['snapshot']}** ({results['snapshot_basis']})",
        f"Universe: {u['drug_nodes']:,} drug nodes from {u['products_parsed']:,} NDC products "
        f"-> {u['cases']} cases, {u['controls']:,} controls",
        f"Recall window: {results['window_months']} months before the reference date",
        "",
        "| Exposure | OR | 95% CI | Fisher p | cases exposed | controls exposed |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in results["associations"].items():
        lo, hi = row["ci95"]
        lines.append(
            f"| {row['exposure']} | {row['odds_ratio']} | {lo}-{hi} | {row['fisher_p']} "
            f"| {100 * row['exposed_share_cases']:.1f}% | {100 * row['exposed_share_controls']:.1f}% |"
        )
    strat = results["recall_any_stratified"]
    lines += ["", "## Recall exposure stratified by concentration", "", "| Stratum | cases | controls | OR | 95% CI | p |", "|---|---|---|---|---|---|"]
    for name, row in strat["strata"].items():
        lo, hi = row["ci95"]
        lines.append(
            f"| {name} | {row['cases']} | {row['controls']:,} | {row['odds_ratio']} | {lo}-{hi} | {row['fisher_p']} |"
        )
    mh = strat["mantel_haenszel"]
    lines += ["", f"Mantel-Haenszel pooled OR: {mh.get('odds_ratio')}"]

    dens = results["ndc_density_stratified"]
    lines += [
        "",
        "## Recall (NDC-linked) stratified by product count",
        "",
        "| NDC products | cases | controls | OR | 95% CI | p |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in dens["strata"].items():
        lo, hi = row["ci95"]
        lines.append(
            f"| {name} | {row['cases']} | {row['controls']:,} | {row['odds_ratio']} | {lo}-{hi} | {row['fisher_p']} |"
        )
    lines += ["", f"Mantel-Haenszel pooled OR: {dens['mantel_haenszel'].get('odds_ratio')}"]

    inj = results["injectable_subgroup"]
    lines += [
        "",
        f"## Injectable subgroup ({inj['cases']} cases, {inj['controls']:,} controls)",
        "",
        "| Exposure | OR | 95% CI | Fisher p | cases exposed | controls exposed |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("recall_ndc", "recall_any"):
        row = inj[name]
        lo, hi = row["ci95"]
        lines.append(
            f"| {name} | {row['odds_ratio']} | {lo}-{hi} | {row['fisher_p']} "
            f"| {100 * row['exposed_share_cases']:.1f}% | {100 * row['exposed_share_controls']:.1f}% |"
        )
    lines += [
        "",
        "## Measured concentration",
        "",
        f"- sole-source share, cases: {results['concentration']['sole_source_cases_pct']}%",
        f"- sole-source share, controls: {results['concentration']['sole_source_controls_pct']}%",
        f"- median labelers per drug, cases: {results['concentration']['median_labelers_cases']}, "
        f"controls: {results['concentration']['median_labelers_controls']}",
        "",
        "Interpretation limit: a single cross-section estimates association at one date, "
        "not incidence, causation, or forecast skill.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shortage snapshot case-control study")
    parser.add_argument("--cache", default="data/raw")
    parser.add_argument("--window-months", type=int, default=24)
    parser.add_argument("--out", default="data/runs/casecontrol/results.json")
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    results = run(cache, window_months=args.window_months)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(to_markdown(results), encoding="utf-8")

    print(to_markdown(results))
    print(f"\nwrote {out} and {md}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
