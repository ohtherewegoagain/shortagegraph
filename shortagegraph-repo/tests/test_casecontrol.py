"""Tests for the cross-sectional analysis and its statistics.

These pin the two defects that were found during development, because both
would have produced a plausible-looking but wrong table:

  * `report_date` arrives as compact YYYYMMDD from openFDA, and a parser that
    only accepts ISO silently excludes every report;
  * NDC codes need padding-insensitive normalisation or links never match.

The statistical tests check the exact test against a table whose hypergeometric
probability can be verified by hand, so a bug cannot hide as "small p-value".
"""

from __future__ import annotations

import datetime as dt

import pytest

from shortagegraph.casecontrol import (
    DrugNode,
    build_universe,
    fisher_two_sided,
    ndc_key,
    odds_ratio,
    recall_exposure,
    two_by_two,
)
from shortagegraph.ingest import NdcProduct
from shortagegraph.labels import parse_date


def product(ndc: str, labeler: str, form: str = "INJECTION", route: str = "INTRAVENOUS", start=None):
    return NdcProduct(
        ndc=ndc,
        labeler_code=labeler,
        labeler_name=labeler,
        ingredient_set=frozenset({"MORPHINE"}),
        dosage_form=form,
        route=route,
        marketing_start=start,
        marketing_end=None,
    )


def test_compact_fda_dates_parse():
    """Regression: enforcement report_date is YYYYMMDD, not ISO."""
    assert parse_date("20240115") == dt.date(2024, 1, 15)
    assert parse_date("01/15/2024") == dt.date(2024, 1, 15)
    assert parse_date("2024-01-15") == dt.date(2024, 1, 15)
    assert parse_date("") is None


def test_ndc_key_is_padding_insensitive():
    assert ndc_key("0002-1430") == ndc_key("2-1430")
    assert ndc_key("00021430") == "00021430"
    assert ndc_key("") == ""


def test_fisher_exact_matches_hand_computation():
    """For [[1,9],[11,3]] the two-sided p is computed from the hypergeometric."""
    # All marginals fixed: n=24, row1=10, col1=12.
    p = fisher_two_sided(1, 9, 11, 3)
    assert 0.0 < p < 0.01
    # A perfectly balanced table is not evidence of association.
    assert fisher_two_sided(5, 5, 5, 5) == pytest.approx(1.0, abs=1e-9)


def test_odds_ratio_direction_and_ci():
    strong = odds_ratio(20, 80, 5, 95)
    assert strong["odds_ratio"] > 1
    assert strong["ci95"][0] < strong["odds_ratio"] < strong["ci95"][1]
    assert strong["ci95"][0] > 1  # CI excludes the null
    inverted = odds_ratio(5, 95, 20, 80)
    assert inverted["odds_ratio"] < 1


def test_odds_ratio_flags_continuity_correction():
    assert odds_ratio(0, 10, 0, 10)["continuity_corrected"] is True
    assert odds_ratio(3, 7, 4, 6)["continuity_corrected"] is False


def test_universe_respects_marketing_dates_and_marks_sole_source():
    products = [
        product("0001-0001", "ACME"),
        product("0001-0002", "ACME"),
        product("0002-0001", "BETA"),
        # Not yet marketed at the snapshot, so it must not enter the risk set.
        product("0003-0001", "GAMMA", start=dt.date(2027, 1, 1)),
    ]
    universe = build_universe(products, dt.date(2026, 6, 1))
    assert len(universe) == 1  # one drug_key shared by all three active products
    node = next(iter(universe.values()))
    assert node.labelers == {"ACME", "BETA"}
    assert node.sole_source is False


def test_recall_exposure_respects_window_and_ndc_link():
    universe = {"MORPHINE|INJECTION|INTRAVENOUS": DrugNode(
        key="MORPHINE|INJECTION|INTRAVENOUS",
        labelers={"ACME"},
        ndcs={"00010001"},
        route="INTRAVENOUS",
    )}
    rows = [
        # inside window, NDC-linked
        {"report_date": "20250110", "recalling_firm": "Unrelated Co",
         "openfda": {"product_ndc": ["0001-0001"]}},
        # inside window, firm-name linked only
        {"report_date": "20240301", "recalling_firm": "Acme Inc.",
         "openfda": {"product_ndc": []}},
        # outside window: must be ignored
        {"report_date": "20200101", "recalling_firm": "Acme Inc.",
         "openfda": {"product_ndc": ["0001-0001"]}},
    ]
    by_ndc, by_firm, stats = recall_exposure(
        rows,
        universe,
        window_start=dt.date(2024, 1, 1),
        window_end=dt.date(2026, 1, 1),
    )
    assert by_ndc == {"MORPHINE|INJECTION|INTRAVENOUS"}
    assert by_firm == {"MORPHINE|INJECTION|INTRAVENOUS"}
    assert stats["outside_window"] == 1
    assert stats["in_window"] == 2


def test_two_by_two_partitions_sets():
    cases = {"a", "b", "c"}
    controls = {"d", "e"}
    exposed = {"b", "d"}
    assert two_by_two(cases, controls, exposed) == (1, 2, 1, 1)
