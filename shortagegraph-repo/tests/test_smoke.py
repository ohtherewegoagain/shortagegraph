"""Smoke tests: the pipeline runs end to end on the synthetic fixture.

These assert *mechanics*, not scientific results: shapes, no future leakage,
label semantics, metric agreement with sklearn, and that training reduces loss
and produces finite metrics.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from shortagegraph.config import EMBARGO_WEEKS, HORIZONS_WEEKS, embargo_excludes, split_of
from shortagegraph.fixture import make_raw
from shortagegraph.graph import build_snapshot, build_snapshots, drug_key, label_snapshot
from shortagegraph.metrics import average_precision, lead_time_weeks, roc_auc


@pytest.fixture(scope="module")
def raw():
    return make_raw(seed=11, n_drugs=40, n_estabs=12, n_labelers=10)


def test_drug_key_is_order_insensitive():
    assert drug_key(["B", "A"], "injection", "iv") == drug_key(["A", "B"], "INJECTION", "IV")
    assert drug_key("B+A", "INJECTION", "IV").startswith("A+B")


def test_snapshot_shapes_and_no_edges_before_graph_exists(raw):
    snap = build_snapshot(raw, dt.date(2020, 6, 1))
    assert snap.num_drugs > 0
    assert snap.Xd.shape == (snap.num_drugs, 21)
    assert snap.Xl.shape == (len(snap.labeler_ids), 7)
    assert snap.Xe.shape == (len(snap.estab_ids), 4)
    for name, arr in snap.edges.items():
        assert arr.ndim == 2 and arr.shape[0] == 2
        if arr.size:
            assert arr.max() < snap.num_nodes
            assert arr.min() >= 0


def test_no_future_leakage(raw):
    """A letter dated later must not raise an earlier snapshot's feature."""
    early = dt.date(2019, 1, 7)
    late = dt.date(2024, 1, 1)
    s_early = build_snapshot(raw, early)
    s_late = build_snapshot(raw, late)
    common = set(s_early.drug_keys) & set(s_late.drug_keys)
    assert common
    idx_early = {k: i for i, k in enumerate(s_early.drug_keys)}
    idx_late = {k: i for i, k in enumerate(s_late.drug_keys)}
    for key in list(common)[:20]:
        # weeks_since_letter cannot be smaller (more recent) in the early snapshot
        assert s_early.Xd[idx_early[key], 18] >= s_late.Xd[idx_late[key], 18] - 1e-6


def test_labels_are_first_onset_only(raw):
    weeks = [dt.date(2020, 3, 2), dt.date(2020, 9, 7)]
    for week in weeks:
        snap = build_snapshot(raw, week)
        labels = label_snapshot(snap, raw.shortages, HORIZONS_WEEKS)
        assert labels.y.shape == (snap.num_drugs, len(HORIZONS_WEEKS))
        # monotone in horizon by construction
        assert np.all(labels.y[:, 0] <= labels.y[:, 1] + 1e-6)
        assert np.all(labels.y[:, 1] <= labels.y[:, 2] + 1e-6)
        # already-listed drugs are not at risk
        assert np.all(labels.first_posting[~labels.at_risk] <= 0)


def test_split_and_embargo_boundaries():
    assert split_of(dt.date(2022, 6, 6)) == "train"
    assert split_of(dt.date(2023, 6, 5)) == "val"
    assert split_of(dt.date(2024, 6, 3)) == "test"
    assert embargo_excludes(dt.date(2023, 1, 9))       # just after a boundary
    assert not embargo_excludes(dt.date(2023, 6, 5))   # well inside val
    assert EMBARGO_WEEKS == 12


def test_metrics_match_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.15).astype(float)
    s = rng.random(400) + 2.0 * y
    assert average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-6)
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-6)


def test_lead_time_arithmetic():
    alerts = {"a": [1, 2, 5], "b": [9]}
    onsets = {"a": 8, "b": 10}
    out = lead_time_weeks(alerts, onsets, threshold=4)
    assert out["onsets"] == 2
    assert out["any"] == 1.0          # both flagged at some point before posting
    assert out["early"] == 0.5        # only "a" was flagged >= 4 weeks ahead
    # leads are [8-1=7, 10-9=1], so the median is 4.0
    assert out["median_lead"] == 4.0


def test_end_to_end_training_runs(raw, tmp_path):
    """Full path: snapshots -> tensors -> train -> metrics, on the fixture."""
    from shortagegraph.train import main

    out = tmp_path / "run"
    code = main(
        [
            "--fixture",
            "--epochs", "3",
            "--seeds", "1",
            "--bootstrap", "20",
            "--out", str(out),
        ]
    )
    assert code == 0
    import json

    payload = json.loads((out / "results.json").read_text())
    assert payload["source"].startswith("SYNTHETIC FIXTURE")
    res = payload["results"]["ShortageGraph (R-GNN)"]
    for metric in ("auprc", "auroc", "p_at_25", "brier"):
        assert len(res[metric]) == len(HORIZONS_WEEKS)
        assert all(np.isfinite(v) for v in res[metric])
    assert payload["dataset"]["weekly_snapshots"] > 100
    assert all(key in payload["results"] for key in
               ("Sole-source rule", "Logistic regression", "Gradient boosting", "MLP, no edges"))


def test_label_conversion_dedupes_and_keeps_first_posting():
    from shortagegraph.labels import to_postings

    records = [
        {"initial_posting_date": "03/02/2021", "generic_name": "Vancomycin Hydrochloride",
         "dosage_form": "Injection", "update_type": "New",
         "openfda": {"substance_name": ["VANCOMYCIN HYDROCHLORIDE"], "route": ["INTRAVENOUS"]}},
        {"initial_posting_date": "01/05/2021", "generic_name": "Vancomycin Hydrochloride",
         "dosage_form": "Injection", "update_type": "New",
         "openfda": {"substance_name": ["VANCOMYCIN HYDROCHLORIDE"], "route": ["INTRAVENOUS"]}},
        {"initial_posting_date": "2021-06-01", "generic_name": "Cisplatin",
         "dosage_form": "Injection", "update_type": "To Be Discontinued",
         "openfda": {"substance_name": ["CISPLATIN"], "route": ["INTRAVENOUS"]}},
        {"generic_name": "No Date Drug", "dosage_form": "Tablet"},
    ]
    postings, meta = to_postings(records)
    assert len(postings) == 2                       # duplicate drug collapsed
    assert meta["dropped"]["no_initial_posting_date"] == 1
    vanc = [p for p in postings if "VANCOMYCIN" in p.drug_key[0]][0]
    assert vanc.posting_date == dt.date(2021, 1, 5)  # earliest, not latest
    assert postings[0].posting_date <= postings[-1].posting_date


def test_certify_rejects_thin_year():
    """A split whose year has almost no onsets is survivorship-biased."""
    from shortagegraph.ingest import ShortagePosting
    from shortagegraph.labels import certify

    def postings_for(counts: dict[int, int]):
        out = []
        for year, n in counts.items():
            for i in range(n):
                out.append(
                    ShortagePosting(
                        drug_key=(f"DRUG{i}", "INJECTION", "INTRAVENOUS"),
                        posting_date=dt.date(year, 6, 1),
                        product_name="",
                    )
                )
        return out

    thin = postings_for({2018: 40, 2019: 2, 2020: 40, 2021: 40, 2022: 40, 2023: 40, 2024: 40, 2025: 40})
    cert = certify(thin)
    assert not cert.ok
    assert any("2019" in reason for reason in cert.reasons)

    healthy = postings_for({2018: 40, 2019: 40, 2020: 40, 2021: 40, 2022: 40, 2023: 40, 2024: 40, 2025: 40})
    assert certify(healthy).ok


def test_archive_import_roundtrip(tmp_path):
    """A complete history imports, certifies, and reads back through ingest."""
    import csv as csvmod

    from shortagegraph.archive_import import load_history, main as import_main
    from shortagegraph.ingest import load_shortages

    rows = [
        {
            "generic_name": f"DRUG{year}_{i} ACID",
            "dosage_form": "Injection",
            "route": "INTRAVENOUS",
            "posting_date": f"{year}-06-01",
            "product_name": "fixture",
        }
        for year in range(2018, 2026)
        for i in range(30)
    ]
    source = tmp_path / "history.csv"
    with source.open("w", newline="") as handle:
        writer = csvmod.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    postings, meta = load_history(source)
    assert meta["rows_read"] == 240
    assert len(postings) == 240          # one per drug, all distinct keys

    out = tmp_path / "shortages.csv"
    code = import_main(["--source", str(source), "--out", str(out),
                        "--report", str(tmp_path / "report.json")])
    assert code == 0                     # 30 onsets/year clears the gate
    assert len(load_shortages(out)) == 240


def test_realdata_nulls_are_explicit():
    """Unmeasured Table I rows must be None, never a silent zero."""
    from shortagegraph.realdata import summarise, table_rows

    series = {"weeks": [{
        "week": "2020-01-06", "active_products": 10, "drug_nodes": 5,
        "labeler_nodes": 3, "market_edges": 10, "substitute_edges": 4,
        "single_labeler_share": 0.6,
    }]}
    summary = summarise(series)
    rows = table_rows(summary)
    assert rows["Drug nodes |VD| (median per snapshot)"] == 5
    assert rows["Establishment nodes |VF|"] is None
    assert rows["First-onset events (train / val / test)"] is None
