"""Train and evaluate under the strict temporal split (Section IV).

Produces:
  * dataset statistics for Table I
  * per-model, per-horizon metrics for Table II, with a bootstrap CI by drug
  * the weekly top-25 review list and lead-time statistics

Run against the synthetic fixture (pipeline test, never paper numbers):

    python -m shortagegraph.train --fixture --out data/runs/fixture

Run against real FDA data placed under data/raw (see README):

    python -m shortagegraph.train --from-raw data/raw --out data/runs/real
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .baselines import GradientBoostingBaseline, LogisticBaseline, sole_source_scores
from .config import EMBARGO_WEEKS, HORIZONS_WEEKS, REVIEW_LIST_SIZE, SPLITS, Paths
from .fixture import make_raw, write_fixture
from .graph import RELATIONS, RawData, Snapshot, build_snapshots, label_snapshot
from .metrics import (
    average_precision,
    bootstrap_ci_by_group,
    brier_score,
    lead_time_weeks,
    precision_at_k,
    roc_auc,
)
from .model import EdgeFreeMLP, ShortageGraphNet, positive_weight, weighted_bce

MODEL_ORDER = [
    "Prevalence (random)",
    "Sole-source rule",
    "Logistic regression",
    "Gradient boosting",
    "MLP, no edges",
    "ShortageGraph (R-GNN)",
]


# --- tensor assembly --------------------------------------------------------


def edge_tensors(snapshot: Snapshot) -> tuple[torch.Tensor, torch.Tensor]:
    indices, types = [], []
    for r, name in enumerate(RELATIONS):
        arr = snapshot.edges.get(name)
        if arr is None or arr.size == 0:
            continue
        indices.append(arr)
        types.append(np.full(arr.shape[1], r, dtype=np.int64))
    if not indices:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.long)
    return (
        torch.from_numpy(np.concatenate(indices, axis=1)),
        torch.from_numpy(np.concatenate(types)),
    )


def snapshot_tensors(snapshot: Snapshot) -> dict:
    edge_index, edge_type = edge_tensors(snapshot)
    return {
        "Xd": torch.from_numpy(snapshot.Xd),
        "Xl": torch.from_numpy(snapshot.Xl),
        "Xe": torch.from_numpy(snapshot.Xe),
        "edge_index": edge_index,
        "edge_type": edge_type,
    }


def build_dataset(snapshots: list[Snapshot], raw: RawData) -> list[dict]:
    rows = []
    for snapshot in snapshots:
        labels = label_snapshot(snapshot, raw.shortages, HORIZONS_WEEKS)
        if labels.at_risk.sum() == 0:
            continue
        row = snapshot_tensors(snapshot)
        row.update(
            {
                "snapshot": snapshot,
                "y": torch.from_numpy(labels.y),
                "mask": torch.from_numpy(labels.at_risk),
                "first_posting": labels.first_posting,
                "week": snapshot.week,
                "split": snapshot.split,
                "embargoed": snapshot.embargoed,
            }
        )
        rows.append(row)
    return rows


# --- training ---------------------------------------------------------------


def train_one(model, rows: list[dict], val_rows: list[dict], *, epochs: int, lr: float,
              seed: int, patience: int = 6) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    alpha = positive_weight(np.concatenate([r["y"].numpy()[r["mask"].numpy()] for r in rows]))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state, best_score, bad_epochs = None, -np.inf, 0

    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(rows))
        for idx in order:
            row = rows[idx]
            optimiser.zero_grad()
            pred = model(row["Xd"], row["Xl"], row["Xe"], row["edge_index"], row["edge_type"])
            loss = weighted_bce(pred, row["y"], row["mask"], alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

        score = evaluate_rows(model, val_rows)["auprc"][1] if val_rows else -np.inf
        if score > best_score + 1e-5:
            best_score, bad_epochs = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def predict_rows(model, rows: list[dict]) -> list[np.ndarray]:
    model.eval()
    out = []
    for row in rows:
        pred = model(row["Xd"], row["Xl"], row["Xe"], row["edge_index"], row["edge_type"])
        out.append(pred.numpy())
    return out


def evaluate_rows(model, rows: list[dict]) -> dict[str, list[float]]:
    preds = predict_rows(model, rows)
    return pooled_metrics(preds, rows)


def pooled_metrics(preds: list[np.ndarray], rows: list[dict]) -> dict[str, list[float]]:
    """Pool at-risk drug-weeks across snapshots, per horizon."""
    out = {"auprc": [], "auroc": [], "p_at_25": [], "brier": []}
    for h in range(len(HORIZONS_WEEKS)):
        y_all, s_all = [], []
        for pred, row in zip(preds, rows):
            mask = row["mask"].numpy()
            if mask.sum() == 0:
                continue
            y_all.append(row["y"].numpy()[mask, h])
            s_all.append(pred[mask] if pred.ndim == 1 else pred[mask, h])
        if not y_all:
            out["auprc"].append(float("nan"))
            out["auroc"].append(float("nan"))
            out["p_at_25"].append(float("nan"))
            out["brier"].append(float("nan"))
            continue
        y = np.concatenate(y_all)
        s = np.concatenate(s_all)
        out["auprc"].append(average_precision(y, s))
        out["auroc"].append(roc_auc(y, s))
        out["brier"].append(brier_score(y, s))
        per_week = []
        for pred, row in zip(preds, rows):
            mask = row["mask"].numpy()
            if mask.sum() == 0:
                continue
            scores = pred[mask] if pred.ndim == 1 else pred[mask, h]
            per_week.append(precision_at_k(row["y"].numpy()[mask, h], scores, REVIEW_LIST_SIZE))
        out["p_at_25"].append(float(np.nanmean(per_week)))
    return out


# --- dataset statistics (Table I) ------------------------------------------


def dataset_stats(snapshots: list[Snapshot], raw: RawData) -> dict:
    kept = [s for s in snapshots if not s.embargoed]
    drug_counts = [s.num_drugs for s in kept]
    single = [
        float(np.mean(s.Xd[:, 0] == 1)) for s in kept if s.num_drugs
    ]
    edge_counts = [sum(s.edges[name].shape[1] for name in RELATIONS) for s in kept]
    onsets = {"train": 0, "val": 0, "test": 0}
    for snapshot in kept:
        labels = label_snapshot(snapshot, raw.shortages, HORIZONS_WEEKS)
        hits = int((labels.y[:, -1] > 0).sum())
        events = 0
        for i, first in enumerate(labels.first_posting):
            if 0 < first <= 12:
                events += 1
        onsets[snapshot.split] = onsets.get(snapshot.split, 0) + events
    return {
        "weekly_snapshots": len(kept),
        "drug_nodes_median": int(np.median(drug_counts)) if drug_counts else 0,
        "labeler_nodes_median": int(np.median([len(s.labeler_ids) for s in kept])) if kept else 0,
        "establishment_nodes_median": int(np.median([len(s.estab_ids) for s in kept])) if kept else 0,
        "edges_median": int(np.median(edge_counts)) if edge_counts else 0,
        "single_labeler_pct": round(100 * float(np.mean(single)), 1) if single else 0.0,
        "recalls_linked": len(raw.recalls),
        "letters_linked": len([l for l in raw.letters if l.fei]),
        "first_onset_events": onsets,
        "study_start": str(SPLITS[0].start),
        "study_end": str(SPLITS[-1].end),
        "embargo_weeks": EMBARGO_WEEKS,
    }


# --- main -------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")
    if args.fixture:
        raw = make_raw()
        write_fixture(raw, Paths().ensure().fixtures / "synthetic.json")
        source = "SYNTHETIC FIXTURE (pipeline test only - not paper results)"
    elif args.synthetic:
        from .synthesize import load as load_synthetic

        raw = load_synthetic(Path(args.synthetic))
        source = (
            f"CALIBRATED SYNTHETIC ({args.synthetic}) - generated from measured "
            "public-data marginals; a smoke test, not paper results"
        )
    else:
        from .ingest import load_raw_from_dir  # local import: only needed for real runs

        raw = load_raw_from_dir(Path(args.from_raw))
        source = f"FDA sources under {args.from_raw}"

        # A survivorship-biased history produces plausible-looking numbers from
        # corrupted negatives, so this refuses rather than warns.
        from .labels import certify

        cert = certify(raw.shortages)
        if not cert.ok and not args.force_uncertified_labels:
            raise SystemExit(
                "label source failed certification; refusing to train on an "
                "incomplete shortage history:\n  - "
                + "\n  - ".join(cert.reasons)
                + "\n(override with --force-uncertified-labels for debugging only)"
            )

    snapshots = build_snapshots(raw, SPLITS[0].start, SPLITS[-1].end)
    rows = build_dataset(snapshots, raw)
    train = [r for r in rows if r["split"] == "train" and not r["embargoed"]]
    val = [r for r in rows if r["split"] == "val" and not r["embargoed"]]
    test = [r for r in rows if r["split"] == "test" and not r["embargoed"]]
    if not train or not test:
        raise SystemExit("empty train or test split: check the study window and embargo")

    drug_dim = train[0]["Xd"].shape[1]
    labeler_dim = train[0]["Xl"].shape[1]
    estab_dim = train[0]["Xe"].shape[1]

    results: dict[str, dict] = {}
    per_seed: dict[str, list[dict]] = {name: [] for name in MODEL_ORDER}

    # rule and tabular baselines on pooled drug-weeks
    X_train = np.concatenate([r["Xd"].numpy()[r["mask"].numpy()] for r in train])
    y_train = np.concatenate([r["y"].numpy()[r["mask"].numpy()] for r in train])
    X_test = np.concatenate([r["Xd"].numpy()[r["mask"].numpy()] for r in test])
    y_test = np.concatenate([r["y"].numpy()[r["mask"].numpy()] for r in test])

    n_h = len(HORIZONS_WEEKS)
    prevalence = {h: float(y_test[:, h].mean()) for h in range(n_h)}
    rule_preds = [sole_source_scores(r["Xd"].numpy()) for r in test]
    per_seed["Sole-source rule"].append(pooled_metrics(rule_preds, test))

    # Each column comes from the model fitted for that horizon. Filling a
    # column from another horizon's fit would make the table mean nothing.
    lr_columns: dict[int, list[np.ndarray]] = {}
    gb_columns: dict[int, list[np.ndarray]] = {}
    for h in range(n_h):
        lr_model = LogisticBaseline(horizon=h).fit(X_train, y_train)
        gb_model = GradientBoostingBaseline(horizon=h).fit(X_train, y_train)
        lr_columns[h] = [lr_model.predict(r["Xd"].numpy()) for r in test]
        gb_columns[h] = [gb_model.predict(r["Xd"].numpy()) for r in test]

    def stack_columns(columns: dict[int, list[np.ndarray]]) -> list[np.ndarray]:
        return [np.stack([columns[h][i] for h in range(n_h)], axis=1) for i in range(len(test))]

    per_seed["Logistic regression"].append(pooled_metrics(stack_columns(lr_columns), test))
    per_seed["Gradient boosting"].append(pooled_metrics(stack_columns(gb_columns), test))

    # neural models, one run per seed
    for seed in range(args.seeds):
        gnn = ShortageGraphNet(
            drug_dim, labeler_dim, estab_dim,
            dim=args.dim, layers=args.layers, dropout=args.dropout,
        )
        train_one(gnn, train, val, epochs=args.epochs, lr=args.lr, seed=seed)
        per_seed["ShortageGraph (R-GNN)"].append(evaluate_rows(gnn, test))

        mlp = EdgeFreeMLP(drug_dim, hidden=args.hidden, dropout=args.dropout)
        train_one(mlp, train, val, epochs=args.epochs, lr=args.lr, seed=seed)
        per_seed["MLP, no edges"].append(evaluate_rows(mlp, test))

    # average across seeds (and across the 3 horizon-specific baseline fits)
    for name, runs in per_seed.items():
        if not runs:
            continue
        results[name] = {
            metric: [float(np.nanmean([run[metric][h] for run in runs])) for h in range(len(HORIZONS_WEEKS))]
            for metric in ("auprc", "auroc", "p_at_25", "brier")
        }
    results["Prevalence (random)"] = {
        "auprc": [prevalence[h] for h in range(len(HORIZONS_WEEKS))],
        "auroc": [0.5] * len(HORIZONS_WEEKS),
        "p_at_25": [prevalence[h] for h in range(len(HORIZONS_WEEKS))],
        "brier": [float(np.mean((y_test[:, h]) ** 2)) for h in range(len(HORIZONS_WEEKS))],
    }

    # bootstrap CI at the 8-week horizon, resampling drugs
    mask = np.concatenate([r["mask"].numpy() for r in test])
    groups = np.concatenate([[r["snapshot"].drug_keys[i] for i in np.where(r["mask"].numpy())[0]] for r in test])
    gnn_preds = predict_rows(gnn, test)
    y8 = np.concatenate([r["y"].numpy()[r["mask"].numpy(), 1] for r in test])
    s8 = np.concatenate([p[row["mask"].numpy(), 1] for p, row in zip(gnn_preds, test)])
    ci = bootstrap_ci_by_group(y8, s8, groups, n_resamples=args.bootstrap)

    # weekly review list and lead time
    week_no = {snapshot.week: i for i, snapshot in enumerate(snapshots)}
    alert_weeks: dict[str, list[int]] = {}
    onset_weeks: dict[str, int] = {}
    for row, pred in zip(test, gnn_preds):
        keys = row["snapshot"].drug_keys
        mask_row = row["mask"].numpy()
        scores = pred[:, 1].copy()
        scores[~mask_row] = -np.inf
        top = np.argsort(-scores)[:REVIEW_LIST_SIZE]
        for idx in top:
            alert_weeks.setdefault(keys[idx], []).append(week_no[row["snapshot"].week])
        first = row["first_posting"]
        for i, delta in enumerate(first):
            if 0 < delta <= HORIZONS_WEEKS[-1]:
                onset_weeks.setdefault(keys[i], week_no[row["snapshot"].week] + int(delta))
    lead = lead_time_weeks(alert_weeks, onset_weeks, threshold=4)

    payload = {
        "source": source,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_stats(snapshots, raw),
        "prevalence": prevalence,
        "results": results,
        "bootstrap_ci_8w": {"point": ci.point, "low": ci.low, "high": ci.high,
                            "n_resamples": ci.n_resamples, "grouping": "drug"},
        "lead_time": lead,
        "config": {
            "epochs": args.epochs, "lr": args.lr, "dim": args.dim, "layers": args.layers,
            "dropout": args.dropout, "seeds": args.seeds, "horizons": list(HORIZONS_WEEKS),
        },
    }
    return payload


def format_markdown(payload: dict) -> str:
    horizons = HORIZONS_WEEKS
    lines = [
        f"Source: {payload['source']}",
        "",
        "| Model | " + " | ".join(f"AUPRC h={h}" for h in horizons) + " | "
        + " | ".join(f"AUROC h={h}" for h in horizons) + " |",
        "|---" * (1 + 2 * len(horizons)) + "|",
    ]
    for name in MODEL_ORDER:
        if name not in payload["results"]:
            continue
        res = payload["results"][name]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{res['auprc'][h]:.3f}" for h in range(len(horizons))) + " | "
            + " | ".join(f"{res['auroc'][h]:.3f}" for h in range(len(horizons))) + " |"
        )
    ci = payload["bootstrap_ci_8w"]
    lines += [
        "",
        f"8-week AUPRC 95% CI (drug-level bootstrap): {ci['point']:.3f} [{ci['low']:.3f}-{ci['high']:.3f}]",
        f"Lead time: {payload['lead_time']}",
        f"Dataset: {json.dumps(payload['dataset'])}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate ShortageGraph")
    parser.add_argument("--fixture", action="store_true", help="use the synthetic fixture")
    parser.add_argument("--from-raw", type=str, default=None, help="directory of prepared FDA inputs")
    parser.add_argument("--synthetic", type=str, default=None,
                        help="path to a calibrated synthetic graph (synthesize.py output)")
    parser.add_argument("--force-uncertified-labels", action="store_true",
                        help="train even if the shortage history fails the completeness audit (debugging only)")
    parser.add_argument("--out", type=str, default="data/runs/latest")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args(argv)

    if not args.fixture and not args.from_raw and not args.synthetic:
        parser.error("pass --fixture, --synthetic PATH, or --from-raw DIR")

    payload = run(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = format_markdown(payload)
    (out / "results.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
