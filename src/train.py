"""Train and evaluate a model that predicts whether an NJ Transit train will be
5+ minutes late at a stop, using only information known from the schedule.

Run from the repo root:   python src/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# ----------------------------- settings -----------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "archive"
OUT_DIR = ROOT / "outputs"

LATE_MINUTES = 5          # a stop counts as "late" at 5+ minutes
SAMPLE_FRAC = 0.25        # share of rows kept from each monthly file (1.0 = all rows)
LAST_MONTH = "2019_12"    # stop before the 2020 COVID service cuts
VAL_START = "2019-05-01"  # validation period: used only to choose a decision threshold
TEST_START = "2019-07-01" # test period: never used for any choices
RANDOM_STATE = 42

CAT = ["line", "to_station"]
NUM = ["hour", "dayofweek", "month", "is_weekend", "is_rush"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ------------------------- stage 1: raw data -------------------------
def load_raw() -> pd.DataFrame:
    """Read the monthly CSVs (only the columns we need)."""
    usecols = ["date", "to", "scheduled_time", "delay_minutes", "status", "line", "type"]
    files = sorted(
        p for p in DATA_DIR.glob("20*_*.csv")
        if "invalid" not in p.name and p.stem <= LAST_MONTH
    )
    if not files:
        raise FileNotFoundError(f"No monthly CSV files found in {DATA_DIR}")
    frames = []
    for f in files:
        part = pd.read_csv(f, usecols=usecols, parse_dates=["date", "scheduled_time"])
        if SAMPLE_FRAC < 1:
            part = part.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
        frames.append(part)
        print(f"loaded {f.name}: {len(part):,} rows kept")
    return pd.concat(frames, ignore_index=True)


# ------------------------ stage 2: cleaned data ------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep NJ Transit rows that have a measured delay."""
    print("\nstatus counts:\n", df["status"].value_counts(dropna=False).to_string())
    print("\ntype counts:\n", df["type"].value_counts(dropna=False).to_string())
    n_start = len(df)
    df = df[df["type"] == "NJ Transit"]
    df = df.dropna(subset=["delay_minutes", "scheduled_time", "to", "line"])
    print(f"\nrows after cleaning: {len(df):,} (of {n_start:,})")
    return df.reset_index(drop=True)


# -------------------- stage 3: analysis-ready features --------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn each stop into schedule-known features plus the late/not-late label."""
    out = pd.DataFrame(
        {
            "line": df["line"].astype(str),
            "to_station": df["to"].astype(str),
            "hour": df["scheduled_time"].dt.hour,
            "dayofweek": df["scheduled_time"].dt.dayofweek,
            "month": df["scheduled_time"].dt.month,
        }
    )
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    rush_hours = out["hour"].between(6, 9) | out["hour"].between(16, 19)
    out["is_rush"] = ((out["is_weekend"] == 0) & rush_hours).astype(int)
    out["service_date"] = df["date"].values
    out["late"] = (df["delay_minutes"] >= LATE_MINUTES).astype(int)
    return out


def split_by_time(feats: pd.DataFrame):
    """Older months to fit, middle months to choose a threshold, newest months to test."""
    fit = feats[feats["service_date"] < VAL_START]
    val = feats[(feats["service_date"] >= VAL_START) & (feats["service_date"] < TEST_START)]
    test = feats[feats["service_date"] >= TEST_START]
    return fit, val, test


# ------------------------------- models -------------------------------
def make_logistic() -> Pipeline:
    prep = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), CAT + ["hour", "dayofweek", "month"])],
        remainder="passthrough",
    )
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=200))])


def make_boosted_trees() -> Pipeline:
    prep = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CAT)],
        remainder="passthrough",
    )
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        categorical_features=[True] * len(CAT) + [False] * len(NUM),
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def best_threshold(y_true, proba) -> float:
    """Probability cutoff that gives the best F1 on the validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    return float(thresholds[np.argmax(f1)])


def score(name, y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "threshold": round(threshold, 3),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
    }


# ------------------------------- outputs -------------------------------
def save_charts(feats: pd.DataFrame, importance: pd.Series) -> None:
    by_hour = feats.groupby("hour")["late"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_hour.index, by_hour.values, color="#2b6cb0")
    ax.set(xlabel="Scheduled hour of day", ylabel="% of stops 5+ min late", title="Late stops by hour")
    fig.tight_layout(); fig.savefig(OUT_DIR / "late_rate_by_hour.png", dpi=150); plt.close(fig)

    by_line = (feats.groupby("line")["late"].mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(by_line.index, by_line.values, color="#2b6cb0")
    ax.set(xlabel="% of stops 5+ min late", title="Late stops by line")
    fig.tight_layout(); fig.savefig(OUT_DIR / "late_rate_by_line.png", dpi=150); plt.close(fig)

    grid = feats.pivot_table(index="hour", columns="dayofweek", values="late", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(grid.values, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(len(grid.columns)), [DAYS[d] for d in grid.columns])
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set(xlabel="Day of week", ylabel="Scheduled hour", title="% of stops 5+ min late")
    fig.colorbar(image, ax=ax, label="% late")
    fig.tight_layout(); fig.savefig(OUT_DIR / "late_rate_heatmap.png", dpi=150); plt.close(fig)

    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#2b6cb0")
    ax.set(xlabel="Drop in PR-AUC when the feature is shuffled", title="Which features matter most")
    fig.tight_layout(); fig.savefig(OUT_DIR / "feature_importance.png", dpi=150); plt.close(fig)


def save_tables(feats: pd.DataFrame, raw_clean: pd.DataFrame) -> None:
    """Small summary tables that the dashboard reads (the raw data is not committed)."""
    rates = (
        feats.groupby(["line", "dayofweek", "hour"])
        .agg(stops=("late", "size"), late_stops=("late", "sum"))
        .reset_index()
    )
    rates.to_csv(OUT_DIR / "late_rate_by_line_hour_day.csv", index=False)
    stations = (
        feats[["line", "to_station"]].drop_duplicates().sort_values(["line", "to_station"])
    )
    stations.to_csv(OUT_DIR / "stations_by_line.csv", index=False)


def write_results(metrics: list, info: dict) -> None:
    lines = [
        "# Results",
        "",
        f"Target: a stop is \"late\" if the train is {LATE_MINUTES}+ minutes behind schedule.",
        f"Training data: {info['fit_rows']:,} stops before {VAL_START}. "
        f"Threshold chosen on {info['val_rows']:,} stops from {VAL_START} to {TEST_START}. "
        f"Test: {info['test_rows']:,} stops from {TEST_START} to the end of 2019.",
        f"Share of late stops in the test period: {info['test_late_rate']:.1%}.",
        "",
        "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['model']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m['roc_auc']:.3f} | {m['pr_auc']:.3f} |"
        )
    lines += [
        "",
        "PR-AUC of a model with no skill equals the late rate above. "
        "Rows use only schedule-known features, so these numbers show how much "
        "delay risk is predictable before a train runs.",
    ]
    (OUT_DIR / "results.md").write_text("\n".join(lines) + "\n")


# --------------------------------- main ---------------------------------
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    raw = load_raw()
    cleaned = clean(raw)
    feats = build_features(cleaned)

    fit, val, test = split_by_time(feats)
    print(f"\nfit {len(fit):,} | val {len(val):,} | test {len(test):,}")
    print(f"late share: fit {fit['late'].mean():.1%} | test {test['late'].mean():.1%}")
    if min(len(fit), len(val), len(test)) == 0:
        raise ValueError("One of the time periods is empty. Check LAST_MONTH, VAL_START, TEST_START.")

    X = CAT + NUM
    results = []

    # Baseline 0: always predict "on time"
    dummy = DummyClassifier(strategy="most_frequent").fit(fit[X], fit["late"])
    results.append(score("Always predict on time", test["late"], dummy.predict_proba(test[X])[:, 1], 0.5))

    # Baseline 1: logistic regression. Final model: gradient boosted trees.
    models = {"Logistic regression (baseline)": make_logistic(), "Gradient boosting": make_boosted_trees()}
    for name, model in models.items():
        print(f"\ntraining {name} ...")
        model.fit(fit[X], fit["late"])
        threshold = best_threshold(val["late"], model.predict_proba(val[X])[:, 1])
        results.append(score(name, test["late"], model.predict_proba(test[X])[:, 1], threshold))

    print("\nTEST RESULTS")
    print(pd.DataFrame(results).to_string(index=False))

    final = models["Gradient boosting"]
    sample = test.sample(n=min(30_000, len(test)), random_state=RANDOM_STATE)
    perm = permutation_importance(
        final, sample[X], sample["late"], scoring="average_precision",
        n_repeats=3, random_state=RANDOM_STATE,
    )
    importance = pd.Series(perm.importances_mean, index=X)
    print("\nfeature importance:\n", importance.sort_values(ascending=False).round(4).to_string())

    save_charts(feats, importance)
    save_tables(feats, cleaned)
    joblib.dump(final, OUT_DIR / "model.joblib")
    info = {
        "fit_rows": len(fit), "val_rows": len(val), "test_rows": len(test),
        "test_late_rate": float(test["late"].mean()), "total_rows": len(feats),
    }
    write_results(results, info)
    (OUT_DIR / "metrics.json").write_text(json.dumps({"results": results, "info": info}, indent=2))
    print(f"\nDone. Files saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
