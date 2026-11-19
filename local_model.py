"""
Glacier-wide discharge model that runs locally on a laptop CPU.

    AWS_PROFILE=kopah pixi run -e local python local_model.py

Why this exists
---------------
The target, glacier-wide discharge, is a single time series (one value per month), so the
problem is ~300 samples, not millions of pixel rows. That series is ~99% slow trend
(10 -> 17 Gt/yr, 2000-2025) and ~1% seasonal anomaly (std ~0.25 Gt/yr). The forcings we
have (runoff, air temperature, ocean thermal forcing) are seasonal and carry no information
about the multi-year trend, and tree models cannot extrapolate beyond the training range.
Asking XGBoost to predict the *level* of discharge from these forcings therefore fails out
of sample: it latches onto whatever input happens to drift with time (terminus area, OTF,
x/y) and predicts past levels when that input moves.

So we split the problem:

    discharge(t) = slow(t) + anomaly(t)

* slow(t)    : centered 12-month running mean. Forecast naively from the past
               (last 12-month mean + damped recent trend, see slow_forecast), since
               none of our forcings explain it.
* anomaly(t) : what a seasonality model should explain. Fit a ladder of models
               of increasing complexity and keep a rung only if it beats the one below.

Every model is scored with an expanding-window backtest: for each cutoff (1 January of each
year), train on everything before the cutoff and predict the next HORIZON months. No test
data is used for early stopping or model selection.
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import S3_BUCKET, GLACIER_NAME, TARGET_COL, storage_options

########## Settings ##########
RUNOFF = "Basin-Averaged Meltwater Runoff (mmWE/day)"
AIRT   = "Air Temperature (C)"
OTF    = "Ocean Thermal Forcing (C)"
TERM   = "Terminus Area (km^2)"

CUTOFFS  = pd.date_range("2012-01-01", "2025-01-01", freq="YS")   # one backtest fold per year
HORIZON  = 12                                                       # months predicted after each cutoff
MAX_LAG  = 4                                                        # months of forcing history used as features
LOCAL_CACHE = "data/monthly_non_spatial.parquet"
FIG_DIR     = "figures"
RESULTS     = "local_results.md"


########## Data ##########
def load_monthly() -> pd.DataFrame:
    """Monthly means of the non-spatial table, indexed by month start. Cached locally."""
    if os.path.exists(LOCAL_CACHE):
        return pd.read_parquet(LOCAL_CACHE)
    raw = pd.read_parquet(f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_non_spatial.parquet",
                          storage_options=storage_options)
    raw = raw.reset_index().drop(columns=["index"], errors="ignore")
    raw["time"] = pd.to_datetime(raw["time"]).dt.normalize()
    df = raw.resample("MS", on="time").mean()
    os.makedirs(os.path.dirname(LOCAL_CACHE), exist_ok=True)
    df.to_parquet(LOCAL_CACHE)
    return df


def slow_forecast(y_past: pd.Series, horizon: int, trend_months: int = 24, damping: float = 0.5) -> np.ndarray:
    """
    Naive forecast of the slow component for the `horizon` months after y_past ends:
    the last 12-month mean, plus `damping` x the recent trend (change between the last two
    12-month means), extrapolated from the centre of the last 12-month window.
    damping=0 is flat persistence; damping=0.5 was better in backtests (2012-2025).
    """
    base = y_past.iloc[-12:].mean()
    rate = (base - y_past.iloc[-trend_months:-trend_months + 12].mean()) / (trend_months - 12)
    months_ahead = np.arange(horizon) + 6.5
    return base + damping * rate * months_ahead


def slow_component(y: pd.Series) -> pd.Series:
    """Centered 12-month running mean (NaN within 6 months of either end of y)."""
    return y.rolling(12, center=True, min_periods=12).mean()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Annual harmonics plus 0..MAX_LAG month lags of each forcing."""
    X = pd.DataFrame(index=df.index)
    doy = 2 * np.pi * (df.index.month - 0.5) / 12
    X["sin1"], X["cos1"] = np.sin(doy), np.cos(doy)
    X["sin2"], X["cos2"] = np.sin(2 * doy), np.cos(2 * doy)
    forcings = {"runoff": df[RUNOFF], "airt": df[AIRT], "otf": df[OTF],
                # Month-to-month change in terminus area (advance > 0). Sparse before 2013,
                # so only short gaps are interpolated; longer gaps stay NaN and those rows drop out.
                "dterm": df[TERM].interpolate(limit=3, limit_area="inside").diff()}
    for name, s in forcings.items():
        for k in range(MAX_LAG + 1):
            X[f"{name}_lag{k}"] = s.shift(k)
    return X


def cols(*groups) -> list:
    """Feature columns for the named groups, e.g. cols("harm", "runoff")."""
    out = []
    for g in groups:
        out += ["sin1", "cos1", "sin2", "cos2"] if g == "harm" else [f"{g}_lag{k}" for k in range(MAX_LAG + 1)]
    return out


########## Anomaly models: fit(X_train, a_train) -> predict(X_test) ##########
# Each takes the training anomaly series and features, and returns predictions for X_test.
def zero(Xtr, atr, Xte):
    return np.zeros(len(Xte))

def climatology(Xtr, atr, Xte):
    clim = atr.groupby(atr.index.month).mean()
    return Xte.index.month.map(clim).to_numpy(dtype=float)

def ridge(feature_cols):
    def fit_predict(Xtr, atr, Xte):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
        m.fit(Xtr[feature_cols], atr)
        return m.predict(Xte[feature_cols])
    return fit_predict

def xgboost(feature_cols):
    # Shallow, heavily regularized, fixed number of rounds: there are only ~150-300 samples
    params = dict(objective="reg:squarederror", max_depth=2, eta=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0, seed=42)
    def fit_predict(Xtr, atr, Xte):
        m = xgb.train(params, xgb.DMatrix(Xtr[feature_cols], label=atr), num_boost_round=200)
        return m.predict(xgb.DMatrix(Xte[feature_cols]))
    return fit_predict

# The ladder: each rung adds one thing to the rung above it.
ANOMALY_MODELS = {
    "0 zero anomaly":                 (zero, []),
    "1 monthly climatology":          (climatology, []),
    "2 ridge: harmonics":             (ridge(cols("harm")), cols("harm")),
    "3 ridge: harmonics+runoff":      (ridge(cols("harm", "runoff")), cols("harm", "runoff")),
    "4 ridge: harmonics+runoff+airT": (ridge(cols("harm", "runoff", "airt")), cols("harm", "runoff", "airt")),
    "5 ridge: +OTF":                  (ridge(cols("harm", "runoff", "airt", "otf")), cols("harm", "runoff", "airt", "otf")),
    "6 xgboost: harmonics+runoff+airT": (xgboost(cols("harm", "runoff", "airt")), cols("harm", "runoff", "airt")),
    "7 ridge: harmonics+Δterminus":   (ridge(cols("harm", "dterm")), cols("harm", "dterm")),
}


########## Level baselines (for comparison with the original approach) ##########
def xgb_level(train, test):
    """The original formulation: XGBoost predicting discharge level from contemporaneous forcings."""
    feats = [AIRT, OTF, RUNOFF]
    params = dict(objective="reg:squarederror", max_depth=5, eta=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=10, seed=42)
    tr = train.dropna(subset=[TARGET_COL])
    m = xgb.train(params, xgb.DMatrix(tr[feats], label=tr[TARGET_COL]), num_boost_round=200)
    return m.predict(xgb.DMatrix(test[feats]))


########## Backtest ##########
def backtest(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET_COL]
    X = make_features(df)
    true_anom = y - slow_component(y)          # truth for scoring (uses the whole record)
    rows = []
    for cutoff in CUTOFFS:
        test_idx = pd.date_range(cutoff, periods=HORIZON, freq="MS").intersection(df.index)
        y_past = y[y.index < cutoff].dropna()
        # Anomaly target from training data only, so nothing after the cutoff leaks in
        a_train = (y_past - slow_component(y_past)).dropna()
        slow_fc = slow_forecast(y_past, len(test_idx))

        preds = {"persistence (last value)": np.full(len(test_idx), y_past.iloc[-1]),
                 "persistence (last 12-mo mean)": np.full(len(test_idx), y_past.iloc[-12:].mean()),
                 "original: xgboost on level": xgb_level(df[df.index < cutoff], df.loc[test_idx])}
        for name, (fit_predict, feats) in ANOMALY_MODELS.items():
            ok = X.loc[a_train.index, feats].notna().all(axis=1)       # e.g. air T starts 2008-07
            Xte = X.loc[test_idx, feats]
            # Missing forcings in the test window (e.g. OTF after 2024) -> no prediction
            complete = Xte.notna().all(axis=1).to_numpy()
            anom_pred = np.full(len(test_idx), np.nan)
            if complete.any() and ok.sum() >= 36:      # need >= 3 years of complete training rows
                anom_pred[complete] = fit_predict(X.loc[a_train.index[ok]], a_train[ok], Xte[complete])
            preds[name] = slow_fc + anom_pred
            preds[f"{name} [anomaly]"] = anom_pred

        for name, p in preds.items():
            rows.append(pd.DataFrame({"cutoff": cutoff, "time": test_idx, "model": name, "pred": p,
                                      "truth": y.loc[test_idx].to_numpy(),
                                      "truth_anom": true_anom.loc[test_idx].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def score(bt: pd.DataFrame) -> pd.DataFrame:
    """RMSE of the level forecast and (for anomaly models) of the anomaly itself, pooled over folds."""
    out = []
    for name, g in bt[~bt.model.str.endswith("[anomaly]")].groupby("model"):
        g = g.dropna(subset=["pred", "truth"])
        row = {"model": name, "n": len(g), "RMSE level (Gt/yr)": np.sqrt(np.mean((g.pred - g.truth) ** 2))}
        a = bt[bt.model == f"{name} [anomaly]"].dropna(subset=["pred", "truth_anom"])
        if len(a):
            rmse = np.sqrt(np.mean((a.pred - a.truth_anom) ** 2))
            row["RMSE anomaly (Gt/yr)"] = rmse
            row["anomaly skill vs zero"] = 1 - rmse ** 2 / np.mean(a.truth_anom ** 2)
            row["corr anomaly"] = np.corrcoef(a.pred, a.truth_anom)[0, 1] if a.pred.std() > 0 else np.nan
        out.append(row)
    return pd.DataFrame(out).set_index("model").sort_index()


########## Figures ##########
def plot(df: pd.DataFrame, bt: pd.DataFrame, models: list, path: str):
    y = df[TARGET_COL]
    anom = y - slow_component(y)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax1.plot(y.index, y, color="black", lw=1, label="truth")
    ax2.plot(anom.index, anom, color="black", lw=1, label="truth")
    for i, name in enumerate(models):
        lvl = bt[bt.model == name]
        an = bt[bt.model == f"{name} [anomaly]"]
        for j, (_, g) in enumerate(lvl.groupby("cutoff")):
            ax1.plot(g.time, g.pred, color=f"C{i}", lw=1.2, label=name if j == 0 else None)
        for j, (_, g) in enumerate(an.groupby("cutoff")):
            ax2.plot(g.time, g.pred, color=f"C{i}", lw=1.2, label=name if j == 0 else None)
    for c in CUTOFFS:
        for ax in (ax1, ax2):
            ax.axvline(c, color="0.85", lw=0.5, zorder=0)
    ax1.set_ylabel(TARGET_COL); ax1.set_title(f"Expanding-window backtest: {HORIZON}-month forecasts from each 1 Jan cutoff")
    ax2.set_ylabel("Seasonal anomaly (Gt/yr)\n(discharge - centered 12-mo mean)")
    ax2.axhline(0, color="0.6", lw=0.5)
    ax1.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="lower left")
    ax1.set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2026-03-01"))
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main():
    df = load_monthly()
    bt = backtest(df)
    scores = score(bt)
    print(scores.round(3).to_string())

    os.makedirs(FIG_DIR, exist_ok=True)
    plot(df, bt, ["original: xgboost on level", "persistence (last value)",
                  "6 xgboost: harmonics+runoff+airT"], f"{FIG_DIR}/local_backtest.png")
    plot(df, bt, ["1 monthly climatology", "4 ridge: harmonics+runoff+airT",
                  "6 xgboost: harmonics+runoff+airT"], f"{FIG_DIR}/local_backtest_anomaly_models.png")

    with open(RESULTS, "w") as f:
        f.write(f"## Local backtest ({CUTOFFS[0].year}-{CUTOFFS[-1].year} cutoffs, {HORIZON}-month horizon)\n\n")
        f.write(scores.round(3).to_markdown() + "\n")
    bt.to_parquet("data/local_backtest.parquet")
    return df, bt, scores


if __name__ == "__main__":
    df, bt, scores = main()
