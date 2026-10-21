"""
EDMD Framework: event-inclusion simulation benchmark
=====================================================

Every number this script reports is computed from a generated event stream.
Nothing is hardcoded.

Method
------
1. A vectorized generator draws accounts with a latent risk factor, four event
   classes (Transaction, Payment, BureauRefresh, AccountChange), an event time
   and an arrival time for every event, and a default label that depends on the
   same latent factors that drive behaviour.
2. Each architecture is modelled by the rule that decides which events
   contribute to a feature value on three paths:
       train     : the value the training pipeline produces for (account, t)
       serve     : the value available to the scoring service at decision time t
       eventual  : the value the production (serving) store holds at the end of
                   the simulation, after any recovery or correction
   The rules are written out in `inclusion_masks` below. No streaming engine is
   run; the simulation isolates the semantics of each architecture.
3. The same feature definitions are applied to every architecture.
4. Metrics
       TSS    : mean over features of mean |train - serve| / sd(truth)
       SERR   : mean over features of mean |serve - truth| / sd(truth)
       LERR   : share of injected late events (lateness beyond their class
                bound, event time inside a feature window) that are reflected
                in the eventual serving-path value, macro-averaged over the
                four event classes
       AUC    : XGBoost trained on train-path features of training accounts,
                scored on serve-path features of held-out accounts
       ORACLE : same model trained and scored on complete (truth) features
       CORR   : EDMD only, share of served Platinum records that later receive
                a corrected version (the storage and compute cost of versioning)

Usage
-----
    python src/evaluation/simulation_benchmark.py --seeds 42 43 44 45 46 \
        --n_accounts 50000 --output_dir artifacts/simulation
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

MIN = 1.0 / 1440.0          # one minute in days
HOUR = 1.0 / 24.0

# Class lateness bounds (days), as in the paper's Table 3
BOUND = {"txn": 15 * MIN, "pay": 72 * HOUR, "bur": 30.0, "acc": 24 * HOUR}

ARCHS = ["EDMD", "Feature store", "Kappa", "Three-tier Medallion", "Lambda"]

# Architecture configuration (documented in the paper, Benchmark setup)
CFG = {
    "platinum_latency": 5 * MIN,   # EDMD serve reads Platinum written >= 5 min ago
    "kappa_allowed_lateness": 72 * HOUR,
    "medallion_lookback_days": 3,  # incremental Gold recomputes the last K event-days
    "fs_materialize_every": 15 * MIN,
}

FEATURES = [
    "txn_count_30d", "txn_amount_30d", "txn_amount_7d", "utilization_30d",
    "pay_ratio_90d", "late_pay_count_90d",
    "bureau_score_latest", "bureau_score_delta_90d",
    "limit_latest", "limit_decrease_90d",
]


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------
def generate(n, seed, late_rate=0.10, t=180.53):
    """Return (accounts DataFrame, dict of event arrays per class, as-of time t).

    Time is in days. All feature windows end at the as-of time t."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)                                  # latent risk
    shock = rng.random(n) < 0.08                              # recent deterioration
    base_limit = np.exp(rng.normal(8.5, 0.6, n)).clip(500, 50000)
    p_def = 1 / (1 + np.exp(-(-4.3 + 1.5 * z + 2.5 * shock)))
    y = (rng.random(n) < p_def).astype(int)

    def arrivals(cls, te):
        m = len(te)
        if cls == "txn":
            lag = rng.exponential(3 * MIN, m)
        elif cls == "pay":
            lag = rng.uniform(24 * HOUR, 66 * HOUR, m)
        elif cls == "bur":
            lag = rng.uniform(1.0, 20.0, m)
        else:
            lag = rng.exponential(2 * HOUR, m)
        late = rng.random(m) < late_rate
        b = BOUND[cls]
        lag[late] = b + rng.exponential(0.3 * b, late.sum())
        return te + lag, late

    ev = {}
    # Transactions over the last 30 days
    rate = 2.0 * np.exp(0.25 * z)
    k = rng.poisson(rate * 30)
    acc = np.repeat(np.arange(n), k)
    te = t - rng.uniform(0, 30, len(acc))
    recent = te > t - 20
    spend_mult = np.where(shock[acc] & recent, 1.8, 1.0)
    mu = np.log(base_limit[acc] * 0.012 * (1 + 0.4 * np.clip(z[acc], 0, None)))
    amt = np.exp(rng.normal(mu, 0.6)) * spend_mult
    ta, late = arrivals("txn", te)
    ev["txn"] = dict(acc=acc, te=te, ta=ta, late=late, amount=amt)

    # Payments: three monthly statements in the 90-day window
    acc = np.repeat(np.arange(n), 3)
    kk = np.tile(np.arange(3), n)
    due = t - 90 + 25 + kk * 30 + rng.uniform(0, 5, len(acc))
    dbd = np.round(rng.normal(5 - 3.0 * z[acc] - 6 * (shock[acc] & (kk == 2)), 4))
    te = due - dbd
    keep = te <= t
    stmt = base_limit[acc] * np.clip(rng.beta(2, 5, len(acc)) * (1 + 0.3 * np.clip(z[acc], 0, None)), 0.01, 1)
    ratio = rng.beta(2, 2, len(acc)) * (1 / (1 + np.exp(-(1.0 - 1.0 * z[acc] - 1.5 * shock[acc]))))
    ta, late = arrivals("pay", te)
    ev["pay"] = dict(acc=acc[keep], te=te[keep], ta=ta[keep], late=late[keep],
                     amount=(ratio * stmt)[keep], stmt=stmt[keep], dbd=dbd[keep])

    # Bureau refreshes: monthly, three in the window
    acc = np.repeat(np.arange(n), 3)
    kk = np.tile(np.arange(3), n)
    te = t - 90 + kk * 30 + rng.uniform(0, 30, len(acc))
    drift = 6 * np.clip(z[acc], 0, None) + 18 * shock[acc] * (kk == 2)
    score = 720 - 35 * z[acc] - drift * kk + rng.normal(0, 12, len(acc))
    ta, late = arrivals("bur", te)
    ev["bur"] = dict(acc=acc, te=te, ta=ta, late=late, score=score)

    # Account changes: onboarding record (old, always arrived) + rare limit changes
    acc0 = np.arange(n)
    te0 = np.full(n, t - 365.0)
    ta0 = te0 + HOUR
    chg = rng.random(n) < (0.06 + 0.10 * (z > 1))
    acc1 = np.where(chg)[0]
    te1 = t - rng.uniform(0, 90, len(acc1))
    new_lim = base_limit[acc1] * np.where(z[acc1] > 1, 0.7, 1.2)
    ta1, late1 = arrivals("acc", te1)
    ev["acc"] = dict(acc=np.r_[acc0, acc1], te=np.r_[te0, te1], ta=np.r_[ta0, ta1],
                     late=np.r_[np.zeros(n, bool), late1],
                     limit=np.r_[base_limit, new_lim],
                     decrease=np.r_[np.zeros(n), (new_lim < base_limit[acc1]).astype(float)],
                     onboarding=np.r_[np.ones(n, bool), np.zeros(len(acc1), bool)])
    accounts = pd.DataFrame({"z": z, "shock": shock, "y": y})
    return accounts, ev, t


# --------------------------------------------------------------------------
# Inclusion rules
# --------------------------------------------------------------------------
def inclusion_masks(arch, path, e, t, W):
    """Boolean mask of events that contribute to a feature with window W days,
    for one architecture and path. Window: event time in (t - W, t]."""
    te, ta = e["te"], e["ta"]
    inwin = (te > t - W) & (te <= t)
    if arch == "Truth":
        return inwin
    if arch == "EDMD":
        # One computation over one log. Serve reads the latest Platinum version
        # written at least `platinum_latency` before t; late events beyond the
        # class bound are folded in by Algorithm 1 as new versions. Training
        # reads the versions that were served (Property 1).
        if path in ("serve", "train"):
            return inwin & (ta <= t - CFG["platinum_latency"])
        return inwin                                  # eventual: corrected
    if arch == "Kappa":
        # One streaming job, event-time windows, single global allowed lateness.
        # Events later than that are dropped; replay applies the same rule.
        ok = ta <= te + CFG["kappa_allowed_lateness"]
        if path == "serve":
            return inwin & ok & (ta <= t)
        return inwin & ok                             # train (replay) and eventual
    if arch == "Lambda":
        # Batch view from the last nightly run b (event-time window ending at b)
        # plus a speed layer of everything that arrived after b, windowed by
        # processing time. Training uses a batch recompute with full hindsight.
        b = np.floor(t)
        if path == "serve":
            batch = (te > b - W) & (te <= b) & (ta <= b)
            speed = (ta > b) & (ta <= t)
            return batch | speed
        if path == "train":
            return inwin
        return inwin                                  # next batch run catches up
    if arch == "Three-tier Medallion":
        # Nightly incremental Gold that recomputes only the last K event-days.
        # Events arriving after that lookback are never merged.
        K = CFG["medallion_lookback_days"]
        merged = ta <= np.floor(te) + 1 + K
        if path == "serve":
            b = np.floor(t)
            return (te > b - W) & (te <= b) & merged & (ta <= b)
        return inwin & merged
    if arch == "Feature store":
        # Offline store: point-in-time join on event timestamp over the full
        # history (includes events that arrived after t). Online store:
        # materialized every 15 min by ingestion time.
        if path == "train":
            return inwin
        if path == "serve":
            m = CFG["fs_materialize_every"]
            last_run = np.floor(t / m) * m
            return inwin & (ta <= last_run)
        return inwin
    raise ValueError(arch)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def _sum(acc, w, mask, n):
    return np.bincount(acc[mask], weights=None if w is None else w[mask], minlength=n)


def _latest(acc, te, val, mask, n):
    out = np.full(n, np.nan)
    a, t_, v = acc[mask], te[mask], val[mask]
    if len(a) == 0:
        return out
    o = np.lexsort((t_, a))
    a, v = a[o], v[o]
    last = np.r_[a[1:] != a[:-1], True]
    out[a[last]] = v[last]
    return out


def _earliest(acc, te, val, mask, n):
    out = np.full(n, np.nan)
    a, t_, v = acc[mask], te[mask], val[mask]
    if len(a) == 0:
        return out
    o = np.lexsort((t_, a))
    a, v = a[o], v[o]
    first = np.r_[True, a[1:] != a[:-1]]
    out[a[first]] = v[first]
    return out


def compute_features(arch, path, ev, t, n):
    T, P, B, A = ev["txn"], ev["pay"], ev["bur"], ev["acc"]
    f = {}
    m30 = inclusion_masks(arch, path, T, t, 30)
    m7 = inclusion_masks(arch, path, T, t, 7)
    f["txn_count_30d"] = _sum(T["acc"], None, m30, n).astype(float)
    f["txn_amount_30d"] = _sum(T["acc"], T["amount"], m30, n)
    f["txn_amount_7d"] = _sum(T["acc"], T["amount"], m7, n)

    mp = inclusion_masks(arch, path, P, t, 90)
    paid = _sum(P["acc"], P["amount"], mp, n)
    stmt = _sum(P["acc"], P["stmt"], mp, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["pay_ratio_90d"] = np.where(stmt > 0, paid / stmt, np.nan)
    f["late_pay_count_90d"] = _sum(P["acc"], (P["dbd"] < 0).astype(float), mp, n)

    mb = inclusion_masks(arch, path, B, t, 90)
    latest = _latest(B["acc"], B["te"], B["score"], mb, n)
    f["bureau_score_latest"] = latest
    f["bureau_score_delta_90d"] = latest - _earliest(B["acc"], B["te"], B["score"], mb, n)

    # Account master: onboarding record is outside the window but always present
    ma = inclusion_masks(arch, path, A, t, 90) | A["onboarding"]
    f["limit_latest"] = _latest(A["acc"], A["te"], A["limit"], ma, n)
    f["limit_decrease_90d"] = _sum(A["acc"], A["decrease"], ma & ~A["onboarding"], n)
    f["utilization_30d"] = f["txn_amount_30d"] / f["limit_latest"]
    return pd.DataFrame(f)[FEATURES]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def norm_mad(a, b, truth):
    """Mean over features of mean |a - b| / sd(truth). NaNs imputed with truth mean."""
    vals = {}
    for c in FEATURES:
        mu, sd = np.nanmean(truth[c]), np.nanstd(truth[c])
        x = np.nan_to_num(a[c].to_numpy(), nan=mu)
        y = np.nan_to_num(b[c].to_numpy(), nan=mu)
        vals[c] = float(np.mean(np.abs(x - y)) / sd) if sd > 0 else 0.0
    return float(np.mean(list(vals.values()))), vals


def lerr(arch, ev, t):
    """Share of injected late events inside a feature window that are reflected
    in the eventual serving-path value."""
    windows = {"txn": 30, "pay": 90, "bur": 90, "acc": 90}
    hit = tot = 0
    for cls, W in windows.items():
        e = ev[cls]
        inwin = (e["te"] > t - W) & (e["te"] <= t)
        late = e["late"] & inwin
        if cls == "acc":
            late &= ~e["onboarding"]
        m = inclusion_masks(arch, "eventual", e, t, W)
        hit += int((late & m).sum())
        tot += int(late.sum())
    # Macro average over event classes so that high-volume transactions do not
    # dominate; the event-weighted value is hit / tot.
    return float(np.nanmean(list(lerr_by_class(arch, ev, t).values())))


def lerr_by_class(arch, ev, t):
    windows = {"txn": 30, "pay": 90, "bur": 90, "acc": 90}
    out = {}
    for cls, W in windows.items():
        e = ev[cls]
        inwin = (e["te"] > t - W) & (e["te"] <= t)
        late = e["late"] & inwin
        if cls == "acc":
            late &= ~e["onboarding"]
        m = inclusion_masks(arch, "eventual", e, t, W)
        out[cls] = float((late & m).sum() / late.sum()) if late.sum() else float("nan")
    return out


def fit_auc(Xtr, ytr, Xte, yte, seed):
    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="auc", random_state=seed, n_jobs=2)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p)), p


def run_seed(seed, n, late_rate=0.10, n_boot=500):
    t0 = time.time()
    accts, ev, t = generate(n, seed, late_rate)
    truth = compute_features("Truth", "serve", ev, t, n)
    y = accts["y"].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.5, stratify=y, random_state=seed)

    res = {"seed": seed, "n_accounts": n, "late_rate": late_rate,
           "default_rate": float(y.mean()),
           "n_events": {k: int(len(v["acc"])) for k, v in ev.items()}}
    oracle_auc, p_or = fit_auc(truth.iloc[idx_tr], y[idx_tr], truth.iloc[idx_te], y[idx_te], seed)
    res["oracle_auc"] = oracle_auc
    preds = {}
    res["arch"] = {}
    for a in ARCHS:
        tr = compute_features(a, "train", ev, t, n)
        sv = compute_features(a, "serve", ev, t, n)
        evn = compute_features(a, "eventual", ev, t, n)
        tss, tss_f = norm_mad(tr, sv, truth)
        serr, _ = norm_mad(sv, truth, truth)
        auc, p = fit_auc(tr.iloc[idx_tr], y[idx_tr], sv.iloc[idx_te], y[idx_te], seed)
        preds[a] = p
        r = {"tss": tss, "tss_by_feature": tss_f, "serve_error": serr,
             "lerr": lerr(a, ev, t), "lerr_by_class": lerr_by_class(a, ev, t), "auc": auc}
        if a == "EDMD":
            # Correction rate: served Platinum records that later change
            diff = {}
            for c in FEATURES:
                s, e_ = sv[c].to_numpy(), evn[c].to_numpy()
                ch = ~((s == e_) | (np.isnan(s) & np.isnan(e_)))
                diff[c] = float(ch.mean())
            r["correction_rate_by_feature"] = diff
            r["correction_rate"] = float(np.mean(list(diff.values())))
        res["arch"][a] = r
    # Paired bootstrap of AUC differences (EDMD minus baseline) on held-out accounts
    rng = np.random.default_rng(seed + 1000)
    yt = y[idx_te]
    boots = {a: [] for a in ARCHS if a != "EDMD"}
    for _ in range(n_boot):
        s = rng.integers(0, len(yt), len(yt))
        if yt[s].min() == yt[s].max():
            continue
        e_auc = roc_auc_score(yt[s], preds["EDMD"][s])
        for a in boots:
            boots[a].append(e_auc - roc_auc_score(yt[s], preds[a][s]))
    res["auc_diff_ci"] = {a: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                          for a, v in boots.items() if v}
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def kappa_sensitivity(seed, n):
    accts, ev, t = generate(n, seed)
    truth = compute_features("Truth", "serve", ev, t, n)
    out = {}
    orig = CFG["kappa_allowed_lateness"]
    for label, L in [("1 h", HOUR), ("24 h", 24 * HOUR), ("72 h", 72 * HOUR), ("30 d", 30.0)]:
        CFG["kappa_allowed_lateness"] = L
        tr = compute_features("Kappa", "train", ev, t, n)
        sv = compute_features("Kappa", "serve", ev, t, n)
        out[label] = {"tss": norm_mad(tr, sv, truth)[0], "lerr": lerr("Kappa", ev, t)}
    CFG["kappa_allowed_lateness"] = orig
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n_accounts", type=int, default=50000)
    ap.add_argument("--late_rates", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20])
    ap.add_argument("--output_dir", default="artifacts/simulation")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    runs = []
    for s in args.seeds:
        r = run_seed(s, args.n_accounts, 0.10)
        print(f"seed {s} done in {r['runtime_s']}s  oracle AUC {r['oracle_auc']:.3f}")
        runs.append(r)
    json.dump({"config": CFG, "runs": runs}, open(f"{args.output_dir}/main_runs.json", "w"), indent=2)

    rows = []
    for a in ARCHS:
        for k in ["tss", "serve_error", "lerr", "auc"]:
            v = np.array([r["arch"][a][k] for r in runs])
            rows.append({"architecture": a, "metric": k, "mean": v.mean(), "sd": v.std(ddof=1)})
    v = np.array([r["oracle_auc"] for r in runs])
    rows.append({"architecture": "Oracle (complete features)", "metric": "auc", "mean": v.mean(), "sd": v.std(ddof=1)})
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{args.output_dir}/summary.csv", index=False)
    print(summ.pivot(index="architecture", columns="metric", values="mean").round(4))

    # Late-rate sensitivity (first seed)
    sens = []
    for lr in args.late_rates:
        r = run_seed(args.seeds[0], args.n_accounts, lr, n_boot=0)
        for a in ARCHS:
            sens.append({"late_rate": lr, "architecture": a, "tss": r["arch"][a]["tss"],
                         "lerr": r["arch"][a]["lerr"], "auc": r["arch"][a]["auc"]})
    pd.DataFrame(sens).to_csv(f"{args.output_dir}/late_rate_sensitivity.csv", index=False)

    json.dump(kappa_sensitivity(args.seeds[0], args.n_accounts),
              open(f"{args.output_dir}/kappa_sensitivity.json", "w"), indent=2)
    print("done")


if __name__ == "__main__":
    main()
