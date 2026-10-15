"""Train, calibrate and evaluate the two-head risk model.
Evaluation is honest about what is possible: the attack-IP head is tested on a future period; takeover
logins are too few and too early in the timeline for a time split, so that head is cross-validated by user."""
from __future__ import annotations
import json, math, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import FEATURES, build_features
from .modeling import OffHead, RiskModel, _ecdf

_F = {f: i for i, f in enumerate(FEATURES)}
Q = np.linspace(0, 1, 20001)


def _wilson(k, m, z=1.96):
    if m == 0: return (None, None)
    p = k / m; d = 1 + z * z / m; c = p + z * z / (2 * m); r = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m))
    return round(float((c - r) / d), 3), round(float((c + r) / d), 3)


def _predict_ip(gb, X, rows, cols, chunk=1_000_000):
    return np.concatenate([gb.predict_proba(X[rows[i:i + chunk]][:, cols])[:, 1] for i in range(0, len(rows), chunk)]) \
        if len(rows) else np.array([])


def _dev_pipeline():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))


def train(df: pd.DataFrame, out_dir: str, feedback_rows: pd.DataFrame | None = None,
          bulk_threshold: int = C.BULK_USER_THRESHOLD, max_train_rows: int = 4_000_000,
          seed: int = 0, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time(); rng = np.random.default_rng(seed)
    df = df.sort_values("ts_ms", kind="stable").reset_index(drop=True)
    n = len(df)
    log(f"[1/7] building features for {n:,} logins ...")
    X = build_features(df).to_numpy(dtype=np.float32)
    atk, ato, ok = df["attack_ip"].to_numpy(), df["ato"].to_numpy(), df["success"].to_numpy()
    user = df["user"].to_numpy()
    bulk = df.groupby("user")["success"].transform("size").to_numpy() > bulk_threshold
    del df
    normal = ~atk & ~ato
    a, b = int(n * 0.6), int(n * 0.8)
    log(f"      train={a:,} validation={b-a:,} test={n-b:,} | bulk pseudo-account rows excluded from training: {bulk.mean():.1%}")

    # ------------------------------------------------------------------ IP / network head
    log("[2/7] attack-IP head: choosing a SMALL model on the validation period ...")
    ip_feats = [f for f in FEATURES if f not in C.ALLTIME_FEATURES]; ip_cols = [_F[f] for f in ip_feats]
    tr_rows = np.flatnonzero(~bulk[:a] & ~ato[:a])
    if len(tr_rows) > max_train_rows: tr_rows = np.sort(rng.choice(tr_rows, max_train_rows, replace=False))
    va_rows = np.flatnonzero(~bulk[a:b]) + a
    te_rows = np.flatnonzero(~bulk[b:]) + b
    if atk[tr_rows].sum() < 20: raise SystemExit("Too few attack labels in the training period - check column mapping / sample size.")
    Xtr, ytr = X[tr_rows][:, ip_cols], atk[tr_rows].astype(int)
    if feedback_rows is not None and len(feedback_rows):
        Xtr = np.vstack([Xtr, feedback_rows[ip_feats].to_numpy(np.float32)]); ytr = np.concatenate([ytr, feedback_rows["label"].to_numpy(int)])
        wtr = np.concatenate([np.ones(len(tr_rows), np.float32), np.full(len(feedback_rows), C.FEEDBACK_WEIGHT, np.float32)])
        log(f"      + {len(feedback_rows)} analyst-validated rows (weight x{C.FEEDBACK_WEIGHT:g})")
    else:
        wtr = np.ones(len(tr_rows), np.float32)
    sub = va_rows if len(va_rows) <= 1_500_000 else np.sort(rng.choice(va_rows, 1_500_000, replace=False))
    grid, best = [], None
    for trees in (30, 60, 120):
        for leaves in (8, 15):
            gb = HistGradientBoostingClassifier(max_iter=trees, max_leaf_nodes=leaves, learning_rate=0.1,
                                                min_samples_leaf=200, early_stopping=False, random_state=seed).fit(Xtr, ytr, sample_weight=wtr)
            v = roc_auc_score(atk[sub], _predict_ip(gb, X, sub, ip_cols)) if 0 < atk[sub].sum() < len(sub) else 0.5
            grid.append({"trees": trees, "leaves": leaves, "validation_auc": round(float(v), 4)})
            if best is None or v > best[0]: best = (v, gb, trees, leaves)
    _, ip_gb, trees, leaves = best
    log(f"      chosen on validation: {trees} trees, {leaves} leaves (validation AUC {best[0]:.3f})")

    log("[3/7] calibrating the attack-IP head on normal validation logins ...")
    ref_rows = va_rows[normal[va_rows]]
    if len(ref_rows) > 2_000_000: ref_rows = np.sort(rng.choice(ref_rows, 2_000_000, replace=False))
    ip_ref = np.quantile(_predict_ip(ip_gb, X, ref_rows, ip_cols), Q)

    # ------------------------------------------------------------------ takeover head
    log("[4/7] takeover head: fitting on deviation features of established accounts ...")
    warm_all = X[:, _F["history_count_log"]] >= math.log1p(C.WARM_MIN_HISTORY)
    pos = np.flatnonzero(ato & warm_all & ~bulk)
    neg_pool = np.flatnonzero(~bulk[:b] & normal[:b] & ok[:b] & warm_all[:b])
    neg = np.sort(rng.choice(neg_pool, min(1_000_000, len(neg_pool)), replace=False)) if len(neg_pool) else neg_pool
    dev_cols = [_F[f] for f in C.DEV_FEATURES]
    dev_lr = None; takeover_report = {"warm_takeover_logins": int(len(pos)), "all_takeover_logins": int(ato.sum())}
    Xd_fb = None
    if len(pos) >= 10 and len(neg) >= 100:
        fit_rows = np.r_[pos, neg]; Xf = np.nan_to_num(X[fit_rows][:, dev_cols], nan=0.0); yf = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(int)
        wf = np.ones(len(fit_rows), np.float32)
        if feedback_rows is not None and len(feedback_rows):
            fb = feedback_rows[feedback_rows["history_count_log"] >= math.log1p(C.WARM_MIN_HISTORY)]
            if len(fb):
                Xf = np.vstack([Xf, np.nan_to_num(fb[C.DEV_FEATURES].to_numpy(np.float32), nan=0.0)]); yf = np.r_[yf, fb["label"].to_numpy(int)]
                wf = np.r_[wf, np.full(len(fb), C.FEEDBACK_WEIGHT, np.float32)]
        dev_lr = _dev_pipeline(); dev_lr.fit(Xf, yf, logisticregression__sample_weight=wf)
        # honest evaluation: 5-fold, grouped by USER (a user is never in train and test together)
        gk = GroupKFold(n_splits=5); oof = np.zeros(len(fit_rows))
        for tri, tei in gk.split(Xf[:len(fit_rows)], yf[:len(fit_rows)], groups=user[fit_rows]):
            m = _dev_pipeline().fit(Xf[tri], yf[tri]); oof[tei] = m.decision_function(Xf[tei])
        yo = yf[:len(fit_rows)]; po, no = oof[yo == 1], oof[yo == 0]
        rule = (X[fit_rows][:, _F["new_country"]] > 0); rule_p, rule_n = rule[yo == 1], rule[yo == 0]
        cv = {"auc": round(float(roc_auc_score(yo, oof)), 3), "recall_at_fixed_false_alarm_rate_of_warm_normal_logins": {}}
        for f in (0.005, 0.01, 0.02):
            t = np.quantile(no, 1 - f); k = int((po >= t).sum()); lo, hi = _wilson(k, len(po))
            cv["recall_at_fixed_false_alarm_rate_of_warm_normal_logins"][f"{f:.1%}"] = {"caught": k, "of": int(len(po)), "recall": round(k / len(po), 3), "ci95": [lo, hi]}
        k = int(rule_p.sum()); lo, hi = _wilson(k, len(rule_p))
        cv["baseline_rule_new_country"] = {"caught": k, "of": int(len(rule_p)), "recall": round(k / len(rule_p), 3), "ci95": [lo, hi],
                                           "false_alarm_rate": round(float(rule_n.mean()), 4)}
        t = np.quantile(no, 1 - float(rule_n.mean())); k2 = int((po >= t).sum())
        cv["fitted_model_at_the_rules_false_alarm_rate"] = {"caught": k2, "of": int(len(po)), "recall": round(k2 / len(po), 3), "ci95": list(_wilson(k2, len(po)))}
        takeover_report["cross_validated_by_user"] = cv
        takeover_report["max_catchable_share_of_all_takeovers"] = round(len(pos) / max(int(ato.sum()), 1), 3)
        vref = va_rows[normal[va_rows] & ok[va_rows] & warm_all[va_rows]]
        if len(vref) < 200: vref = neg
        dev_ref = np.quantile(dev_lr.decision_function(np.nan_to_num(X[vref][:, dev_cols], nan=0.0)), Q)
    else:
        log("      (too few takeover logins to fit this head: it is switched off, i.e. always 0)")
        dev_ref = np.array([0.0, 1.0])
        dev_lr = OffHead()
        takeover_report["note"] = "not trained: fewer than 10 usable takeover logins"

    medium_thr, high_thr = 100 * (1 - C.HEAD_MEDIUM_BUDGET), 100 * (1 - C.HEAD_HIGH_BUDGET)
    model = RiskModel(ip_gb, ip_feats, ip_ref, dev_lr, C.DEV_FEATURES, dev_ref, C.WARM_MIN_HISTORY, medium_thr, high_thr, {})

    log("[5/7] evaluating the attack-IP head on the untouched TEST period ...")
    p_te = _predict_ip(ip_gb, X, te_rows, ip_cols); y_te = atk[te_rows]
    risk_ip_te = 100 * _ecdf(ip_ref, p_te)
    leg = normal[te_rows]
    fpr, tpr, _ = roc_curve(y_te, p_te)
    ip_report = {
        "chosen": {"trees": trees, "leaves": leaves}, "tuning_grid_validation_auc": grid,
        "test_rows_ordinary_users": int(len(te_rows)), "test_attack_rate": round(float(y_te.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_te, p_te)), 4), "pr_auc": round(float(average_precision_score(y_te, p_te)), 4),
        "recall_at_fixed_false_alarm_rate": {f"{f:.1%}": round(float(tpr[fpr <= f].max()), 4) for f in (0.001, 0.005, 0.01, 0.02, 0.05)},
        "legit_flagged_in_test_pct": {"Medium+ (target %.1f)" % (100 * C.HEAD_MEDIUM_BUDGET): round(float((risk_ip_te[leg] >= medium_thr).mean() * 100), 3),
                                      "High (target %.1f)" % (100 * C.HEAD_HIGH_BUDGET): round(float((risk_ip_te[leg] >= high_thr).mean() * 100), 3)},
    }
    parts = np.array_split(np.arange(len(te_rows)), 6)
    ip_report["legit_flagged_by_test_part_Medium+_pct"] = [round(float((risk_ip_te[pi][leg[pi]] >= medium_thr).mean() * 100), 2) for pi in parts]
    if ato.any():
        ip_all, _, _ = model.heads(X[np.flatnonzero(ato)])
        takeover_report["ip_head_flags_takeover_logins_pct"] = {"Medium+": round(float((ip_all >= medium_thr).mean() * 100), 1), "High": round(float((ip_all >= high_thr).mean() * 100), 1)}

    log("[6/7] saving ...")
    report = {"rows": {"total": n, "train": a, "validation": b - a, "test": n - b},
              "bulk_pseudo_account_rows_pct": round(float(bulk.mean() * 100), 2),
              "thresholds": {"medium": medium_thr, "high": high_thr, "meaning": "risk = percentile among normal logins, per head; final risk = max of heads"},
              "ip_head": ip_report, "takeover_head": takeover_report,
              "feedback_rows_used": 0 if feedback_rows is None else int(len(feedback_rows))}
    version = time.strftime("%Y%m%d-%H%M%S")
    model.meta = {"version": version, "trained_rows": n, "features": FEATURES, "report": report}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"model_{version}.joblib"; model.save(path); shutil.copy(path, out / "latest.joblib")
    (out / f"report_{version}.json").write_text(json.dumps(report, indent=2)); (out / "latest_report.json").write_text(json.dumps(report, indent=2))
    log(f"[7/7] done in {time.time()-t0:.0f}s -> {path}")
    return report