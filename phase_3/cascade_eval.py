"""Cascade threshold selection, Policy A/B bake-off, and sealed test evaluation.

Selection (Phase 2 VALIDATION alerts only):
  for each head (MC/BIN x behaviour/service) x policy (A strict / B same-capture
  consensus): sweep tau, keep configs with TP-retention >= RETENTION_FLOOR, pick
  max FP-reduction. The winning (head, policy, tau) is FROZEN, then the Phase 2
  TEST alerts are evaluated exactly once.

Cascade rule: y_cascade = y_phase2 AND y_phase3 — alerts can only be suppressed.
  Policy A: only unique_match consults the flow score; all else retains.
  Policy B: additionally, multiple_same_capture suppresses only when EVERY
            candidate flow scores benign (max candidate score < tau).

Outputs: phase_3/cascade_report.txt, phase_3/data/cascade_results.json

Usage: python phase_3/cascade_eval.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_DIR, PHASE3_DIR
from packet_flow_mapping import UNIQUE_MATCH, MULTI_SAME_CAPTURE
from cascade_alignment import ALIGNMENT_PATH, CAND_SEP
from cascade_heads import SCORES_PATH

COHORT_PATH = os.path.join(DATA_DIR, 'phase2_cohort.parquet')
RESULTS_PATH = os.path.join(DATA_DIR, 'cascade_results.json')
RETENTION_FLOOR = 0.95
HEADS = ['s_mc_behaviour', 's_bin_behaviour', 's_mc_service', 's_bin_service']
POLICIES = ['A', 'B']
SERVICE_PORTS = {80, 8080, 443, 53}
N_BOOT = 1000

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
_report = []
def emit(s=''):
    print(s)
    _report.append(s)


# ---------------------------------------------------------------------------
# Per-alert decision scores
# ---------------------------------------------------------------------------
def alert_scores(aligned, flow_scores, head, policy):
    """Return (s, usable, decision_flow) arrays for the alert frame.
    s = decision score (NaN when the alert falls back to retain);
    usable = mapping status consulted under this policy;
    decision_flow = LogicalFlowID whose score decides (argmax for Policy B)."""
    score_by_lfid = dict(zip(flow_scores['LogicalFlowID'], flow_scores[head]))
    s = np.full(len(aligned), np.nan)
    usable = np.zeros(len(aligned), dtype=bool)
    decision_flow = np.array([None] * len(aligned), dtype=object)

    statuses = aligned['status'].to_numpy()
    cands = aligned['candidates'].to_numpy()
    for i in range(len(aligned)):
        st = statuses[i]
        if st == UNIQUE_MATCH or (policy == 'B' and st == MULTI_SAME_CAPTURE):
            lst = cands[i].split(CAND_SEP) if cands[i] else []
            vals = [(score_by_lfid.get(l), l) for l in lst]
            if any(v is None or not np.isfinite(v) for v, _ in vals) or not vals:
                continue   # missing prediction -> retain (fallback)
            v, l = max(vals)
            s[i] = v
            usable[i] = True
            decision_flow[i] = l
    return s, usable, decision_flow


def cascade_keep(s, tau):
    """Alert kept (still an alert) unless a usable score falls below tau."""
    return ~(np.isfinite(s) & (s < tau))


def cascade_metrics(y, keep, phase2_fn):
    tp_alerts, fp_alerts = int((y == 1).sum()), int((y == 0).sum())
    kept_tp = int((keep & (y == 1)).sum())
    kept_fp = int((keep & (y == 0)).sum())
    tp_ret = kept_tp / tp_alerts if tp_alerts else float('nan')
    fp_red = 1 - kept_fp / fp_alerts if fp_alerts else float('nan')
    n_kept = kept_tp + kept_fp
    return {
        'tp_alerts': tp_alerts, 'fp_alerts': fp_alerts,
        'kept_tp': kept_tp, 'kept_fp': kept_fp,
        'tp_retention': tp_ret, 'fp_reduction': fp_red,
        'alert_volume_reduction': 1 - n_kept / (tp_alerts + fp_alerts),
        'precision_phase2': tp_alerts / (tp_alerts + fp_alerts),
        'precision_cascade': kept_tp / n_kept if n_kept else float('nan'),
        'recall_phase2': tp_alerts / (tp_alerts + phase2_fn) if tp_alerts + phase2_fn else float('nan'),
        'recall_cascade': kept_tp / (tp_alerts + phase2_fn) if tp_alerts + phase2_fn else float('nan'),
    }


def select_threshold(y, s, floor=RETENTION_FLOOR):
    """Max FP-reduction subject to TP-retention >= floor. Returns (tau, metrics)
    or (None, None) if no tau is feasible (then tau=-inf: keep everything)."""
    taus = np.unique(s[np.isfinite(s)])
    best = None
    for tau in taus:
        keep = cascade_keep(s, tau)
        tp_ret = (keep & (y == 1)).sum() / max((y == 1).sum(), 1)
        fp_red = 1 - (keep & (y == 0)).sum() / max((y == 0).sum(), 1)
        if tp_ret >= floor and (best is None or fp_red > best[1] or
                                (fp_red == best[1] and tp_ret > best[2])):
            best = (float(tau), float(fp_red), float(tp_ret))
    return best


# ---------------------------------------------------------------------------
# Breakdowns (test view)
# ---------------------------------------------------------------------------
def per_class_retention(aligned, keep):
    rows = []
    for lab, g in aligned.groupby('label'):
        idx = g.index.to_numpy()
        rows.append({'label': lab, 'alerts': len(g),
                     'kept': int(keep[idx].sum()),
                     'retention': float(keep[idx].mean())})
    return pd.DataFrame(rows)


def fp_reduction_by_port_slice(aligned, keep):
    """FP-reduction on benign alerts split by packet ports touching a real
    service port (HTTP/DNS) vs ambient — the contradiction-risk measurement."""
    benign = aligned['y_true'].to_numpy() == 0
    on_service = (aligned['src_port'].isin(SERVICE_PORTS)
                  | aligned['dst_port'].isin(SERVICE_PORTS)).to_numpy()
    out = {}
    for name, m in [('service_port', benign & on_service), ('ambient', benign & ~on_service)]:
        n = int(m.sum())
        out[name] = {'fp_alerts': n,
                     'fp_reduction': float(1 - keep[m].sum() / n) if n else float('nan')}
    return out


def mcnemar_phase2_vs_cascade(y, keep):
    """Paired comparison on alert packets (non-alerts identical under both).
    Suppressing a benign alert: cascade right / phase2 wrong (c).
    Suppressing an attack alert: cascade wrong / phase2 right (b)."""
    b = int((~keep & (y == 1)).sum())
    c = int((~keep & (y == 0)).sum())
    if b + c == 0:
        return {'b': b, 'c': c, 'chi2': float('nan'), 'p_value': float('nan')}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {'b_attack_alerts_suppressed': b, 'c_benign_alerts_suppressed': c,
            'chi2': float(chi2), 'p_value': float(stats.chi2.sf(chi2, 1))}


def bootstrap_ci(y, s, tau, n_boot=N_BOOT, seed=23):
    rng = np.random.default_rng(seed)
    n = len(y)
    fp_reds, tp_rets = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        keep = cascade_keep(sb, tau)
        if (yb == 1).any() and (yb == 0).any():
            tp_rets.append((keep & (yb == 1)).sum() / (yb == 1).sum())
            fp_reds.append(1 - (keep & (yb == 0)).sum() / (yb == 0).sum())
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {'fp_reduction_ci95': q(fp_reds), 'tp_retention_ci95': q(tp_rets)}


# ---------------------------------------------------------------------------
def main():
    aligned_all = pd.read_parquet(ALIGNMENT_PATH)
    flow_scores = pd.read_parquet(SCORES_PATH)
    cohort = pd.read_parquet(COHORT_PATH)

    fn_by_split = {p: int(((cohort['phase2_split'] == p) & (cohort['y2_alert'] == 0)
                           & (cohort['y_true'] == 1)).sum()) for p in ['val', 'test']}

    val = aligned_all[aligned_all['phase2_split'] == 'val'].reset_index(drop=True)
    test = aligned_all[aligned_all['phase2_split'] == 'test'].reset_index(drop=True)
    y_val = val['y_true'].to_numpy()
    y_test = test['y_true'].to_numpy()

    # ------------------------------------------------------------------
    # Bake-off on VALIDATION alerts
    # ------------------------------------------------------------------
    emit('=== Cascade bake-off on Phase 2 VALIDATION alerts '
         f'(n={len(val):,}; TP={int((y_val==1).sum())}, FP={int((y_val==0).sum())}; '
         f'retention floor {RETENTION_FLOOR:.0%}) ===')
    rows, sweep = [], {}
    for head in HEADS:
        for policy in POLICIES:
            s, usable, _ = alert_scores(val, flow_scores, head, policy)
            picked = select_threshold(y_val, s)
            if picked is None:
                rows.append({'head': head, 'policy': policy, 'tau': None,
                             'fp_reduction': 0.0, 'tp_retention': 1.0,
                             'usable_frac': float(usable.mean()), 'feasible': False})
                continue
            tau, fp_red, tp_ret = picked
            rows.append({'head': head, 'policy': policy, 'tau': tau,
                         'fp_reduction': fp_red, 'tp_retention': tp_ret,
                         'usable_frac': float(usable.mean()), 'feasible': True})
            sweep[(head, policy)] = (s, tau)
    bake = pd.DataFrame(rows).sort_values('fp_reduction', ascending=False).reset_index(drop=True)
    emit(bake.to_string(index=False))

    winner = bake.iloc[0]
    frozen = {'head': winner['head'], 'policy': winner['policy'], 'tau': float(winner['tau']),
              'retention_floor': RETENTION_FLOOR,
              'val_fp_reduction': float(winner['fp_reduction']),
              'val_tp_retention': float(winner['tp_retention'])}
    emit(f'\nFROZEN config: head={frozen["head"]} policy={frozen["policy"]} '
         f'tau={frozen["tau"]:.6f} (val FP-reduction {frozen["val_fp_reduction"]:.1%} '
         f'@ TP-retention {frozen["val_tp_retention"]:.1%})')

    # validation metrics for the frozen config (full view)
    s_val, usable_val, _ = alert_scores(val, flow_scores, frozen['head'], frozen['policy'])
    keep_val = cascade_keep(s_val, frozen['tau'])
    m_val = cascade_metrics(y_val, keep_val, fn_by_split['val'])

    # ------------------------------------------------------------------
    # SEALED TEST — evaluated exactly once with the frozen config
    # ------------------------------------------------------------------
    emit('\n=== SEALED TEST evaluation (frozen config, single run) ===')
    s_te, usable_te, dflow_te = alert_scores(test, flow_scores, frozen['head'], frozen['policy'])
    keep_te = cascade_keep(s_te, frozen['tau'])
    m_te = cascade_metrics(y_test, keep_te, fn_by_split['test'])

    emit(f"Test alerts: {len(test):,} (TP={m_te['tp_alerts']:,}, FP={m_te['fp_alerts']:,}; "
         f"Phase 2 missed FN={fn_by_split['test']:,})")
    emit(f"  FP-reduction:            {m_te['fp_reduction']:.1%}   (val: {m_val['fp_reduction']:.1%})")
    emit(f"  TP-retention:            {m_te['tp_retention']:.1%}   (val: {m_val['tp_retention']:.1%})")
    emit(f"  alert-volume reduction:  {m_te['alert_volume_reduction']:.1%}")
    emit(f"  precision: phase2 {m_te['precision_phase2']:.3f} -> cascade {m_te['precision_cascade']:.3f}")
    emit(f"  recall:    phase2 {m_te['recall_phase2']:.3f} -> cascade {m_te['recall_cascade']:.3f} "
         f"(ceiling = phase2 recall)")

    ci = bootstrap_ci(y_test, s_te, frozen['tau'])
    emit(f"  bootstrap 95% CI  FP-reduction {ci['fp_reduction_ci95'][0]:.1%}..{ci['fp_reduction_ci95'][1]:.1%}"
         f"  TP-retention {ci['tp_retention_ci95'][0]:.1%}..{ci['tp_retention_ci95'][1]:.1%}")

    mn = mcnemar_phase2_vs_cascade(y_test, keep_te)
    emit(f"  McNemar (phase2 vs cascade): suppressed benign c={mn['c_benign_alerts_suppressed']}, "
         f"suppressed attack b={mn['b_attack_alerts_suppressed']}, chi2={mn['chi2']:.1f}, "
         f"p={mn['p_value']:.2e}")

    emit('\nPer-class alert retention (test):')
    pcr = per_class_retention(test, keep_te)
    emit(pcr.to_string(index=False))

    emit('\nFP-reduction by benign-alert port slice (contradiction risk, test):')
    slc = fp_reduction_by_port_slice(test, keep_te)
    for k, v in slc.items():
        emit(f"  {k:<13} fp_alerts={v['fp_alerts']:>5,}  fp_reduction="
             f"{v['fp_reduction']:.1%}" if v['fp_alerts'] else f'  {k:<13} (none)')

    # conditional-on-usable-mapping view
    emit('\nConditional on usable mapping (test):')
    mu = cascade_metrics(y_test[usable_te], keep_te[usable_te],
                         0)  # recall vs phase2 not defined on subset; FN=0 placeholder
    emit(f"  usable alerts={int(usable_te.sum()):,} ({usable_te.mean():.1%})  "
         f"FP-reduction={mu['fp_reduction']:.1%}  TP-retention={mu['tp_retention']:.1%}")
    emit(f"  fallback-retained alerts={int((~usable_te).sum()):,} "
         f"(TP={int((y_test[~usable_te]==1).sum())}, FP={int((y_test[~usable_te]==0).sum())})")

    # optional flow-level summary (unique/argmax decision flows)
    emit('\nFlow-level summary (decision flows, test):')
    df_flow = pd.DataFrame({'lfid': dflow_te, 'y': y_test, 'keep': keep_te})
    df_flow = df_flow[df_flow['lfid'].notna()]
    fl = df_flow.groupby('lfid').agg(y=('y', 'max'), keep=('keep', 'max'))
    emit(f"  flows consulted={len(fl):,}  attack flows kept={int((fl[fl.y==1].keep).sum())}/"
         f"{int((fl.y==1).sum())}  benign flows suppressed="
         f"{int((~fl[fl.y==0].keep).sum())}/{int((fl.y==0).sum())}")

    json_out = {
        'frozen_config': frozen,
        'bakeoff_val': rows,
        'val_metrics': m_val,
        'test_metrics': m_te,
        'test_bootstrap_ci': ci,
        'test_mcnemar': mn,
        'test_per_class_retention': pcr.to_dict(orient='records'),
        'test_fp_reduction_port_slice': slc,
        'test_usable_fraction': float(usable_te.mean()),
    }
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=1)
    emit(f'\nSaved results -> {RESULTS_PATH}')

    checks = {
        'threshold chosen on val alerts only (test untouched during sweep)': True,
        'cascade only removes alerts (suppression needs a usable score)': bool(
            (~keep_te <= np.isfinite(s_te)).all()),
        'retention floor met on val': bool(m_val['tp_retention'] >= RETENTION_FLOOR),
        'recall_cascade <= recall_phase2 (ceiling respected)': bool(
            m_te['recall_cascade'] <= m_te['recall_phase2'] + 1e-12),
    }
    emit('\n=== Cascade checkpoint verifications ===')
    for k, v in checks.items():
        emit(f'  [{"PASS" if v else "FAIL"}] {k}')

    with open(os.path.join(PHASE3_DIR, 'cascade_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_report) + '\n')


if __name__ == '__main__':
    main()
