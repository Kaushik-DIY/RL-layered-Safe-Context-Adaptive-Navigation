"""Industry-KPI translation + statistics layer (P2 of the 2026-07 replan).

Consumes any battery CSV with the standard episode-metric columns (headroom
probe, corner demo, Pareto sweep, S4/S5 batteries) and produces:

  1. an INDUSTRY-KPI table per group -- the same numbers a fleet engineer uses:
       throughput      missions/hour (from time-to-goal), success %
       compliance      % missions with zero stopping-distance violations
                       (ISO 3691-4 framing), collision-free %
       MTBI            missions between protective-stop interventions
                       (the AMR downtime driver), stops/mission
       stop-loss       estimated seconds per mission lost to stop-recover cycles
                       (documented model: each stop costs tau + brake + re-accel
                       ramps; an ESTIMATE -- the realized cost is already inside
                       time-to-goal, this line just attributes it)
       energy          relative to the baseline group (%), from the |a|*v*dt proxy
       smoothness      RMS jerk relative to baseline (payload-stress proxy;
                       reported relative -- the 2D proxy is not ISO-2631 weighted)
       social          personal-space intrusion s/mission

  2. PAIRED statistics vs a named baseline group (plan 4.3): Wilcoxon signed-rank
     on seed-paired episodes (falls back to Mann-Whitney U when unpaired),
     Cliff's delta effect size, bootstrap 95% CI on the mean difference.
     Underpowered claims are worse than modest claims: 'not significant' is
     printed plainly when p >= 0.05.

    python scripts/kpi_report.py experiments/results/headroom_probe.csv \
        --group supervisor --baseline always-max --filter scale=industrial
    python scripts/kpi_report.py experiments/results/corner_breach.csv \
        --group supervisor --baseline always-max
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.common.platform import load_platform  # noqa: E402

# metrics reported + tested (column -> pretty name, higher_is_better)
TEST_METRICS = [
    ("time_to_goal", "time-to-goal [s]", False),
    ("violation_steps", "violation steps", False),
    ("protective_stops", "protective stops", False),
    ("energy", "energy proxy", False),
    ("rms_jerk", "RMS jerk", False),
    ("intrusion_time", "intrusion time [s]", False),
]


def kpi_table(df: pd.DataFrame, group: str, platform: str) -> pd.DataFrame:
    p = load_platform(platform)
    # documented downtime model: a protective stop wastes the deadtime plus a
    # brake-to-zero and re-accel ramp at the platform's service rates
    t_stop = p.cbf.tau + 0.5 * p.robot.v_max / p.cbf.a_brake \
        + 0.5 * p.robot.v_max / p.robot.a_max_mpc
    rows = {}
    for g, sub in df.groupby(group):
        t = sub["time_to_goal"].dropna()
        pstops = sub["protective_stops"]
        stop_eps = (pstops > 0).mean()
        rows[g] = {
            "missions/h": 3600.0 / t.mean() if len(t) else np.nan,
            "success %": 100.0 * sub["success"].mean(),
            "ISO-clean %": 100.0 * (sub["violation_steps"] == 0).mean(),
            "collision-free %": 100.0 * (1.0 - sub["collision"].mean()),
            "stops/mission": pstops.mean(),
            "MTBI [missions]": (1.0 / stop_eps) if stop_eps > 0 else np.inf,
            "stop-loss s/mission": pstops.mean() * t_stop,
            "energy": sub["energy"].mean(),
            "RMS jerk": sub["rms_jerk"].mean(),
            "intrusion s": sub["intrusion_time"].mean(),
        }
    out = pd.DataFrame(rows).T
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta in [-1, 1]: P(a>b) - P(a<b) (nonparametric effect size)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def bootstrap_ci(diff: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(diff, len(diff), replace=True))
             for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(df: pd.DataFrame, group: str, baseline: str, other: str,
            pair_on: str | None) -> pd.DataFrame:
    a = df[df[group] == other]
    b = df[df[group] == baseline]
    rows = []
    for col, name, _ in TEST_METRICS:
        if col not in df:
            continue
        av, bv = a[col].astype(float).values, b[col].astype(float).values
        paired = False
        if pair_on and pair_on in df:
            am = a.set_index(pair_on)[col].astype(float)
            bm = b.set_index(pair_on)[col].astype(float)
            common = am.index.intersection(bm.index)
            if len(common) >= 5:
                av, bv, paired = am[common].values, bm[common].values, True
        # drop incomplete pairs/rows (time_to_goal is NaN on failed episodes)
        ok = np.isfinite(av) & np.isfinite(bv) if paired else None
        if paired:
            av, bv = av[ok], bv[ok]
        else:
            av, bv = av[np.isfinite(av)], bv[np.isfinite(bv)]
        if len(av) < 5 or len(bv) < 5:
            continue
        if paired:
            d = av - bv
            if np.allclose(d, 0):
                p_val = 1.0
            else:
                p_val = float(stats.wilcoxon(av, bv, zero_method="zsplit").pvalue)
            lo, hi = bootstrap_ci(d)
        else:
            p_val = float(stats.mannwhitneyu(av, bv).pvalue) \
                if not (np.allclose(av.mean(), bv.mean()) and av.std() + bv.std() == 0) else 1.0
            rng = np.random.default_rng(0)
            boots = [rng.choice(av, len(av)).mean() - rng.choice(bv, len(bv)).mean()
                     for _ in range(5000)]
            lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        rows.append({
            "metric": name,
            f"{other} mean": av.mean(), f"{baseline} mean": bv.mean(),
            "diff": av.mean() - bv.mean(), "CI95": f"[{lo:+.3f}, {hi:+.3f}]",
            "p": p_val, "cliffs_d": cliffs_delta(av, bv),
            "test": "wilcoxon(paired)" if paired else "mann-whitney",
            "verdict": ("significant" if p_val < 0.05 else "NOT significant"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="battery CSV with episode metrics")
    ap.add_argument("--group", required=True, help="grouping column (e.g. supervisor)")
    ap.add_argument("--baseline", required=True, help="baseline group value")
    ap.add_argument("--pair-on", default="episode",
                    help="column pairing episodes across groups (seed pairing)")
    ap.add_argument("--platform", default="industrial")
    ap.add_argument("--filter", default=None,
                    help="col=value row filter, e.g. scale=industrial")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.filter:
        col, val = args.filter.split("=", 1)
        df = df[df[col].astype(str) == val]
    if args.group not in df:
        raise SystemExit(f"no column '{args.group}' in {args.csv}")

    print(f"== industry KPIs ({Path(args.csv).name}"
          + (f", {args.filter}" if args.filter else "") + ") ==")
    with pd.option_context("display.width", 160, "display.float_format",
                           lambda v: f"{v:8.2f}"):
        print(kpi_table(df, args.group, args.platform).to_string())

    groups = [g for g in df[args.group].unique() if g != args.baseline]
    for other in groups:
        print(f"\n== {other} vs {args.baseline} "
              f"(paired on '{args.pair_on}' where possible) ==")
        rep = compare(df, args.group, args.baseline, other, args.pair_on)
        with pd.option_context("display.width", 200, "display.float_format",
                               lambda v: f"{v:8.3f}"):
            print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
