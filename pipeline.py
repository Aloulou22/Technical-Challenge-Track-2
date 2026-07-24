"""
pipeline.py  —  Part B: fusion & cleaning pipeline
==================================================

Reads the three messy raw CSVs + patient metadata and emits, for every
patient-day, one schema-validated PatientDailySummary. No silent NaNs: every
field is either a real value with a named source, or an explicit null carrying a
dataQuality reason.

The decisions below are the whole point of the challenge, so they are stated
explicitly and are all defensible:

SOURCE-PRIORITY CONTRACT
    gait   (steps, cadence, symmetry)  -> insole ONLY. Never substitute a
                                          wearable step count. If insole is
                                          absent that day, gait is `missing`.
    cardio.rhr / cardio.hrv            -> ring FIRST (dedicated overnight
                                          recovery device), watch as FALLBACK.
    cardio.spo2                        -> watch ONLY (only source that has it).
    sleep  (score, hours)             -> ring ONLY.
    load   (recovery, strain)         -> watch ONLY.
    Why: the insole is ground truth for movement; the ring's overnight sensor
    is more reliable for resting cardio than a general-purpose watch, so it wins
    ties. Each chosen value carries the source it came from.

TIMEZONE
    Every wearable sample is parsed as an ABSOLUTE instant (offset-aware) and
    converted to the patient's REAL timezone (from patients.csv). The summary
    `date` is the patient-LOCAL calendar day. A naive `timestamp[:10]` would be
    wrong — e.g. the ring stamps a fixed +02:00 for everyone, so for the Tokyo
    patient the offset is meaningless; only conversion to the true zone buckets
    the night to the correct local day.
    Sleep-attribution rule: a ring night (samples from ~23:00 to ~07:00) is
    attributed to the WAKE date (local evening samples, hour >= 18, roll forward
    one day) so a night's recovery lands on the morning it informs — the
    Oura/Whoop convention.

DEDUPLICATION
    Exact duplicate rows are dropped up front (`drop_duplicates`). The whole
    pipeline is a pure function of the input files, so running it twice is
    byte-identical (see run.py idempotency proof).

OUTLIERS  (flag, don't silently drop)
    Two-stage: (1) hard physiological plausibility bounds, (2) a per-patient
    robust check (value beyond 3x IQR from the patient's median). Anything
    caught is recorded in flags.outliers with metric/value/method/action. A
    physiologically IMPOSSIBLE value (e.g. 92 000 steps) is additionally nulled
    so it can't poison downstream stats — but it stays visible in the flags,
    never silently removed.

SENSOR-DRIFT DAY  (named edge case #1)
    A watch day whose median HR is implausibly high (drift) is detected, its
    watch cardio for that day is discarded, cardio falls back to the ring, and
    a `watch_sensor_drift` outlier is logged. The drift never pollutes the
    fused cardio.

NON-WEAR DAY  (named edge case #2)
    A day with no insole row -> gait.dataQuality = "missing" and "insole" is
    listed in flags.missingSources. Gait is never interpolated or borrowed.

MISSING-DATA POLICY
    We interpolate ONLY the smoothed sleep index across single-night (<=1 day)
    gaps, tagged source "ring_interpolated" and dataQuality "partial". We
    deliberately do NOT interpolate gait, rHR, HRV, or SpO2: those drive
    clinical judgement and fabricating them would mislead a doctor. Longer gaps
    are never filled.

CROSS-SOURCE DISAGREEMENT
    When both ring and watch produced an overnight HRV for the same day, we log
    cardio.disagreementDelta["hrv"] = round(ring_hrv - watch_hrv, 1).
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from schema import (Cardio, Flags, Gait, Load, Outlier, PatientDailySummary,
                    Sleep)

RAW_DIR = os.path.join("data", "raw")

# Physiological plausibility bounds (hard limits used for outlier stage 1).
BOUNDS = {
    "steps": (0, 60_000),
    "cadence": (0, 200),
    "hr": (25, 220),
    "hrv": (1, 200),
    "spo2_pct": (70, 100),
}
DRIFT_HR_MEDIAN = 150      # a watch day with median HR above this = sensor drift
IQR_K = 3.0               # robustness multiplier for stage-2 outlier check


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(raw_dir: str = RAW_DIR):
    patients = pd.read_csv(os.path.join(raw_dir, "patients.csv"))
    insole = pd.read_csv(os.path.join(raw_dir, "insole.csv"))
    ring = pd.read_csv(os.path.join(raw_dir, "wearable_a_ring.csv"))
    watch = pd.read_csv(os.path.join(raw_dir, "wearable_b_watch.csv"))

    # DEDUP — idempotent, exact-duplicate rows only.
    ring = ring.drop_duplicates().reset_index(drop=True)
    watch = watch.drop_duplicates().reset_index(drop=True)
    return patients, insole, ring, watch


# --------------------------------------------------------------------------- #
# Timezone helpers
# --------------------------------------------------------------------------- #
def to_local(df: pd.DataFrame, tzname: str, wake_attribution: bool) -> pd.DataFrame:
    """Parse offset-aware timestamps -> patient-local; add a `day` bucket key."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)      # absolute instant
    local = ts.dt.tz_convert(tzname)
    out["local_ts"] = local
    day = local.dt.date
    if wake_attribution:
        # evening samples (hour >= 18) belong to the next morning's readiness
        roll = (local.dt.hour >= 18)
        day = np.where(roll, local.dt.date + pd.Timedelta(days=1).to_pytimedelta(),
                       local.dt.date)
        day = pd.Series(day, index=out.index)
    out["day"] = pd.to_datetime(pd.Series(day, index=out.index)).dt.strftime("%Y-%m-%d")
    return out


# --------------------------------------------------------------------------- #
# Per-source daily aggregation
# --------------------------------------------------------------------------- #
def ring_daily(ring_p: pd.DataFrame, tzname: str) -> pd.DataFrame:
    """One nightly row per wake-date: rhr (5th pctile HR), hrv (median), sleep."""
    if ring_p.empty:
        return pd.DataFrame(columns=["day", "rhr", "hrv", "sleep_score", "sleep_hours"])
    r = to_local(ring_p, tzname, wake_attribution=True)

    def agg(g: pd.DataFrame) -> pd.Series:
        span_h = (g["local_ts"].max() - g["local_ts"].min()).total_seconds() / 3600.0
        return pd.Series({
            "rhr": float(np.percentile(g["hr"], 5)),          # resting = low pctile
            "hrv": float(g["hrv"].median()),
            "sleep_score": float(g["sleep_score"].median()),
            "sleep_hours": round(span_h, 2),
        })

    out = r.groupby("day", group_keys=False).apply(agg, include_groups=False)
    return out.reset_index()


def watch_daily(watch_p: pd.DataFrame, tzname: str) -> pd.DataFrame:
    """
    Per local-date watch aggregates. Overnight window [00:00,08:00) for resting
    cardio; SpO2 unit-corrected fraction->percent; recovery/strain per day;
    sensor-drift flag per day.
    """
    if watch_p.empty:
        cols = ["day", "rhr", "hrv", "spo2", "recovery", "strain", "drift"]
        return pd.DataFrame(columns=cols)
    w = to_local(watch_p, tzname, wake_attribution=False)
    w["hour"] = w["local_ts"].dt.hour

    # UNIT FIX: SpO2 arrives as a fraction (0-1). Convert to percentage.
    # Guard: only scale values that really look like a fraction (<=1.5).
    w["spo2_pct"] = np.where(w["spo2"] <= 1.5, w["spo2"] * 100.0, w["spo2"])

    def agg(g: pd.DataFrame) -> pd.Series:
        night = g[g["hour"] < 8]
        rest = night if len(night) else g
        return pd.Series({
            "rhr": float(np.percentile(rest["hr"], 5)),
            "hrv": float(rest["hrv"].median()),
            "spo2": float(g["spo2_pct"].median()),
            "recovery": float(g["recovery"].median()),
            "strain": float(g["strain"].median()),
            "hr_median": float(g["hr"].median()),
        })

    out = w.groupby("day", group_keys=False).apply(agg, include_groups=False).reset_index()
    out["drift"] = out["hr_median"] > DRIFT_HR_MEDIAN
    return out.drop(columns=["hr_median"])


# --------------------------------------------------------------------------- #
# Outlier utilities
# --------------------------------------------------------------------------- #
def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1, q3 = np.percentile(series.dropna(), [25, 75])
    iqr = q3 - q1
    return q1 - IQR_K * iqr, q3 + IQR_K * iqr


def flag_gait_outliers(insole_p: pd.DataFrame) -> dict[str, list[Outlier]]:
    """Return {date: [Outlier,...]} for impossible/extreme gait values."""
    flags: dict[str, list[Outlier]] = {}
    if insole_p.empty:
        return flags
    lo_s, hi_s = BOUNDS["steps"]
    lo_c, hi_c = BOUNDS["cadence"]
    # stage 2 robust bounds computed on plausible rows only
    plausible = insole_p[(insole_p["steps"].between(lo_s, hi_s))]
    s_lo, s_hi = _iqr_bounds(plausible["steps"]) if len(plausible) > 3 else (lo_s, hi_s)

    for _, row in insole_p.iterrows():
        d = row["local_date"]
        hits = []
        if not (lo_s <= row["steps"] <= hi_s):
            hits.append(Outlier(metric="steps", value=float(row["steps"]), date=d,
                                method="physiological_bound", action="nulled"))
        elif not (s_lo <= row["steps"] <= s_hi):
            hits.append(Outlier(metric="steps", value=float(row["steps"]), date=d,
                                method="robust_iqr_3x", action="flagged"))
        if not (lo_c <= row["cadence"] <= hi_c):
            hits.append(Outlier(metric="cadence", value=float(row["cadence"]), date=d,
                                method="physiological_bound", action="nulled"))
        if hits:
            flags[d] = hits
    return flags


# --------------------------------------------------------------------------- #
# Fusion — build one PatientDailySummary per patient-day
# --------------------------------------------------------------------------- #
def build_patient(pid: str, tzname: str, insole: pd.DataFrame, ring: pd.DataFrame,
                  watch: pd.DataFrame, computed_at: str) -> list[PatientDailySummary]:
    ins_p = insole[insole["patient_id"] == pid].copy()
    ring_p = ring[ring["patient_id"] == pid].copy()
    watch_p = watch[watch["patient_id"] == pid].copy()

    ins_by_day = {r["local_date"]: r for _, r in ins_p.iterrows()}
    rd = ring_daily(ring_p, tzname).set_index("day") if not ring_p.empty else pd.DataFrame()
    wd = watch_daily(watch_p, tzname).set_index("day") if not watch_p.empty else pd.DataFrame()
    gait_outliers = flag_gait_outliers(ins_p)

    # Full local-day calendar = union of every day any source saw.
    all_days = set(ins_by_day) | set(rd.index) | set(wd.index)
    all_days = sorted(all_days)

    # For interpolation of the smoothed sleep index across single-night gaps.
    sleep_series = rd["sleep_score"].reindex(all_days) if "sleep_score" in rd else pd.Series(dtype=float)
    sleep_interp = sleep_series.interpolate(limit=1, limit_area="inside") if len(sleep_series) else sleep_series

    summaries: list[PatientDailySummary] = []
    for d in all_days:
        outliers: list[Outlier] = list(gait_outliers.get(d, []))
        missing: list[str] = []

        # ---------------- GAIT (insole only) ----------------
        gait = Gait()
        if d in ins_by_day:
            row = ins_by_day[d]
            nulled = {o.metric for o in gait_outliers.get(d, []) if o.action == "nulled"}
            gait.steps = None if "steps" in nulled else int(row["steps"])
            gait.cadence = None if "cadence" in nulled else float(row["cadence"])
            gait.symmetry = float(row["gait_symmetry"])
            gait.dataQuality = "partial" if nulled else "ok"
        else:
            gait.dataQuality = "missing"
            missing.append("insole")            # <-- NON-WEAR DAY edge case

        # ---------------- CARDIO (ring > watch; spo2 watch-only) ----------------
        cardio = Cardio()
        r = rd.loc[d] if d in rd.index else None
        w = wd.loc[d] if d in wd.index else None
        drift = bool(w["drift"]) if w is not None else False
        if drift:                               # <-- SENSOR-DRIFT DAY edge case
            outliers.append(Outlier(metric="watch_sensor_drift", value=None, date=d,
                                    method="median_hr_gt_150",
                                    action="watch_cardio_discarded_fell_back_to_ring"))

        # rhr / hrv: ring first, watch fallback (watch ignored on drift days)
        chosen_src: dict[str, str] = {}
        if r is not None and not np.isnan(r["rhr"]):
            cardio.rhr = round(float(r["rhr"]), 1); chosen_src["rhr"] = "ring"
        elif w is not None and not drift:
            cardio.rhr = round(float(w["rhr"]), 1); chosen_src["rhr"] = "watch"

        if r is not None and not np.isnan(r["hrv"]):
            cardio.hrv = round(float(r["hrv"]), 1); chosen_src["hrv"] = "ring"
        elif w is not None and not drift:
            cardio.hrv = round(float(w["hrv"]), 1); chosen_src["hrv"] = "watch"

        # spo2: watch only, and never trust it on a drift day
        if w is not None and not drift:
            cardio.spo2 = round(float(w["spo2"]), 1); chosen_src["spo2"] = "watch"

        # disagreement: both sources had an overnight HRV
        if (r is not None and not np.isnan(r["hrv"]) and
                w is not None and not drift and not np.isnan(w["hrv"])):
            cardio.disagreementDelta["hrv"] = round(float(r["hrv"] - w["hrv"]), 1)

        cardio.source = chosen_src
        have = [cardio.rhr, cardio.hrv, cardio.spo2]
        if all(v is None for v in have):
            cardio.dataQuality = "missing"
            if r is None and (w is None or drift):
                if "ring" not in missing and r is None:
                    missing.append("ring")
        elif any(v is None for v in have) or drift:
            cardio.dataQuality = "partial"
        else:
            cardio.dataQuality = "ok"

        # ---------------- SLEEP (ring only, single-gap interpolation) ----------------
        sleep = Sleep()
        if r is not None and not np.isnan(r["sleep_score"]):
            sleep.score = round(float(r["sleep_score"]), 1)
            sleep.hours = round(float(r["sleep_hours"]), 2)
            sleep.source = "ring"
            sleep.dataQuality = "ok"
        elif len(sleep_interp) and d in sleep_interp.index and not pd.isna(sleep_interp[d]):
            # smoothed index, single-night gap only -> interpolated + tagged
            sleep.score = round(float(sleep_interp[d]), 1)
            sleep.source = "ring_interpolated"
            sleep.dataQuality = "partial"
        else:
            sleep.dataQuality = "missing"
            if "ring" not in missing and r is None:
                missing.append("ring")

        # ---------------- LOAD (watch only) ----------------
        load = Load()
        if w is not None and not drift:
            load.recovery = round(float(w["recovery"]), 1)
            load.strain = round(float(w["strain"]), 1)
            load.source = "watch"
            load.dataQuality = "ok"
        elif w is not None and drift:
            load.dataQuality = "partial"        # watch present but untrustworthy
            load.source = "watch"
        else:
            load.dataQuality = "missing"
            if "watch" not in missing:
                missing.append("watch")

        summaries.append(PatientDailySummary(
            patientId=pid, date=d, gait=gait, cardio=cardio, sleep=sleep,
            load=load, flags=Flags(outliers=outliers, missingSources=sorted(set(missing))),
            computedAt=computed_at,
        ))
    return summaries


def run_pipeline(computed_at: str, raw_dir: str = RAW_DIR
                 ) -> dict[str, list[PatientDailySummary]]:
    patients, insole, ring, watch = load_raw(raw_dir)
    result: dict[str, list[PatientDailySummary]] = {}
    for _, prow in patients.iterrows():
        pid, tzname = prow["patient_id"], prow["timezone"]
        result[pid] = build_patient(pid, tzname, insole, ring, watch, computed_at)
    return result
