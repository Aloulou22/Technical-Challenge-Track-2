"""
generate_data.py  —  Part A: synthetic messy data generator
============================================================

Generates realistic, INTENTIONALLY messy multi-source sensor data for a small
cohort of fake patients, so the fusion/cleaning pipeline (Part B) has something
real to chew on.

Three sources, three different "personalities" of mess:

  insole  (ground truth for gait)   -> data/raw/insole.csv
      cols: patient_id, local_date, steps, cadence, gait_symmetry
      Stored ALREADY bucketed to patient-local day (that's how a gait platform
      exports). So there is deliberately NO timezone ambiguity here — the insole
      is the day anchor. Mess: non-wear GAPS (missing dates) + impossible spikes.

  wearable_a_ring   -> data/raw/wearable_a_ring.csv
      cols: patient_id, timestamp, hr, hrv, sleep_score
      ~5-min samples across the sleep window. Timestamps are written with a
      WRONG fixed +02:00 offset for every patient regardless of their real
      timezone (a classic firmware bug). The pipeline must convert to each
      patient's true local day to bucket a night correctly. Mess: missing nights.

  wearable_b_watch  -> data/raw/wearable_b_watch.csv
      cols: patient_id, timestamp, hr, hrv, spo2, recovery, strain
      30-min samples across 24h, timestamps in correct UTC. Mess:
        - spo2 stored as a FRACTION (0.90-0.99) instead of a percentage (UNIT BUG)
        - exact DUPLICATE rows
        - one full SENSOR-DRIFT day (hr~200, spo2~0.6, hrv~2) per patient
        - hrv systematically ~20% off from the ring (cross-source DISAGREEMENT)

Everything is driven by a single fixed SEED and a fixed ANCHOR_DATE, so running
this twice produces byte-identical CSVs (idempotency starts here, at the source).

Run:  python generate_data.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Reproducibility knobs
# --------------------------------------------------------------------------- #
SEED = 42
ANCHOR_DATE = "2025-06-30"   # last day of the 30-day window (fixed, not "today")
N_DAYS = 30
RAW_DIR = os.path.join("data", "raw")

# The ring firmware bug: it stamps everything at this offset for ALL patients.
RING_BUG_OFFSET = timezone(timedelta(hours=2))   # +02:00

# --------------------------------------------------------------------------- #
# Cohort. Deliberately spread across timezones so the ring's fixed +02:00 bug
# lands differently for each patient. Age bands are for the PHI-safe narrative
# later (Part C) — never a DOB.
# --------------------------------------------------------------------------- #
PATIENTS = [
    # id,       real local tz,        age_band,       baseline physiology
    ("P001", "Europe/Paris",        "adult_35_44", dict(rhr=58, hrv=55, steps=8500)),
    ("P002", "America/New_York",    "adult_55_64", dict(rhr=66, hrv=38, steps=5200)),
    ("P003", "Asia/Tokyo",          "adult_25_34", dict(rhr=52, hrv=72, steps=11000)),
    ("P004", "Europe/London",       "adult_65_74", dict(rhr=71, hrv=31, steps=3800)),
]


def _rng_for(patient_id: str) -> np.random.Generator:
    """Deterministic per-patient generator so patients are independent but stable."""
    return np.random.default_rng(SEED + int(patient_id[1:]))


def _dates(anchor: str, n: int) -> list[datetime]:
    end = datetime.strptime(anchor, "%Y-%m-%d").date()
    return [end - timedelta(days=i) for i in range(n - 1, -1, -1)]


# --------------------------------------------------------------------------- #
# Source 1: INSOLE  (ground truth for gait, per local day)
# --------------------------------------------------------------------------- #
def gen_insole(patient_id: str, base: dict, dates) -> pd.DataFrame:
    rng = _rng_for(patient_id)
    rows = []

    # Pick 3 non-wear (gap) days and 2 impossible-spike days, deterministically.
    day_idx = np.arange(len(dates))
    nonwear = set(rng.choice(day_idx, size=3, replace=False).tolist())
    spike_days = set(rng.choice([i for i in day_idx if i not in nonwear],
                                size=2, replace=False).tolist())

    for i, d in enumerate(dates):
        if i in nonwear:
            # Non-wear day: emit NO row at all (a true gap the pipeline must flag).
            continue

        steps = int(rng.normal(base["steps"], base["steps"] * 0.20))
        steps = max(steps, 0)
        cadence = float(rng.normal(112, 6))            # steps/min while walking
        symmetry = float(np.clip(rng.normal(0.94, 0.03), 0.70, 1.0))  # 1.0 = perfect

        if i in spike_days:
            # Impossible physiological spike (sensor fault), NOT dropped here —
            # the generator injects it; the pipeline must detect & flag it.
            steps = int(rng.choice([92000, 88000]))
            cadence = float(rng.choice([410.0, 385.0]))

        rows.append(dict(patient_id=patient_id, local_date=d.isoformat(),
                         steps=steps, cadence=round(cadence, 1),
                         gait_symmetry=round(symmetry, 3)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Source 2: WEARABLE A — RING  (5-min sleep samples, WRONG +02:00 offset)
# --------------------------------------------------------------------------- #
def gen_ring(patient_id: str, base: dict, dates, real_tz: ZoneInfo) -> pd.DataFrame:
    rng = _rng_for(patient_id)
    rows = []

    missing_nights = set(rng.choice(np.arange(len(dates)), size=4,
                                    replace=False).tolist())

    for i, d in enumerate(dates):
        if i in missing_nights:
            continue  # ring not worn / not synced that night

        # Sleep window ~ 23:00 -> 07:00 of the NEXT calendar day, in real local time.
        night_start_local = datetime(d.year, d.month, d.day, 23, 0,
                                     tzinfo=real_tz)
        sleep_score = float(np.clip(rng.normal(78, 10), 30, 100))
        # nightly HRV baseline for this patient/night (ring's reading)
        night_hrv = float(np.clip(rng.normal(base["hrv"], 6), 10, 140))

        # 8 hours of 5-min samples -> 96 samples
        for k in range(0, 8 * 60, 5):
            t_local = night_start_local + timedelta(minutes=k)
            # Convert the true instant to what the BUGGY ring records:
            # it discards the real zone and stamps the same wall-clock at +02:00.
            true_instant_utc = t_local.astimezone(timezone.utc)
            buggy_stamp = true_instant_utc.astimezone(RING_BUG_OFFSET)

            hr = float(np.clip(rng.normal(base["rhr"] + 4, 4), 35, 120))
            hrv = float(np.clip(rng.normal(night_hrv, 5), 8, 150))
            rows.append(dict(
                patient_id=patient_id,
                timestamp=buggy_stamp.isoformat(),   # carries +02:00 always
                hr=round(hr, 1),
                hrv=round(hrv, 1),
                sleep_score=round(sleep_score, 1),   # denormalized per sample
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Source 3: WEARABLE B — WATCH  (30-min samples, correct UTC, lots of mess)
# --------------------------------------------------------------------------- #
def gen_watch(patient_id: str, base: dict, dates, real_tz: ZoneInfo) -> pd.DataFrame:
    rng = _rng_for(patient_id)
    rows = []

    drift_day = int(rng.choice(np.arange(2, len(dates) - 2)))  # one bad-sensor day

    for i, d in enumerate(dates):
        # Watch reports a per-day recovery/strain (denormalized onto each sample).
        recovery = float(np.clip(rng.normal(60, 18), 1, 100))
        strain = float(np.clip(rng.normal(11, 4), 0, 21))

        # Its nightly HRV DISAGREES with the ring: ~20% lower on average.
        night_hrv_watch = float(np.clip(rng.normal(base["hrv"] * 0.80, 6), 5, 140))

        for hh in range(0, 24):
            for mm in (0, 30):
                # Anchor the 24h day in the patient's real local time, then store UTC.
                t_local = datetime(d.year, d.month, d.day, hh, mm, tzinfo=real_tz)
                t_utc = t_local.astimezone(timezone.utc)

                if i == drift_day:
                    # SENSOR-DRIFT day: obviously broken readings all day.
                    hr = float(rng.normal(200, 5))
                    hrv = float(rng.normal(2, 0.5))
                    spo2_frac = float(np.clip(rng.normal(0.60, 0.02), 0.40, 0.70))
                else:
                    hr = float(np.clip(rng.normal(base["rhr"] + 15, 12), 40, 180))
                    hrv = float(np.clip(rng.normal(night_hrv_watch, 6), 5, 150))
                    spo2_frac = float(np.clip(rng.normal(0.97, 0.01), 0.90, 1.0))

                rows.append(dict(
                    patient_id=patient_id,
                    timestamp=t_utc.isoformat(),
                    hr=round(hr, 1),
                    hrv=round(hrv, 1),
                    spo2=round(spo2_frac, 4),   # UNIT BUG: fraction, not percent
                    recovery=round(recovery, 1),
                    strain=round(strain, 1),
                ))

    df = pd.DataFrame(rows)

    # Inject exact DUPLICATE rows: duplicate ~2% of rows, deterministically.
    n_dupes = max(1, int(len(df) * 0.02))
    dupe_idx = rng.choice(df.index.to_numpy(), size=n_dupes, replace=False)
    df = pd.concat([df, df.loc[dupe_idx]], ignore_index=True)

    # Shuffle so duplicates aren't conveniently adjacent (more realistic).
    df = df.sample(frac=1.0, random_state=SEED + int(patient_id[1:])).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    dates = _dates(ANCHOR_DATE, N_DAYS)

    insole_all, ring_all, watch_all = [], [], []
    print(f"Generating {N_DAYS} days for {len(PATIENTS)} patients "
          f"(anchor={ANCHOR_DATE}, seed={SEED})\n")

    for pid, tzname, age_band, base in PATIENTS:
        real_tz = ZoneInfo(tzname)
        insole_all.append(gen_insole(pid, base, dates))
        ring_all.append(gen_ring(pid, base, dates, real_tz))
        watch_all.append(gen_watch(pid, base, dates, real_tz))
        print(f"  {pid}  tz={tzname:<18} age_band={age_band}")

    insole = pd.concat(insole_all, ignore_index=True)
    ring = pd.concat(ring_all, ignore_index=True)
    watch = pd.concat(watch_all, ignore_index=True)

    insole.to_csv(os.path.join(RAW_DIR, "insole.csv"), index=False)
    ring.to_csv(os.path.join(RAW_DIR, "wearable_a_ring.csv"), index=False)
    watch.to_csv(os.path.join(RAW_DIR, "wearable_b_watch.csv"), index=False)

    # Also persist the cohort metadata (real tz + age band) for the pipeline.
    meta = pd.DataFrame(
        [dict(patient_id=p, timezone=tz, age_band=ab) for p, tz, ab, _ in PATIENTS]
    )
    meta.to_csv(os.path.join(RAW_DIR, "patients.csv"), index=False)

    print("\nWrote:")
    print(f"  insole.csv            {len(insole):>6} rows  (gaps + impossible spikes)")
    print(f"  wearable_a_ring.csv   {len(ring):>6} rows  (+02:00 offset bug, missing nights)")
    print(f"  wearable_b_watch.csv  {len(watch):>6} rows  (spo2 fraction, dupes, drift day)")
    print(f"  patients.csv          {len(meta):>6} rows  (real tz + age band)")


if __name__ == "__main__":
    main()
