"""
Unit tests for the trickiest, most error-prone parts of the pipeline.
Run:  pytest -q     (from the repo root)

These three are exactly the things that "trip up most people" per the brief:
timezones, dedup/idempotency, and never silently dropping an outlier.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (build_patient, flag_gait_outliers, to_local,  # noqa: E402
                      watch_daily)


# --------------------------------------------------------------------------- #
# 1. TIMEZONE BUCKETING — a UTC instant belonging to the PREVIOUS local day.
# --------------------------------------------------------------------------- #
def test_timezone_bucketing_previous_local_day():
    # 2025-06-02T02:00:00Z. In New York (EDT, -04:00) this is 2025-06-01 22:00
    # local -> it must bucket to 2025-06-01, NOT the UTC date 2025-06-02.
    df = pd.DataFrame({"timestamp": ["2025-06-02T02:00:00+00:00"],
                       "hr": [60], "hrv": [40], "spo2": [0.97],
                       "recovery": [50], "strain": [10]})
    local = to_local(df, "America/New_York", wake_attribution=False)
    assert local["day"].iloc[0] == "2025-06-01", local["day"].iloc[0]

    # The ring's buggy +02:00 stamp for a Tokyo patient must still land on the
    # correct Tokyo local day, proving we parse the absolute instant, not the
    # misleading offset. 2025-06-01T16:00:00+02:00 == 2025-06-01T23:00 JST.
    ring = pd.DataFrame({"timestamp": ["2025-06-01T16:00:00+02:00"],
                         "hr": [55], "hrv": [60], "sleep_score": [80]})
    r = to_local(ring, "Asia/Tokyo", wake_attribution=True)
    # 23:00 local, hour>=18 -> attributed to the WAKE date 2025-06-02
    assert r["day"].iloc[0] == "2025-06-02", r["day"].iloc[0]


# --------------------------------------------------------------------------- #
# 2. DEDUPLICATION — duplicated raw rows must not change the daily aggregate.
# --------------------------------------------------------------------------- #
def test_dedup_is_idempotent():
    base = pd.DataFrame({
        "timestamp": ["2025-06-01T01:00:00+00:00", "2025-06-01T02:00:00+00:00"],
        "hr": [60, 62], "hrv": [40, 42], "spo2": [0.97, 0.96],
        "recovery": [50, 50], "strain": [10, 10],
    })
    doubled = pd.concat([base, base], ignore_index=True)      # exact duplicates

    single = watch_daily(base, "Europe/London")
    deduped = watch_daily(doubled.drop_duplicates().reset_index(drop=True),
                          "Europe/London")
    pd.testing.assert_frame_equal(
        single.reset_index(drop=True), deduped.reset_index(drop=True))


# --------------------------------------------------------------------------- #
# 3. OUTLIER FLAGGING — an impossible spike is FLAGGED (and nulled), not dropped.
# --------------------------------------------------------------------------- #
def test_impossible_spike_flagged_not_dropped():
    insole = pd.DataFrame({
        "patient_id": ["PX", "PX"],
        "local_date": ["2025-06-01", "2025-06-02"],
        "steps": [8000, 92000],          # second row is physiologically impossible
        "cadence": [110.0, 112.0],
        "gait_symmetry": [0.95, 0.94],
    })
    flags = flag_gait_outliers(insole)
    assert "2025-06-02" in flags
    o = flags["2025-06-02"][0]
    assert o.metric == "steps" and o.action == "nulled"

    # And in a full summary, that day still EXISTS (not dropped); steps is null
    # but the outlier is visible in flags.
    summaries = build_patient("PX", "Europe/London", insole,
                              pd.DataFrame(columns=["patient_id", "timestamp",
                                                    "hr", "hrv", "sleep_score"]),
                              pd.DataFrame(columns=["patient_id", "timestamp", "hr",
                                                    "hrv", "spo2", "recovery", "strain"]),
                              computed_at="t")
    day2 = [s for s in summaries if s.date == "2025-06-02"][0]
    assert day2.gait.steps is None                       # nulled, not a silent NaN
    assert any(o.metric == "steps" for o in day2.flags.outliers)  # still visible
    assert day2.gait.dataQuality == "partial"
