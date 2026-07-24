"""
schema.py  —  The versioned data contract.

Everything the pipeline emits is validated against these models before it is
written, and everything the narrative function reads is this same shape. If a
field is ever a "silent NaN", pydantic will reject it here — a value is either a
real number or an explicit null tagged with a dataQuality reason.

schema_version lives on every row so a downstream consumer can migrate safely.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

Quality = Literal["ok", "partial", "missing"]


class Gait(BaseModel):
    # Insole is ALWAYS the source of truth for gait. Never substituted.
    steps: Optional[int] = None
    cadence: Optional[float] = None
    symmetry: Optional[float] = None
    dataQuality: Quality = "missing"


class Cardio(BaseModel):
    rhr: Optional[float] = None
    hrv: Optional[float] = None
    spo2: Optional[float] = None          # stored as a PERCENTAGE (e.g. 97.0)
    # which source each chosen field came from, e.g. {"rhr": "ring", "hrv": "watch"}
    source: dict[str, str] = Field(default_factory=dict)
    # logged when two sources measured the same field, e.g. {"hrv": 9.2}
    disagreementDelta: dict[str, float] = Field(default_factory=dict)
    dataQuality: Quality = "missing"


class Sleep(BaseModel):
    score: Optional[float] = None
    hours: Optional[float] = None
    source: Optional[str] = None
    dataQuality: Quality = "missing"


class Load(BaseModel):
    strain: Optional[float] = None
    recovery: Optional[float] = None
    source: Optional[str] = None
    dataQuality: Quality = "missing"


class Outlier(BaseModel):
    metric: str
    value: Optional[float] = None
    date: str
    method: str          # how it was caught
    action: str          # what we did about it (flagged / nulled / fell_back)


class Flags(BaseModel):
    outliers: list[Outlier] = Field(default_factory=list)
    missingSources: list[str] = Field(default_factory=list)


class PatientDailySummary(BaseModel):
    patientId: str
    date: str                              # patient-LOCAL day, YYYY-MM-DD
    gait: Gait
    cardio: Cardio
    sleep: Sleep
    load: Load
    flags: Flags
    schema_version: str = SCHEMA_VERSION
    computedAt: str
