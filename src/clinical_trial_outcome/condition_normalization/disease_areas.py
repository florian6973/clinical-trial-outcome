"""The 24 disease-domain labels used by the paper analysis artifact."""

from __future__ import annotations

DISEASE_AREAS: tuple[str, ...] = (
    "Blood disease",
    "Cardiovascular disease",
    "Clinical finding",
    "Congenital disease",
    "Disorder of body system",
    "Disorder of digestive system",
    "Disorder of ear",
    "Disorder of endocrine system",
    "Disorder of eye",
    "Disorder of haemostatic system",
    "Disorder of immune function",
    "Disorder of labour / delivery",
    "Disorder of skin",
    "Disorder of the genitourinary system",
    "Infection",
    "Injury",
    "Mental illness",
    "Musculoskeletal disorder",
    "Neoplastic disease",
    "Neurological disorder",
    "Procedure",
    "Respiratory disease",
    "Situation with explicit context",
    "Sleep disorder",
)

if len(DISEASE_AREAS) != 24 or len(set(DISEASE_AREAS)) != 24:
    raise RuntimeError("The paper contract requires exactly 24 unique disease areas")


def validate_disease_area(value: str) -> str:
    """Return *value* if it is one of the paper's 24 disease areas."""

    if value not in DISEASE_AREAS:
        raise ValueError(f"Unknown disease area: {value!r}")
    return value
