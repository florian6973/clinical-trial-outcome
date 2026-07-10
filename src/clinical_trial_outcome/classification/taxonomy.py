"""Locked analytic taxonomies used by the manuscript results."""

from __future__ import annotations

OUTCOME_CATEGORIES: tuple[str, ...] = (
    "Adverse Events/Safety",
    "Biomarkers/Lab Values",
    "Quality of Life/PRO",
    "Pharmacokinetics",
    "Symptoms/Pain",
    "Response Rates",
    "Survival/Time-to-Event",
    "Vital Signs/Physical Measures",
    "Mental Health Outcomes",
    "Functional Status",
    "Imaging Outcomes",
    "Disease Activity Scores",
    "Immunogenicity",
    "Treatment Adherence",
    "Hospitalization/Healthcare Use",
    "Disease Control",
    "Mortality",
    "Microbiological/Virological",
    "Other Clinical Endpoints",
    "Remission/Cure",
    "Dose-Finding",
)

COA_TYPES: tuple[str, ...] = (
    "Patient-Reported Outcome (PRO)",
    "Clinician-Reported Outcome (ClinRO)",
    "Observer-Reported Outcome (ObsRO)",
    "Performance Outcome (PerfO)",
)

if len(OUTCOME_CATEGORIES) != 21 or len(set(OUTCOME_CATEGORIES)) != 21:
    raise RuntimeError("The paper contract requires exactly 21 unique outcome categories")
if len(COA_TYPES) != 4 or len(set(COA_TYPES)) != 4:
    raise RuntimeError("The paper contract requires exactly four unique COA types")


def validate_outcome_category(value: str) -> str:
    if value not in OUTCOME_CATEGORIES:
        raise ValueError(f"Unknown outcome category: {value!r}")
    return value


def validate_coa_type(value: str) -> str:
    if value not in COA_TYPES:
        raise ValueError(f"Unknown COA type: {value!r}")
    return value
