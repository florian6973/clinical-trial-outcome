# Advisor revision publication analysis

## Locked cohort

- Mapping rows: 792,091
- Trials: 73,427
- Unique trial-title endpoint records: 467,903
- Distinct title strings: 390,228
- Normalized outcome strings: 243,916
- Outcome categories: 21
- Disease domains: 24

The mapping CSV is the cohort authority. Trial metadata is joined only to add
phase and year. Older 75,680-trial research artifacts are not blended into the
73,427-trial mapping cohort.

## Analysis units

1. Overall category frequency: unique NCT ID + raw outcome title.
2. Category-domain association: unique NCT ID + raw outcome title + disease domain.
3. Phase/time prevalence: unique NCT ID + outcome category presence.
4. Lexical complexity: unique raw outcome-title string.

The chi-square association is descriptive because trials can contribute
multiple endpoints and domains. Cramer's V measures association strength, and
Pearson standardized residuals identify cells driving the association; the
p-value is not cluster-robust.

## Leading categories

| outcome_category      |   n_endpoint_records |   pct_endpoint_records |
|:----------------------|---------------------:|-----------------------:|
| Adverse Events/Safety |                67278 |                  14.38 |
| Biomarkers/Lab Values |                66122 |                  14.13 |
| Quality of Life/PRO   |                44532 |                   9.52 |
| Pharmacokinetics      |                38886 |                   8.31 |
| Symptoms/Pain         |                31741 |                   6.78 |

## Most common semantic signatures

|   rank | semantic_pattern                                                                     |   n_endpoint_records |   pct_endpoint_records |
|-------:|:-------------------------------------------------------------------------------------|---------------------:|-----------------------:|
|      1 | object_only                                                                          |                94360 |                  20.17 |
|      2 | object+measurement_operator                                                          |                60734 |                  12.98 |
|      3 | object+measurement_operator+participant_population                                   |                57500 |                  12.29 |
|      4 | object+temporal_expression+measurement_operator+baseline_or_comparator+explicit_unit |                25840 |                   5.52 |
|      5 | object+temporal_expression+measurement_operator                                      |                19192 |                   4.10 |
|      6 | object+temporal_expression+measurement_operator+explicit_unit+participant_population |                18528 |                   3.96 |
|      7 | object+temporal_expression+measurement_operator+explicit_unit                        |                17275 |                   3.69 |
|      8 | object+measurement_operator+baseline_or_comparator                                   |                16891 |                   3.61 |

## Largest absolute category-domain residuals

| disease_domain                  | outcome_category            |   within_domain_pct |   standardized_residual |
|:--------------------------------|:----------------------------|--------------------:|------------------------:|
| Mental illness                  | Mental Health Outcomes      |               32.00 |                  247.16 |
| Neoplastic disease              | Survival/Time-to-Event      |               20.61 |                  207.07 |
| Infection                       | Immunogenicity              |               13.97 |                  153.44 |
| Disorder of skin                | Disease Activity Scores     |               20.13 |                  147.04 |
| Disorder of endocrine system    | Biomarkers/Lab Values       |               38.46 |                  136.99 |
| Infection                       | Microbiological/Virological |                6.66 |                  136.13 |
| Situation with explicit context | Pharmacokinetics            |               29.25 |                  133.13 |
| Neoplastic disease              | Response Rates              |               15.59 |                  120.94 |

## Largest period increases

| outcome_category               |   early_pct_trials_with_category |   recent_pct_trials_with_category |   delta_percentage_points |
|:-------------------------------|---------------------------------:|----------------------------------:|--------------------------:|
| Quality of Life/PRO            |                            19.04 |                             30.64 |                     11.60 |
| Adverse Events/Safety          |                            38.49 |                             46.95 |                      8.46 |
| Survival/Time-to-Event         |                            12.63 |                             21.00 |                      8.37 |
| Treatment Adherence            |                             6.23 |                             13.73 |                      7.49 |
| Hospitalization/Healthcare Use |                             4.49 |                              9.94 |                      5.46 |

## Largest period decreases

| outcome_category            |   early_pct_trials_with_category |   recent_pct_trials_with_category |   delta_percentage_points |
|:----------------------------|---------------------------------:|----------------------------------:|--------------------------:|
| Biomarkers/Lab Values       |                            28.54 |                             25.84 |                     -2.70 |
| Disease Activity Scores     |                             9.08 |                              7.63 |                     -1.44 |
| Symptoms/Pain               |                            20.73 |                             19.56 |                     -1.17 |
| Microbiological/Virological |                             4.24 |                              3.52 |                     -0.72 |
| Remission/Cure              |                             2.96 |                              2.49 |                     -0.46 |

## Reproducibility notes

- Source hashes and modification times are in `source_manifest.csv`.
- Semantic rules and regexes are in `semantic_pattern_definitions.csv`.
- All ordering, tie-breaking, sampling, and year/phase windows are deterministic.
- The casebook is descriptive and explicitly separated from formal validation.
