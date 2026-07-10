# Structuring and Normalizing Clinical Trial Outcomes with Large Language Models

This repo contains the code for the project of clinical trial outcome normalization and trend analysis.
The final output with normalized conditions and outcomes for each trial can be downloaded [here](<./Outcome x Condition x Trial Mapping/trials.csv>).

The repo is subdivided in several subfolders:
- SNOMED Outcome Normalization: code to directly map outcomes to SNOMED concepts
- Outcome Structuring: code to implement algorithm 1 of the paper
- Term to Concept Normalization: code to implement algorithm 2 of the paper
- Condition Mapping: code to map conditions to SNOMED high-level categories (algorithm 3 of the paper)
- Final Analysis: code to perform the trend analysis
- Outcome x Condition x Trial Mapping: folder with the final output file

Required packages are listed in `requirements.txt`. Experiments were run on a Linux cluster with Python 3.11.
