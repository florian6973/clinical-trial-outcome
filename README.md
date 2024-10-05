# Clinical Trial Outcome Embedding and SNOMED Mapping Project

This project aims to embed clinical trial outcome titles and SNOMED CT concepts into a shared embedding space using a pre-trained language model. By leveraging these embeddings, we can cluster similar outcomes, visualize the relationships, and map outcomes to standardized SNOMED CT concepts based on similarity. This facilitates the analysis and standardization of clinical trial outcomes.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Data Preparation](#data-preparation)
- [Scripts and Usage](#scripts-and-usage)
  - [1. Embedding Clinical Trial Outcomes](#1-embedding-clinical-trial-outcomes)
  - [2. Clustering and Visualization](#2-clustering-and-visualization)
  - [3. SNOMED CT Hierarchy Mapping](#3-snomed-ct-hierarchy-mapping)
  - [4. Converting CSV Embeddings to Parquet](#4-converting-csv-embeddings-to-parquet)
  - [5. Embedding SNOMED CT Concepts](#5-embedding-snomed-ct-concepts)
  - [6. Computing Similarities and Mapping](#6-computing-similarities-and-mapping)
- [Results](#results)
- [Notes](#notes)
- [License](#license)

## Overview

Clinical trials generate a vast amount of data, much of which is encapsulated in outcome titles. Standardizing these outcomes facilitates data analysis, meta-analyses, and systematic reviews. This project uses embedding techniques to map clinical trial outcome titles to SNOMED CT concepts, enabling:

- Clustering of similar outcomes.
- Visualization of outcome relationships.
- Mapping of outcomes to standardized medical concepts.

The project involves the following steps:

1. **Embedding Clinical Trial Outcomes:** Generating embeddings for outcome titles using a pre-trained language model.
2. **Clustering and Visualization:** Clustering the outcomes based on embeddings and visualizing the clusters.
3. **SNOMED CT Hierarchy Mapping:** Processing SNOMED CT data to create a hierarchy for mapping purposes.
4. **Embedding SNOMED CT Concepts:** Generating embeddings for SNOMED CT concepts.
5. **Computing Similarities and Mapping:** Computing similarities between outcome embeddings and SNOMED embeddings to find the best matches.

## Project Structure

```
clinical_trials/
├── 1_embed_outcomes.py
├── 2_clustering_outcomes.py
├── 3_snomed_mapping.py
├── 4_1_parquet_conversion.py
├── 4_embed_snomed.py
├── 5_compute_similarity.py
├── SNOMED/
│   ├── sct2_Description_Full-en_US1000124_YYYYMMDD.txt
│   └── sct2_Relationship_Full_US1000124_YYYYMMDD.txt
├── outputs/
│   ├── outcome_embeddings.csv
│   ├── outcome_embeddings.parquet
│   ├── snomed_embeddings.csv
│   ├── snomed_embeddings.parquet
│   ├── top5_snomed_matches.csv
│   └── [visualizations and plots]
└── [additional data files]
```

- **`clinical_trials/`:** Contains all the scripts for the project.
- **`SNOMED/`:** Directory to store SNOMED CT data files.
- **`outputs/`:** Directory where embeddings, results, and visualizations are saved.

## Prerequisites

- **Python 3.8+**
- **CUDA-enabled GPU(s)** (for efficient computation)
- **Python Packages:**
  - `torch`
  - `transformers`
  - `huggingface_hub`
  - `pandas`
  - `numpy`
  - `scikit-learn` (`sklearn`)
  - `matplotlib`
  - `seaborn`
  - `hdbscan`
  - `networkx`
  - `pygraphviz` (for graph visualization)
  - `igraph`
  - `tqdm` (for progress bars)
- **Hugging Face Account and API Key**
- **SNOMED CT Data Files:**
  - `sct2_Description_Full-en_US1000124_YYYYMMDD.txt`
  - `sct2_Relationship_Full_US1000124_YYYYMMDD.txt`
- **Clinical Trial Outcome Data:**
  - A text file containing clinical trial outcome titles.

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/clinical_trial_embedding.git
   cd clinical_trial_embedding
   ```

2. **Create a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages:**

   ```bash
   pip install torch transformers huggingface_hub pandas numpy scikit-learn matplotlib seaborn hdbscan networkx pygraphviz igraph tqdm
   ```

   > **Note:** For GPUs, ensure you have the correct version of PyTorch installed. Refer to the [PyTorch website](https://pytorch.org/) for installation instructions specific to your CUDA version.


4. **Set Up Hugging Face Authentication:**

   - Create an account on [Hugging Face](https://huggingface.co/).
   - Generate an API token from your account settings.
   - Replace the placeholder API key in the scripts with your own.

     ```python
     api_key = 'hf_your_actual_api_key_here'
     ```

## Data Preparation

### 1. Clinical Trial Outcomes

- **Data File:** `clinical_trials/AACT/outcomes.txt`
- The file should be a pipe-delimited (`|`) text file with a `title` column containing the outcome titles.

### 2. SNOMED CT Data

- **Files Needed:**
  - `sct2_Description_Full-en_US1000124_YYYYMMDD.txt`
  - `sct2_Relationship_Full_US1000124_YYYYMMDD.txt`

- **Obtain SNOMED CT Data:**
  - Access to SNOMED CT data requires a license.
  - Register at the [SNOMED International](https://www.snomed.org/) website to obtain the data.

- **Place SNOMED Files:**
  - Store the SNOMED files in the `clinical_trials/SNOMED/` directory.

## Scripts and Usage

### 1. Embedding Clinical Trial Outcomes

**Script:** `clinical_trials/1_embed_outcomes.py`

**Description:**

- Loads clinical trial outcome titles.
- Generates embeddings using the `nvidia/NV-Embed-v2` model from Hugging Face.
- Saves the embeddings to `outputs/outcome_embeddings.csv`.

**Usage:**

```bash
python clinical_trials/1_embed_outcomes.py
```

**Notes:**

- Ensure the `data_file` path in the script points to your outcomes file.
- Adjust the `cache_dir`, `offload_folder`, and other paths as necessary.
- Make sure to replace the placeholder API key with your own.

### 2. Clustering and Visualization

**Script:** `clinical_trials/2_clustering_outcomes.py`

**Description:**

- Loads the embeddings of the outcomes.
- Reduces dimensions using PCA and UMAP.
- Performs clustering using HDBSCAN.
- Visualizes and saves the clustering results.

**Usage:**

```bash
python clinical_trials/2_clustering_outcomes.py
```

**Notes:**

- Ensure `embeddings_file` path points to `outputs/outcome_embeddings.csv`.
- The script saves visualizations in the `outputs/` directory.

### 3. SNOMED CT Hierarchy Mapping

**Script:** `clinical_trials/3_snomed_mapping.py`

**Description:**

- Processes SNOMED CT relationship and description files.
- Builds a hierarchy of SNOMED concepts related to clinical findings.
- Visualizes and saves the hierarchy graph.

**Usage:**

```bash
python clinical_trials/3_snomed_mapping.py
```

**Notes:**

- Ensure SNOMED CT data files are in the correct directory and the paths in the script are accurate.
- Adjust `max_depth` in the script to control the depth of the hierarchy visualized.
- The script outputs graphs to the `outputs/` directory.

### 4. Converting CSV Embeddings to Parquet

**Script:** `clinical_trials/4_1_parquet_conversion.py`

**Description:**

- Converts CSV embedding files to Parquet format for efficient reading and processing.

**Usage:**

```bash
python clinical_trials/4_1_parquet_conversion.py
```

**Notes:**

- Ensure the input CSV files exist in the `outputs/` directory.
- Modify the script if you need to convert additional files.

### 5. Embedding SNOMED CT Concepts

**Script:** `clinical_trials/4_embed_snomed.py`

**Description:**

- Reads the SNOMED concepts from the hierarchy.
- Generates embeddings using the same pre-trained model.
- Saves the embeddings to `outputs/snomed_embeddings.csv`.

**Usage:**

```bash
python clinical_trials/4_embed_snomed.py
```

**Notes:**

- Ensure `descendants_with_depth.csv` is available from the previous SNOMED mapping step.
- Adjust paths and API keys as necessary.

### 6. Computing Similarities and Mapping

**Script:** `clinical_trials/5_compute_similarity.py`

**Description:**

- Loads the outcome embeddings and SNOMED embeddings.
- Computes cosine similarities between each outcome and all SNOMED concepts.
- Retrieves the top 5 SNOMED concepts for each outcome based on similarity.
- Saves the results to `outputs/top5_snomed_matches.csv`.

**Usage:**

```bash
python clinical_trials/5_compute_similarity.py
```

**Notes:**

- Ensure the Parquet embedding files are available in the `outputs/` directory.
- Adjust `batch_size` and `top_k` parameters if necessary.

## Results

- **Outcome Embeddings:** `outputs/outcome_embeddings.csv` and `outputs/outcome_embeddings.parquet`
- **SNOMED Embeddings:** `outputs/snomed_embeddings.csv` and `outputs/snomed_embeddings.parquet`
- **Cluster Visualization:** `outputs/umap_clustering_chart.png`
- **SNOMED Hierarchy Graphs:**
  - `outputs/snomed_hierarchy_igraph_with_labels_horizontal.svg`
  - `outputs/snomed_hierarchy_igraph_with_labels_horizontal.png`
- **Top 5 SNOMED Matches:** `outputs/top5_snomed_matches.csv`

The final `top5_snomed_matches.csv` file contains the top 5 SNOMED CT concepts for each clinical trial outcome based on embedding similarity, facilitating the mapping of outcomes to standardized medical concepts.

## Notes

- **Adjusting Paths and Directories:**
  - The scripts use specific paths (e.g., `/nlp/projects/llama/`). Modify these paths to match your local environment.
- **GPU Resources:**
  - The scripts are optimized for systems with multiple GPUs. Adjust the `CUDA_VISIBLE_DEVICES` and `max_memory` settings based on your hardware.
- **API Keys and Authentication:**
  - Do not share your Hugging Face API key publicly. Ensure it is kept secure.
- **Data Privacy:**
  - Ensure compliance with data usage agreements, especially when working with licensed datasets like SNOMED CT.
- **Error Handling:**
  - The scripts include basic error handling. Monitor the console output for any issues and adjust parameters accordingly.

## License

This project is licensed under the [MIT License](LICENSE). Please ensure compliance with the licenses of any third-party tools and datasets used.

---

For any questions or contributions, please open an issue or submit a pull request.
