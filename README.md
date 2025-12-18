# Code that is included
### Part 1: Embedding and Clustering Utilities
**File:** `emdedding_cluster.py`

This module is responsible for the core mathematical operations of the pipeline: vectorization, dimensionality reduction, and unsupervised clustering. It provides a wrapper class `emd_clust` that handles the transition from textual data to cluster labels.

*   **Vectorization (`extract_embeddings_simple`)**:
    *   Utilizes the `SentenceTransformer` library.
    *   The default model is `all-MiniLM-L6-v2`, selected for its balance between inference speed and semantic capture.
    *   Embeddings are generated on the GPU (if available) in batches to optimize performance.

*   **Cluster Estimation (`amount_of_clusters`)**:
    *   Implements an automated mechanism to determine the optimal number of senses (clusters) for a given word.
    *   Iterates through candidate cluster counts (k=2 to 3) and calculates the **Silhouette Score** for each configuration.
    *   Selects the $k$ value that yields the highest cohesion within clusters and separation between clusters.

*   **Clustering Algorithms**:
    *   **K-Means (`clustering`)**: A standard centroid-based implementation.
    *   **K-Means with Power Transformation (`clustering_powertrans`)**: Applies a `PowerTransformer` (Yeo-Johnson method) to the embeddings before clustering. This is used to make the data distribution more Gaussian-like, stabilizing variance and potentially improving K-Means performance.
    *   **Agglomerative Clustering (`agglomerative_clustering` variants)**: Hierarchical clustering using the cosine metric and average linkage. This method was explored as an alternative to K-Means to handle non-spherical cluster shapes. It includes variations with and without L2 normalization of input vectors.

*   **Evaluation (`ARI`)**:
    *   Computes the Adjusted Rand Index (ARI) to compare predicted clusters against ground truth ("gold") sense IDs.

### Part 2: Configuration and Helper Utilities
**File:** `helper.py`

This file manages the user interface, configuration constants, and dynamic class loading. It acts as the orchestration layer between the user's choices and the pipeline execution.

*   **Model Registries**:
    *   Maintains a dictionary of **Translation Models** (NLLB variants ranging from 600M to 3.3B parameters) to allow users to trade off between translation quality and execution speed.
    *   Maintains a registry of **WSD Strategies** (`Tcluster`, `Tcluster_plus`, `TEnsemble`), mapping user-friendly names to actual Python classes.

*   **Interactive Selection**:
    *   Provides command-line interface methods (`get_user_model_choice_translator`, `get_user_model_choice_WSD`) to capture runtime configuration securely.
    *   Ensures that only valid strategies and model architectures are instantiated.

### Part 3: Main Execution Script
**File:** `run_pipe.py`

This script serves as the entry point for the application. It handles file I/O, error management, and the high-level execution flow.

*   **Data Ingestion**:
    *   Parses TSV (Tab-Separated Values) files containing context IDs, target words, and context sentences.
    *   Handles data cleaning, specifically removing dummy columns and filtering header rows.

*   **Execution Flow**:
    *   Initializes the `helper` class to retrieve user configurations.
    *   Instantiates the specific WSD pipeline class (e.g., `TEnsemble`) based on the user's selection.
    *   Triggers the processing pipeline and saves the final output, including predicted sense IDs, to `checkpoint_translated.tsv`.

### Part 4: Baseline WSD Pipeline (Translation + Clustering)
**File:** `translator_clus.py`

This module implements the `TCluster` class, representing the baseline Cross-Lingual WSD strategy. The core hypothesis is that ambiguous words in a source language (Russian) often translate to distinct words in a target language (English) depending on the context.

*   **Phase 1: Batch Translation**:
    *   Translates the entire dataset of Russian contexts into English.
    *   Checks for pre-existing translation columns to avoid redundant computation.

*   **Phase 2: Per-Word Clustering**:
    *   The dataset is grouped by the target `word`. Clustering is performed independently for each unique target word to distinguish its specific senses.
    *   English translations are converted into vector embeddings.
    *   Vectors are clustered (using the methods defined in `emdedding_cluster.py`) to assign a `predict_sense_id`.

### Part 5: Ensemble WSD Pipeline (Multilingual Stacking)
**File:** `TEnsemble.py`

This module implements the `TEnsemble` class, which inherits from `TCluster`. It extends the baseline approach by utilizing a multilingual ensemble strategy to capture semantic nuances that might be lost in a single-direction translation.

*   **Multilingual Translation**:
    *   Instead of translating only to English, the context is translated into three target languages: **English, German, and French**.
    *   Uses dual-mode translation initialization (if enabled) to handle multiple target languages efficiently.

*   **Feature Stacking**:
    *   **English Vectors**: Generated using `all-MiniLM-L6-v2`.
    *   **German/French Vectors**: Generated using the multilingual model `paraphrase-multilingual-MiniLM-L12-v2`.
    *   **Concatenation**: The embeddings from all three languages are horizontally stacked (concatenated) to form a high-dimensional feature vector (Dimension: $3 \times 384$).

*   **Ensemble Clustering**:
    *   The stacked vectors are passed to the `clustering_powertrans` method.
    *   This approach relies on the assumption that errors or ambiguities in one language's translation may be corrected or clarified by the semantic signals in the other languages.
# What is the structue of the best result?
graph TD
    Input[Input: Russian Contexts] --> Split{Parallel Translation}
    
    subgraph "Multilingual Projection"
        Split -->|NLLB| EN[English Translation]
        Split -->|NLLB| DE[German Translation]
        Split -->|NLLB| FR[French Translation]
    end
    
    subgraph "Vectorization"
        EN -->|MiniLM-L6| VecEN[Vector EN <br/> (384 dim)]
        DE -->|Multi-MiniLM-L12| VecDE[Vector DE <br/> (384 dim)]
        FR -->|Multi-MiniLM-L12| VecFR[Vector FR <br/> (384 dim)]
    end
    
    subgraph "Fusion & Analysis"
        VecEN & VecDE & VecFR --> Concat[Concatenation <br/> (1152 dim)]
        Concat --> Norm[Power Transformation <br/> (Yeo-Johnson)]
        Norm --> Cluster[K-Means Clustering]
    end
    
    Cluster --> Output[Output: Sense IDs]
  # How to run it: using bash:
  chmod +x run.sh </br>
  ./run.sh data/my_dataset.tsv
