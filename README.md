# BabyLM_CL_RYS: Growing Smarter, Not Bigger

## Description
This repository contains the code and resources for the project "Growing Smarter, Not Bigger: Curriculum Pretraining and Layer Duplication in Small Language Models".  
The main goal of this study is to determine whether combining Curriculum Learning (CL) with a training-free architectural modification, the Repeat Yourself (RYS) method, can improve the downstream performance of small language models. The project evaluates two CL strategies (human-centric and model-centric) alongside RYS across five experiments trained under the BabyLM Strict-Small protocol data and compute constraints.  
Key features explored in this repository include:
* Curriculum Learning (CL): Implementation of human-centric (Gunning Fog, Dale-Chall) and model-centric (influence-driven) data ordering strategies.  
* Repeat Yourself (RYS) Method: A training-free architectural technique that duplicates transformer layers at inference time without modifying or retraining weights.  
* Architecture: Experiments utilize a 12-layer GPT-2 architecture ($\approx$ 98M parameters) trained for 10 epochs.  

## Repo Structure
Based on the provided project files, here is the organizational structure of the repository:
```text
BabyLM_CL_RYS/
├── docs/                       # Project documentation
│   └── influence-driven/       
├── notebooks/                  # Jupyter notebooks for EDA and result analysis
│   ├── influence-driven/       # Notebooks for the CL and model training for surrogate and influence strategies.
│   ├── rys_results/            # Results of running the rys_analysis notebook.
│   ├── datasets.ipynb          # Notebook for datasets exploration
│   └── rys_analysis.ipynb      # Notebook for RYS layer repetition analysis
├── src/                        # Core source code
│   ├── datasets/               # Tokenization experiments (TODO: Add dataset loading)
│   ├── evaluation/             # Required modifications for scripts for Zero-shot and Fine-tuning evaluation (TODO: Add explanation)
│   ├── experiments/            # (TODO: Consolidate experiment scripts)
│   ├── influence_curriculum/   # Surrogate model and influence-scoring logic
│   └── training/               # Model pretraining scripts
├── tests/                      # Tests for model surrogate and influence strategies
│   ├── test_correctness.py
│   ├── test_jvp_attn_impl.py
│   ├── test_train_curriculum.py
│   └── test_train_word_aware....py
├── .gitignore                  
├── main.py                     # Main entry point for executing experiments
├── pyproject.toml              # Project metadata and dependencies
├── README.md                   # This file
└── uv.lock                     # UV package manager lockfile
```

## Models and Datasets

### Repositories

### Installation
This project uses uv for dependency management (as indicated by the uv.lock file) and standard Python. Ensure you have the correct Python version installed (check the .python-version file).
1. Clone the repository
    ```bash
    git clone https://github.com/eli12gran/BabyLM_CL_RYS.git
    cd BabyLM_CL_RYS
    ```
2. Create a virtual environment and install dependencies using uv:
    ```bash
    uv venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    uv pip install -r pyproject.toml    
    ```

### Execution
* To run a given training pipeline:
    ```bash
    ```


## Key Findings and Results

## Citation