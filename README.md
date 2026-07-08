# SpanishHopeMultidomain Benchmark

This repository provides the complete experimental framework and reproducibility materials accompanying the manuscript:

> **Cross-Domain Generalization of Hope Speech Detection Across Vulnerable Communities: A Benchmark Study of Supervised Models and Large Language Models**

SpanishHopeMultidomain is a benchmark for the controlled evaluation of cross-domain Hope Speech (HS) detection in Spanish social media. The benchmark contains 2,000 manually annotated posts spanning LGBT-related, obesity-related, and racism-related discourse.

The official evaluation protocol explicitly separates social domains. Supervised models are trained exclusively on LGBT-related discourse, whereas obesity-related and racism-related posts are absent from task-specific training and appear only in the multidomain test partition. This design enables controlled evaluation of supervised transfer to previously unseen vulnerable communities.

The same multidomain test instances are used to evaluate instruction-tuned large language models under zero-shot conditions, providing a common evaluation set for comparing supervised cross-domain transfer and zero-shot LLM generalization.

This repository contains the experimental scripts, prediction outputs, evaluation utilities, statistical analyses, model-agreement analyses, and figure-generation code required to reproduce the empirical results reported in the manuscript.

The benchmark data, official partitions, and annotation guidelines are maintained in the companion dataset repository:

`danielgbaena/spanish-hope-multidomain`

## Repository Contents

```text
spanish_hope_multidomain_benchmark/
├── data/
│   ├── train.csv
│   ├── dev.csv
│   ├── test.csv
│   └── test_gold.csv
├── results/
│   ├── svm_predictions.csv
│   ├── svm_results.json
│   ├── beto_seed42_predictions.csv
│   ├── beto_seed52_predictions.csv
│   ├── beto_seed62_predictions.csv
│   ├── beto_majority_predictions.csv
│   ├── beto_multiseed_results.json
│   ├── robertuito_seed42_predictions.csv
│   ├── robertuito_seed52_predictions.csv
│   ├── robertuito_seed62_predictions.csv
│   ├── robertuito_majority_predictions.csv
│   ├── robertuito_multiseed_results.json
│   ├── gpt41_mini_predictions.csv
│   ├── gpt41_mini_results.json
│   ├── gpt4o_mini_predictions.csv
│   ├── gpt4o_mini_results.json
│   ├── qwen25_7b_predictions.csv
│   ├── qwen25_7b_results.json
│   ├── final_results_summary.csv
│   ├── bootstrap_confidence_intervals.csv
│   ├── mcnemar_results.csv
│   ├── model_agreement.csv
│   ├── model_agreement_matrix.csv
│   ├── instance_difficulty.csv
│   ├── instance_difficulty_by_domain.csv
│   ├── unanimous_error_instances.csv
│   ├── main_results_figure.pdf
│   ├── per_domain_results_figure.pdf
│   ├── robustness_degradation.pdf
│   └── gpt41_confusion_matrix.pdf
├── scripts/
│   ├── check_dataset.py
│   ├── svm_baseline.py
│   ├── beto_baseline.py
│   ├── beto_multiseed.py
│   ├── robertuito_baseline.py
│   ├── robertuito_multiseed.py
│   ├── gpt41_mini_crossdomain.py
│   ├── gpt4o_mini_crossdomain.py
│   ├── qwen25_7b_crossdomain.py
│   ├── build_results_summary.py
│   ├── run_bootstrap_confidence_intervals.py
│   ├── run_mcnemar_tests.py
│   ├── run_model_agreement_analysis.py
│   ├── build_main_results_figure.py
│   ├── build_per_domain_figure.py
│   ├── plot_robustness_degradation.py
│   └── build_gpt41_confusion_matrix.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Benchmark Protocol

The benchmark adopts a fixed domain-separated evaluation protocol.

### Training Partition

The training partition contains 1,400 LGBT-related posts and is used for task-specific supervised training.

### Development Partition

The development partition contains 200 LGBT-related posts and is used for model development and validation.

### Test Partition

The multidomain test partition contains 400 posts spanning:

- LGBT-related discourse: 200 instances.
- Obesity-related discourse: 106 instances.
- Racism-related discourse: 94 instances.

The supervised models therefore encounter obesity-related and racism-related discourse only at evaluation time.

Two versions of the test partition are included:

- `test.csv` contains the official test instances without gold labels and can be used for benchmark evaluation or prediction generation.
- `test_gold.csv` contains the same test instances with gold labels and is used by the evaluation, statistical-analysis, and reproducibility scripts.

## Evaluated Models

The experimental framework includes three modeling families.

### Sparse-Feature Machine Learning

- TF-IDF with a linear Support Vector Machine (SVM).

### Supervised Transformer-Based Models

- BETO.
- RoBERTuito.

The transformer-based models are evaluated using three random seeds. Majority-vote predictions are used for the main comparison, while seed-level outputs are retained to support reproducibility and analysis of training variability.

### Instruction-Tuned Large Language Models

- GPT-4.1-mini.
- GPT-4o-mini.
- Qwen2.5-7B.

The instruction-tuned models are evaluated under zero-shot conditions without task-specific training examples.

## Installation

Clone this experimental repository:

```bash
git clone https://github.com/danielgbaena/spanish-hope-multidomain-benchmark.git
cd spanish-hope-multidomain-benchmark
```

Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

### Dataset Validation

```bash
python scripts/check_dataset.py
```

### SVM Baseline

```bash
python scripts/svm_baseline.py
```

### BETO Multi-Seed Evaluation

```bash
python scripts/beto_multiseed.py
```

### RoBERTuito Multi-Seed Evaluation

```bash
python scripts/robertuito_multiseed.py
```

### GPT-4.1-mini Zero-Shot Evaluation

```bash
python scripts/gpt41_mini_crossdomain.py
```

### GPT-4o-mini Zero-Shot Evaluation

```bash
python scripts/gpt4o_mini_crossdomain.py
```

### Qwen2.5-7B Zero-Shot Evaluation

```bash
python scripts/qwen25_7b_crossdomain.py
```

## Evaluation and Statistical Analysis

### Aggregate Results Summary

```bash
python scripts/build_results_summary.py
```

### Bootstrap Confidence Intervals

```bash
python scripts/run_bootstrap_confidence_intervals.py
```

This script computes bootstrap confidence intervals for the evaluated systems on the common multidomain test partition.

### Paired McNemar Significance Tests

```bash
python scripts/run_mcnemar_tests.py
```

This script performs paired significance tests between systems evaluated on the same test instances.

### Model Agreement and Empirical Instance Difficulty

```bash
python scripts/run_model_agreement_analysis.py
```

This script computes:

- pairwise raw agreement between model predictions;
- pairwise Cohen's kappa;
- an agreement matrix;
- the number and proportion of model errors for each test instance;
- empirical instance-difficulty groups relative to the evaluated model set;
- difficulty distributions by social domain;
- instances unanimously misclassified by all evaluated models.

The generated outputs are:

- `results/model_agreement.csv`;
- `results/model_agreement_matrix.csv`;
- `results/instance_difficulty.csv`;
- `results/instance_difficulty_by_domain.csv`;
- `results/unanimous_error_instances.csv`.

The instance-difficulty categories are empirical summaries of prediction behavior across the evaluated model set and should not be interpreted as intrinsic or model-independent properties of the benchmark instances.

## Figure Generation

### Main Results Figure

```bash
python scripts/build_main_results_figure.py
```

### Per-Domain Performance Figure

```bash
python scripts/build_per_domain_figure.py
```

### Cross-Domain Robustness Degradation Figure

```bash
python scripts/plot_robustness_degradation.py
```

### GPT-4.1-mini Confusion Matrix

```bash
python scripts/build_gpt41_confusion_matrix.py
```

## Evaluation Metrics and Analyses

Macro-F1 is the primary evaluation metric.

The experimental framework additionally provides:

- per-domain performance analysis;
- source-domain versus unseen-domain performance comparison;
- bootstrap confidence intervals;
- paired McNemar significance tests;
- pairwise model-agreement analysis;
- Cohen's kappa analysis;
- empirical instance-difficulty analysis;
- domain-specific difficulty distributions;
- identification of unanimous model failures.

## Reproducibility

The repository provides:

- fixed train, development, and test partitions;
- complete prediction outputs for all evaluated systems;
- seed-level predictions for supervised transformer models;
- majority-vote predictions used in the main comparison;
- model-level result files;
- scripts for aggregate and per-domain evaluation;
- bootstrap confidence-interval computation;
- paired statistical significance testing;
- model-agreement and empirical instance-difficulty analysis;
- figure-generation scripts;
- generated tables and figures used to support the manuscript analyses.

Experiments involving proprietary API-based models require valid API credentials and may be affected by future changes to externally hosted model endpoints. The archived prediction outputs and result files are provided to preserve the exact outputs used in the reported analyses.

## Dataset Repository

The benchmark data, official partitions, and complete annotation guidelines are available in the companion repository:

`danielgbaena/spanish-hope-multidomain`

## Citation

If you use SpanishHopeMultidomain or the accompanying experimental framework, please cite the associated manuscript.

```bibtex
@article{garciabaena2026crossdomain,
  title   = {Cross-Domain Generalization of Hope Speech Detection Across Vulnerable Communities: A Benchmark Study of Supervised Models and Large Language Models},
  author  = {García-Baena, Daniel and García-Cumbreras, Miguel Ángel and Jiménez-Zafra, Salud María},
  year    = {2026},
  note    = {Manuscript under review}
}
```

The citation metadata will be updated when publication information becomes available.

## Ethical Considerations

The benchmark contains social media posts discussing vulnerable communities and may include offensive, discriminatory, or emotionally sensitive language.

The benchmark and experimental resources are intended exclusively for research purposes. Model predictions should not be interpreted as autonomous moderation decisions or as assessments of individual social media users.

Researchers using the benchmark should consider the ethical implications of socially oriented NLP research, including privacy, representational limitations, annotation subjectivity, potential model biases, and the risks associated with deployment beyond the controlled benchmark setting.

## License

Reuse and redistribution of the materials in this repository are governed by the terms specified in the `LICENSE` file.

## Contact

For questions concerning the benchmark or experimental framework, please open an issue in the corresponding GitHub repository or contact:

Daniel García-Baena  
daniel.gbaena@gmail.com
