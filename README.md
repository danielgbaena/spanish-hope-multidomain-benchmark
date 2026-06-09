# SpanishHopeMultidomain

SpanishHopeMultidomain is a benchmark for **cross-domain robustness evaluation in Spanish Hope Speech (HS) detection**.

The benchmark was specifically designed to study how NLP systems behave under **unseen-domain conditions** by explicitly separating training and evaluation domains. Unlike conventional random train-test splits, SpanishHopeMultidomain introduces controlled distributional mismatch to enable robustness-oriented evaluation under realistic social media conditions.

The benchmark contains manually annotated Spanish social media posts across three socially relevant domains:

- LGBT-related discourse
- Obesity-related discourse
- Racism-related discourse

The repository includes:

- The benchmark dataset
- Fixed train/development/test partitions
- Annotation resources
- Experimental scripts
- Evaluation utilities
- Statistical analysis scripts
- Figure generation scripts
- Prediction outputs and reproducibility materials

This repository accompanies the paper:

> **SpanishHopeMultidomain: A Benchmark for Cross-Domain Robustness Evaluation in Spanish Hope Speech Detection**

---

# Repository Structure

```text
spanish_hope_multidomain_benchmark/
│
├── data/
│   ├── train.csv
│   ├── dev.csv
│   ├── test.csv
│   └── ...
│
├── results/
│   └── ...
│
├── scripts/
│   ├── svm_baseline.py
│   ├── beto_multiseed.py
│   ├── robertuito_multiseed.py
│   ├── gpt41_mini_crossdomain.py
│   ├── gpt4o_mini_crossdomain.py
│   ├── qwen25_crossdomain.py
│   ├── run_bootstrap_confidence_intervals.py
│   ├── run_mcnemar_tests.py
│   ├── build_main_results_figure.py
│   ├── build_per_domain_figure.py
│   ├── build_gpt41_confusion_matrix.py
│   └── ...
│
├── requirements.txt
├── README.md
└── ...
```

---

# Benchmark Description

The benchmark was intentionally designed for **controlled multidomain robustness evaluation** rather than conventional in-domain supervised classification.

## Domain Distribution

### Training + Development

- LGBT-related discourse only

### Test

- LGBT-related discourse
- Obesity-related discourse
- Racism-related discourse

This setup enables systematic evaluation under explicit **cross-domain topic shift** conditions.

---

## Dataset Repository

The SpanishHopeMultidomain dataset and annotation guidelines are available at:

https://github.com/danielgbaena/spanish-hope-multidomain

---

# Dataset Statistics

| Statistic | Value |
|---|---|
| Number of posts | 2,000 |
| Hope Speech (HS) | 1,000 |
| Non Hope Speech (NHS) | 1,000 |
| Training subset | 1,400 |
| Development subset | 200 |
| Test subset | 400 |

---

# Evaluated Models

The experiments include multiple modeling paradigms:

## Sparse-feature Machine Learning

- TF-IDF + Linear SVM

## Transformer-based Architectures

- BETO
- RoBERTuito

## Instruction-tuned Large Language Models

- GPT-4o-mini
- GPT-4.1-mini
- Qwen2.5-7B

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/danielgbaena/spanish-hope-multidomain.git
cd spanish-hope-multidomain
```

## 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Reproducibility

All experiments were conducted using:

- Fixed train/development/test partitions
- Standardized preprocessing
- Public model checkpoints
- Controlled evaluation conditions
- Multi-seed transformer experiments
- Deterministic prompting configurations for LLMs

The repository includes:

- Training scripts
- Evaluation scripts
- Prompting templates
- Statistical analysis scripts
- Figure generation scripts
- Prediction outputs
- Bootstrap confidence interval computation
- McNemar significance testing

---

# Experimental Scripts

## Supervised Models

### SVM Baseline

```bash
python scripts/svm_baseline.py
```

### BETO Multiseed Evaluation

```bash
python scripts/beto_multiseed.py
```

### RoBERTuito Multiseed Evaluation

```bash
python scripts/robertuito_multiseed.py
```

---

# LLM Evaluation

## GPT-4.1-mini

```bash
python scripts/gpt41_mini_crossdomain.py
```

## GPT-4o-mini

```bash
python scripts/gpt4o_mini_crossdomain.py
```

## Qwen2.5-7B

```bash
python scripts/qwen25_crossdomain.py
```

---

# Statistical Analysis

## Bootstrap Confidence Intervals

```bash
python scripts/run_bootstrap_confidence_intervals.py
```

This script reproduces the confidence interval analysis reported in the paper.

## McNemar Statistical Significance Tests

```bash
python scripts/run_mcnemar_tests.py
```

This script reproduces the pairwise significance analysis between evaluated systems.

---

# Figure Generation

## Main Results Figure

```bash
python scripts/build_main_results_figure.py
```

## Per-domain Results Figure

```bash
python scripts/build_per_domain_figure.py
```

## GPT-4.1-mini Confusion Matrix

```bash
python scripts/build_gpt41_confusion_matrix.py
```

---

# Evaluation Metric

The primary evaluation metric used throughout the benchmark is:

- Macro-F1

Additional analyses include:

- Per-domain robustness evaluation
- Bootstrap confidence intervals
- McNemar statistical significance testing
- Benchmark difficulty analysis
- Qualitative error analysis

---

# Citation

If you use this benchmark or repository, please cite:

```bibtex
@article{garciabaena2026spanishhopemultidomain,
  title={SpanishHopeMultidomain: A Benchmark for Cross-Domain Robustness Evaluation in Spanish Hope Speech Detection},
  author={García-Baena, Daniel and García-Cumbreras, Miguel Ángel and Jiménez-Zafra, Salud María},
  journal={Scientific Reports},
  year={2026}
}
```

---

# Ethical Considerations

The dataset was collected from publicly available social media content.

To preserve user privacy:

- User mentions were removed
- Personally identifiable information was removed
- The benchmark is distributed exclusively for research purposes

The dataset may contain offensive or emotionally sensitive content due to the nature of online discourse.

The evaluated models are intended exclusively for research on robust socially grounded NLP systems and should not be interpreted as fully autonomous moderation systems.

---

# License

This repository is released for research and academic purposes.

Please check the repository license for additional details.

---

# Contact

For questions or issues related to the benchmark or repository:

Daniel García-Baena  
daniel.gbaena@gmail.com
