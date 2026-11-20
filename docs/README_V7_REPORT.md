# Final SpaceNet V7 report generator — REQUIRED 6-SECTION FORMAT

This ZIP is **standalone**. You do NOT need to extract either earlier report ZIP.
It contains the complete updated generator and LaTeX template, including
all improvements from `v7_report_generator_updated.zip`.

## From your `image_processing` repository root

```bash
unzip -o v7_report_generator_final_format.zip
pip install jinja2 matplotlib numpy
python generate_v7_report.py
```

The resulting report is:

`experiment_v7/final_report/report.pdf`

Editable LaTeX and images:

- `experiment_v7/final_report/report.tex`
- `experiment_v7/final_report/figures/workflow.png`
- `experiment_v7/final_report/figures/processing.png` (when NPZ is found)
- `experiment_v7/final_report/figures/validation.png`
- `experiment_v7/final_report/figures/seeds.png`
- `experiment_v7/final_report/figures/reference_paper_xbd_f1.png`
- `experiment_v7/final_report/figures/confusion.png`
- `experiment_v7/final_report/figures/example_*.png` (if saved by evaluator)
- `experiment_v7/final_report/validation_per_seed.csv`
- `experiment_v7/final_report/metrics_used.json`

The six major headings are exactly:

1. Problem Statement
2. How to Overcome
3. Workflow Diagram
4. Papers
5. Steps (including validation/test results and qualitative comparisons)
6. Dataset link

Papers includes the earlier six-channel xBD study, the SpaceNet 8 benchmark
paper, and a separate building-extraction enhancement study. The published
xBD results are identified as literature results and never combined with
our own SpaceNet IoU numbers.

The dataset section includes the official SpaceNet 8 challenge page, AWS
Open Data catalog, and Germany training archive S3 path. All relevant
figures and confusion counts read from your completed V7 experiment files.

This generator performs **no training** and **no inference**; your existing
`best_val.json`, `metrics.json`, saved evaluator examples, and corrected NPZ
images are used as-is.

For a LaTeX-only output: `python generate_v7_report.py --no-pdf`.

If PDF compilation isn't available on Debian:

```bash
sudo apt install latexmk texlive-latex-recommended texlive-latex-extra
python generate_v7_report.py
```

No other ZIP needs to be extracted before this one.
