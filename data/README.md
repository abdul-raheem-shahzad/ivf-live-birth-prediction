# Data

**The dataset is not distributed with this repository.**

`updated_method_encoded.07.25.xlsx` — the curated single-centre IVF/ICSI cohort behind
every result here — is deliberately absent. This directory holds only this note; the
code will tell you where to put the file if you obtain it.

## Why it is not public

The cohort consists of patient-level clinical records: per-cycle hormonal assays,
semen parameters, stimulation doses, embryology counts, embryo grades, and the
live-birth outcome, for 616 real treatment cycles at a single reproductive medicine
centre. Even with direct identifiers removed, a record this detailed and this narrow
in provenance — one centre, a defined date range, a rare combination of values per row
— is not safely anonymous, and publishing it would put the privacy of the patients who
consented to nothing of the sort at unnecessary risk.

The study's ethics approval (Zhongnan Hospital of Wuhan University, No. 2025346K)
covers retrospective analysis with a waiver of informed consent. It does not authorize
public redistribution of the underlying records. Out of respect for that boundary, and
for the patients it protects, the file stays out of this repository.

## Requesting access

The dataset is available from the corresponding authors on reasonable request, subject
to institutional approval and an appropriate data-use agreement:

- Ming Zhang — whu_mz@163.com
- Juan Deng — dengjuan@whu.edu.cn

Bona fide academic requests for verification or methodological reuse are welcome.
Please describe the intended analysis in your request.

## What the file contains

Enough detail to understand the pipeline, prepare a comparable cohort of your own, or
review the code without holding the data:

| Property | Value |
|---|---|
| Rows (fresh embryo-transfer cycles) | 616 |
| Columns | 72 — 62 clinical + 10 embryo-morphology encoding columns |
| Target column | `Transfer outcome (Y)` — `Live birth` / `Non-live birth` |
| Class balance | 251 live birth (40.7%) / 365 non-live birth (59.3%) |
| Missing cells | 1,302 (imputed at model time, on the training split only) |
| Unit of analysis | one treatment cycle |
| Collection period | February 2018 – November 2025 |
| Source | Reproductive Medicine Center, Zhongnan Hospital of Wuhan University |

One row is one cycle; patients contributing more than one cycle are not linked across
rows.

The exact column list, the numerical/categorical split, the target mapping, and the ten
embryo-morphology columns produced by the differential ICM/TE weighting are documented
column-by-column in
[`../docs/dataset_encoding_documentation.md`](../docs/dataset_encoding_documentation.md).
The aggregate outputs of a full run — model metrics, the 50 selected features, and
consensus importance — are committed under [`../results/`](../results), so the reported
numbers can be inspected without the raw data.

## Supplying the file once you have it

Either drop it in this directory:

```
data/updated_method_encoded.07.25.xlsx
```

or keep it somewhere private and point the code at it:

```bash
IVF_DATA_FILE=/secure/path/updated_method_encoded.07.25.xlsx python run_and_visualize.py
```

`.gitignore` excludes `data/*.xlsx`, so a local copy placed here will not be committed
by accident.

If the file is missing, `run_and_visualize.py` stops immediately with a
`FileNotFoundError` naming both options above — it never silently runs on partial data.

## Terms of use

Any copy obtained from the authors is for non-commercial academic research and for
verifying the results reported in the accompanying manuscript. It is not covered by the
MIT license that applies to the code. Do not redistribute it, and do not attempt to
re-identify individuals. If you use it, please cite the article listed in
`CITATION.cff`.
