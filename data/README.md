# Dataset

`updated_method_encoded.07.25.xlsx` - the curated single-centre IVF/ICSI cohort used
for every result in this repository.

| Property | Value |
|---|---|
| Rows (fresh embryo-transfer cycles) | 616 |
| Columns | 72 (62 clinical + 10 embryo-morphology encoding columns) |
| Target column | `Transfer outcome (Y)` |
| Class balance | 251 live birth (40.7%) / 365 non-live birth (59.3%) |
| Missing cells | 1,302 (imputed at model time, training split only) |
| Unit of analysis | one treatment cycle |
| Collection period | February 2018 - November 2025 |
| Source | Reproductive Medicine Center, Zhongnan Hospital of Wuhan University |

One row is one cycle; patients contributing more than one cycle are not linked
across rows.

## How the columns are encoded

See [`../docs/dataset_encoding_documentation.md`](../docs/dataset_encoding_documentation.md)
for the full column-by-column description: target mapping, the numerical /
categorical split, and the ten embryo-morphology columns produced by the
differential ICM/TE weighting.

Note that this file is already partly encoded: the free-text `Embryo morphology`
column has been converted into the ten `blast_*` / `clv_*` numeric columns. The
remaining preprocessing (imputation, scaling, one-hot encoding, feature
selection) happens in code at run time.

## Ethics and provenance

The study was approved by the Ethics Committee of Zhongnan Hospital of Wuhan
University (Ethical Approval No. 2025346K). Given the retrospective design, the
requirement for informed consent was waived. The file contains no direct
identifiers: no names, medical record numbers, dates of birth, or admission
dates - only clinical, laboratory, and embryological measurements plus the
outcome label.

## Using a copy stored elsewhere

The code looks for the dataset in this directory. To run against a copy held
outside the repository, set `IVF_DATA_FILE` to its full path:

    IVF_DATA_FILE=/secure/path/updated_method_encoded.07.25.xlsx python run_and_visualize.py

To keep the file out of version control entirely, uncomment the `data/*.xlsx`
line in `.gitignore`.

## Terms of use

The dataset is provided for non-commercial academic research and for verifying
the results reported in the accompanying manuscript. It is not covered by the
MIT license that applies to the code. If you use it, please cite the article
listed in `CITATION.cff`. Do not attempt to re-identify individuals.
