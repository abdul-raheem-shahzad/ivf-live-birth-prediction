# Dataset Encoding Documentation

How the IVF cohort (`updated_method_encoded.09.25.xlsx`) is turned into the model-ready
feature matrix used by `compare_13models_fixed.py` — from the raw clinical Excel
columns to the 50 features the models actually see.

Everything here happens **in code at run time**. Unlike earlier revisions of this
cohort, the Excel file ships only raw clinical columns; no encoding step is baked into
the spreadsheet.

---

## 0. Dataset at a glance

| Property | Value |
|----------|-------|
| File | `updated_method_encoded.09.25.xlsx` (not distributed — see `data/README.md`) |
| Rows (patients/cycles) | 616 |
| Columns | 62 clinical, including the raw `Embryo morphology` text |
| Target | `Transplant outcome (Y)` — four classes |
| Positive class | live birth (`childbirth`) = **1** (251 / 616 = 40.7%) |
| Missing cells | 1,302 |

---

## 1. Target encoding

The outcome is recorded in **four** classes and binarised by **exact match**, so that a
successful live birth is always the positive class (= 1).

| Raw label | Count | Encoded |
|-----------|------:|:-------:|
| `childbirth` (also accepted: `live birth`, `yes`, `y`, `1`) | 251 | **1** |
| `Not pregnant` | 293 | 0 |
| `abortion` | 45 | 0 |
| `Biochemical pregnancy` | 27 | 0 |

Two things matter here:

- **Match by meaning, not alphabetically.** A plain `LabelEncoder` sorts labels and
  would pick the wrong class as positive, silently inverting precision, recall and F1.
- **Match exactly, not by substring.** `Biochemical pregnancy` contains "pregnan" but is
  *not* a live birth. Exact aliases stop it, and `Not pregnant`, being scored as
  success.

The run prints the labels it saw and the resulting positive count, and raises if no
positive label matched — so a renamed category fails loudly instead of quietly
producing a degenerate model.

---

## 2. Feature type split

Every predictor column is declared as **numerical** or **categorical** by domain
knowledge (not auto-detected), which prevents numeric columns from being accidentally
one-hot encoded.

- **Numerical** (61) → cleaned, imputed, scaled; one column each.
- **Categorical** (8) → one-hot encoded into 0/1 dummy columns.
- **Excluded** → `Transplant outcome (Y)` (the target) and `Embryo morphology` (raw
  text, represented numerically by the ten derived columns in §4).

61 + 8 = **69 raw predictors**.

### Categorical columns (8)

`Medication regimen`, `Method of fertilization`, `female smoking history`,
`male smoking history`, `HCG Day Trigger type`, `Types of embryos transferred`,
`Number of cleavage-stage embryos transferred`, `Number of blastocysts transferred`.

### Numerical columns (61)

All remaining clinical measurements — female and male demographics and anthropometry,
basal / Gn-start-day / HCG-day hormone assays, semen parameters, stimulation doses,
oocyte and embryology counts — **plus** the ten derived embryo columns from §4.

> **Multicollinearity note:** `female Weight (kg)` is **excluded** from the numerical
> list because it is highly collinear with `female BMI (kg/m²)` (r = 0.88, VIF ≈ 6.7).
> BMI (the clinical summary) and `female Height (cm)` (independent) are kept. Male
> height/weight/BMI are **not** collinear (VIF ≈ 1) and are all kept.

---

## 3. Numerical encoding

For every numerical column, in order:

1. **Coerce to number** — `pd.to_numeric(errors='coerce')` turns stray text
   (placeholders like `/`) into `NaN`.
2. **Impute** — `KNNImputer(n_neighbors=5)`, **fit on the training split only** (then
   applied to test) to avoid leakage.
3. **Scale** — `RobustScaler` (median / IQR, robust to outliers), also **fit on train
   only**.

---

## 4. Embryo morphology encoding (derived in code)

`encode_embryo_morphology()` in `compare_13models_fixed.py` parses the raw
`Embryo morphology` text. It runs automatically whenever the derived columns are
absent, so the same pipeline works on a raw or a pre-encoded file.

The text describes **one or two embryos** (two separated by `;`) in one of two grading
systems:

| System | Example | Meaning |
|--------|---------|---------|
| Blastocyst | `4AA` | `(expansion)(ICM)(TE)` |
| Cleavage | `II/8` | `(Roman grade)/(cell count)` |

It becomes **10 numeric columns** — separate slots for embryo 1 (`e1`) and embryo 2
(`e2`), and separate columns per system so the two scales never mix. Any slot that does
not apply is **0** (embryo-2 slots for a single-embryo transfer; blastocyst columns on a
cleavage row).

### 4.1 Grade value tables

**Blastocyst ICM / TE letters** (the leading expansion digit is handled in §4.4):

| Letter | Value |
|:------:|:-----:|
| A | 3 (best) |
| B | 2 |
| C | 1 (poor) |

**Cleavage Roman grade:**

| Grade | Value |
|:-----:|:-----:|
| I | 4 (best) |
| II | 3 |
| III | 2 |
| IV | 1 (poor) |

### 4.2 Blastocyst weighted columns

ICM and TE are weighted **differently** — the methodological contribution — and stored
as the weighted product:

```
stored ICM value = ICM_grade × 0.5
stored TE  value = TE_grade  × 1.0
```

| Column | Formula |
|--------|---------|
| `blast_e1_ICM_w` | ICM_grade × 0.5 |
| `blast_e1_TE_w`  | TE_grade × 1.0 |
| `blast_e2_ICM_w` | ICM_grade × 0.5 |
| `blast_e2_TE_w`  | TE_grade × 1.0 |

This ranks the TE-superior `4BA` above `4AB`, a distinction equal-weight Gardner
scoring cannot make.

### 4.3 Cleavage grade columns

| Column | Value |
|--------|-------|
| `clv_e1_grade` | Roman grade value (I=4 … IV=1) |
| `clv_e2_grade` | Roman grade value |

### 4.4 Clinical "inverted-U" quality scores

Two clinical facts are non-monotonic — quality peaks in a middle window and falls off
toward **both** extremes — so they get their own optimality scores.

**Cleavage cell-count quality** (`clv_e1_cell_quality`, `clv_e2_cell_quality`):

| Cell count | Value | Meaning |
|:----------:|:-----:|---------|
| 8 | 3 | optimal peak |
| 7 or 9 | 2 | high-quality band |
| < 7 or > 9 | 1 | suboptimal (delayed / accelerated) |
| (no cleavage embryo) | 0 | not applicable |

**Blastocyst expansion quality** (`blast_e1_exp_quality`, `blast_e2_exp_quality`):

| Expansion | Value | Meaning |
|:---------:|:-----:|---------|
| 3 or 4 | 2 | optimal clinical stage |
| 1, 2, 5, 6 | 1 | suboptimal (early / mature-hatching) |
| (no blastocyst embryo) | 0 | not applicable |

### 4.5 Worked examples

| Morphology | ICM/TE (e1) | ICM/TE (e2) | clv grade (e1,e2) | exp qual (e1,e2) | cell qual (e1,e2) |
|------------|:-----------:|:-----------:|:-----------------:|:----------------:|:-----------------:|
| `4AA`        | 1.5 / 3.0 | 0 / 0 | 0, 0 | 2, 0 | 0, 0 |
| `4BB; 4BB`   | 1.0 / 2.0 | 1.0 / 2.0 | 0, 0 | 2, 2 | 0, 0 |
| `4BC; 4BB`   | 1.0 / 1.0 | 1.0 / 2.0 | 0, 0 | 2, 2 | 0, 0 |
| `4CB`        | 0.5 / 2.0 | 0 / 0 | 0, 0 | 2, 0 | 0, 0 |
| `II/8`       | 0 / 0 | 0 / 0 | 3, 0 | 0, 0 | 3, 0 |
| `II/7; II/8` | 0 / 0 | 0 / 0 | 3, 3 | 0, 0 | 2, 3 |
| `III/6`      | 0 / 0 | 0 / 0 | 2, 0 | 0, 0 | 1, 0 |

---

## 5. Categorical (one-hot) encoding

Each categorical column is cast to string and expanded with
`pd.get_dummies(..., drop_first=True)`:

- One 0/1 dummy per category, **dropping the first** as a reference (avoids
  multicollinearity among the dummies).
- 8 categorical columns → **15 dummy columns** in this cohort.

### The `_nan` dummies

Some columns (e.g. `Number of cleavage-stage embryos transferred`) are blank (`NaN`) for
patients they do not apply to. Because the column is cast to string before
`get_dummies`, `NaN` becomes the literal category `'nan'`, producing a `..._nan` dummy.

> The `NaN` here is **informative, not random** — those rows are exactly the
> blastocyst-transfer patients, so `..._nan` acts as a transfer-type indicator. Two such
> dummies (`Number of cleavage-stage embryos transferred_nan`,
> `Number of blastocysts transferred_nan`) survive feature selection.
>
> **This behaviour is pandas-version dependent.** `astype(str)` maps `NaN → 'nan'` on
> pandas < 3.0 only; from pandas 3.0 the `_nan` dummies are not produced, the matrix has
> 74 columns instead of 76, and the reported numbers will not reproduce. `requirements.txt`
> pins `pandas==2.3.3` for this reason.

---

## 6. Feature selection

After numeric scaling and one-hot encoding are combined:

- Total processed features: **76** (61 numerical + 15 dummies).
- `SelectKBest(f_classif, k=50)` keeps the **top 50** by ANOVA F-score, **fit on the
  training split only**.

The selected 50 break down as: Embryo / Lab 17, Hormone Levels 16, Treatment / Other 8,
Male Factors 6, Female Factors 3. The full list is in
[`../results/selected_features_by_category.txt`](../results/selected_features_by_category.txt).

---

## 7. End-to-end pipeline order

```
Raw Excel (616 × 62)
   │
   ├─ derive 10 embryo columns from 'Embryo morphology' (if absent)
   ├─ exclude target + raw 'Embryo morphology'
   ├─ target  → 0/1 (childbirth = 1; other three classes = 0)
   ├─ split numerical (61) / categorical (8)   ← drop female Weight (kg)
   │
   ├─ TRAIN/TEST SPLIT (85% / 15%, stratified)   ← before any fitting (no leakage)
   │
   ├─ numerical:  to_numeric → KNNImpute(fit=train) → RobustScale(fit=train)
   ├─ categorical: astype(str) → get_dummies(drop_first)
   ├─ concat → 76 features
   └─ SelectKBest(f_classif, k=50, fit=train)     → 50 model-ready features
```

---

## 8. Column-by-column encoding summary

| Column group | Encoding |
|--------------|----------|
| `Transplant outcome (Y)` | target → 1 (childbirth) / 0 (other three classes) |
| Ages, BMI, height, hormones, semen, doses, counts | numeric → impute + RobustScale |
| `female Weight (kg)` | **dropped** (collinear with BMI) |
| `Medication regimen`, `Method of fertilization`, smoking (×2), `HCG Day Trigger type`, `Types of embryos transferred`, transferred-counts (×2) | one-hot (`drop_first=True`) |
| `Embryo morphology` (raw text) | **excluded** from the matrix; parsed into the 10 columns below |
| `blast_e{1,2}_ICM_w`, `blast_e{1,2}_TE_w` | ICM×0.5, TE×1.0 (numeric) |
| `clv_e{1,2}_grade` | Roman grade I–IV → 4–1 (numeric) |
| `blast_e{1,2}_exp_quality` | expansion inverted-U (2/1/0) |
| `clv_e{1,2}_cell_quality` | cell-count inverted-U (3/2/1/0) |

---

*Implemented in [`../compare_13models_fixed.py`](../compare_13models_fixed.py)
(§1–6 and `encode_embryo_morphology()`).*
