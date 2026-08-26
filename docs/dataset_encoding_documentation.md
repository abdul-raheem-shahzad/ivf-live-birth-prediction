# Dataset Encoding Documentation

How the IVF dataset (`updated_method_encoded.07.25.xlsx`) is encoded — from the raw
clinical Excel columns to the model-ready feature matrix used in
`compare_13models_fixed.py`.

There are **two layers** of encoding:

1. **In-file encoding** — the raw free-text `Embryo morphology` column was converted
   into 10 numeric columns that are already stored in the Excel file.
2. **Modeling-time encoding** — applied in code each run: target encoding, numeric
   cleaning/imputation/scaling, one-hot encoding of categoricals, and feature selection.

---

## 0. Dataset at a glance

| Property | Value |
|----------|-------|
| File | `updated_method_encoded.07.25.xlsx` |
| Rows (patients/cycles) | 616 |
| Columns | 72 (62 clinical + 10 embryo-encoding) |
| Target | `Transfer outcome (Y)` — `Live birth` / `Non-live birth` |
| Positive class | Live birth = **1** (251 / 616 = 40.7%) |

---

## 1. Target encoding

The outcome label is mapped **by meaning**, not alphabetically, so that a successful
**live birth is always the positive class (= 1)**.

| Raw label | Encoded |
|-----------|:------:|
| `Live birth` (also: `childbirth`, `pregnant`, `yes`, `y`, `1`) | **1** |
| `Non-live birth` (anything else) | **0** |

> Why explicit? A plain `LabelEncoder` sorts alphabetically and would pick
> `Non-live birth` as class 1, silently inverting Precision / Recall / F1.

---

## 2. Feature type split

Every predictor column is declared as **numerical** or **categorical** by domain
knowledge (not auto-detected), which prevents numeric columns from being accidentally
one-hot encoded.

- **Numerical** → cleaned, imputed, scaled (kept as-is, one column each).
- **Categorical** → one-hot encoded into 0/1 dummy columns.
- **Excluded** → `Transfer outcome (Y)` (target) and `Embryo morphology` (raw text,
  already represented by the 10 encoded columns).

### Categorical columns (8)

`Ovarian stimulation protocol`, `Method of fertilization`, `Female smoking history`,
`Male smoking history`, `Trigger type`, `Types of embryos transferred`,
`Number of cleavage-stage embryos transferred`, `Number of blastocysts transferred`.

### Numerical columns

All remaining clinical measurements (age, BMI, hormone assays, semen parameters,
stimulation doses, embryology counts) **plus** the 10 embryo-encoding columns from §4.

> **Multicollinearity note:** `Female weight (kg)` is **excluded** from the numerical
> list because it is highly collinear with `Female BMI (kg/m²)` (r = 0.88, VIF ≈ 6.7).
> BMI (the clinical summary) and `Female height (cm)` (independent) are kept. Male
> height/weight/BMI are **not** collinear (VIF ≈ 1) and are all kept.

---

## 3. Numerical encoding

For every numerical column, in order:

1. **Coerce to number** — `pd.to_numeric(errors='coerce')` turns stray text
   (placeholders like `/`) into `NaN`.
2. **Impute** — `KNNImputer(n_neighbors=5)` fills missing values, **fit on the training
   split only** (then applied to test) to avoid data leakage.
3. **Scale** — `RobustScaler` (median / IQR, robust to outliers), also **fit on train
   only**.

---

## 4. Embryo morphology encoding (already stored in the Excel)

The raw `Embryo morphology` text describes **one or two embryos** (two are separated by
`;`) in one of two grading systems:

| System | Example | Meaning |
|--------|---------|---------|
| Blastocyst | `4AA` | `(expansion)(ICM)(TE)` |
| Cleavage | `II/8` | `(Roman grade)/(cell count)` |

It is converted into **10 numeric columns** — separate slots for embryo 1 (`e1`) and
embryo 2 (`e2`), and separate columns for the two systems so the scales never mix.
Any slot that does not apply is **0** (e.g. embryo-2 slots for a single-embryo transfer;
blastocyst columns on a cleavage row).

### 4.1 Grade value tables

**Blastocyst ICM / TE letters** (the leading expansion digit is handled separately in §4.3):

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

Based on correlation analysis, ICM and TE are weighted differently and stored as the
**weighted product**:

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
| `4AB; 4BB`   | 1.5 / 2.0 | 1.0 / 2.0 | 0, 0 | 2, 2 | 0, 0 |
| `5AA`        | 1.5 / 3.0 | 0 / 0 | 0, 0 | 1, 0 | 0, 0 |
| `II/8`       | 0 / 0 | 0 / 0 | 3, 0 | 0, 0 | 3, 0 |
| `II/7; II/8` | 0 / 0 | 0 / 0 | 3, 3 | 0, 0 | 2, 3 |
| `III/6`      | 0 / 0 | 0 / 0 | 2, 0 | 0, 0 | 1, 0 |
| `II/10`      | 0 / 0 | 0 / 0 | 3, 0 | 0, 0 | 1, 0 |

---

## 5. Categorical (one-hot) encoding

Each categorical column is cast to string and expanded with
`pd.get_dummies(..., drop_first=True)`:

- One 0/1 dummy per category, **dropping the first** as a reference (avoids
  multicollinearity among the dummies).
- 8 categorical columns → **15 dummy columns** in this dataset.

### The `_nan` dummies

Some columns (e.g. `Number of cleavage-stage embryos transferred`) are blank (`NaN`) for
patients they don't apply to. Because the column is cast to string before `get_dummies`,
`NaN` becomes the literal category `'nan'`, producing a `..._nan` dummy.

Example — `Number of cleavage-stage embryos transferred` has values `1.0` (160), `2.0`
(83), and `NaN` (373):

| Generated dummy | = 1 when… | Count |
|-----------------|-----------|:-----:|
| *(reference `1.0`, dropped)* | 1 cleavage embryo transferred | 160 |
| `..._2.0` | 2 cleavage embryos transferred | 83 |
| `..._nan` | value missing → **blastocyst transfer** | 373 |

> The `NaN` here is **informative, not random** — those 373 rows are exactly the
> blastocyst-transfer patients, so `..._nan` acts as a transfer-type indicator.
> (Behaviour is pandas-version dependent: `astype(str)` maps `NaN → 'nan'` on older
> pandas; on pandas 3.0 the `_nan` dummy is not produced.)

---

## 6. Feature selection

After numeric scaling and one-hot encoding are combined:

- Total processed features: **~74** (numerical + 15 dummies).
- `SelectKBest(f_classif, k=50)` keeps the **top 50** features by ANOVA F-score,
  **fit on the training split only**.

---

## 7. End-to-end pipeline order

```
Raw Excel (72 cols)
   │
   ├─ exclude target + raw 'Embryo morphology'
   ├─ target  → 0/1 (Live birth = 1)
   ├─ split numerical / categorical (drop female weight)
   │
   ├─ TRAIN/TEST SPLIT (85% / 15%, stratified)   ← before any fitting (no leakage)
   │
   ├─ numerical:  to_numeric → KNNImpute(fit=train) → RobustScale(fit=train)
   ├─ categorical: astype(str) → get_dummies(drop_first)
   ├─ concat → ~74 features
   └─ SelectKBest(f_classif, k=50, fit=train)     → 50 model-ready features
```

---

## 8. Column-by-column encoding summary

| Column group | Encoding |
|--------------|----------|
| `Transfer outcome (Y)` | target → 1 (live birth) / 0 |
| Ages, BMI, height, hormones, semen, doses, counts | numeric → impute + RobustScale |
| `Female weight (kg)` | **dropped** (collinear with BMI) |
| `Ovarian stimulation protocol`, `Method of fertilization`, smoking (×2), `Trigger type`, `Types of embryos transferred`, transferred-counts (×2) | one-hot (`drop_first=True`) |
| `Embryo morphology` (raw text) | **excluded** (encoded in the 10 columns below) |
| `blast_e{1,2}_ICM_w`, `blast_e{1,2}_TE_w` | ICM×0.5, TE×1.0 (numeric) |
| `clv_e{1,2}_grade` | Roman grade I–IV → 4–1 (numeric) |
| `blast_e{1,2}_exp_quality` | expansion inverted-U (2/1/0) |
| `clv_e{1,2}_cell_quality` | cell-count inverted-U (3/2/1/0) |

---

*Related: `embryo_morphology_encoding_v2.md` (detailed derivation of the 10 embryo
columns) and `compare_13models_fixed.py` (the code that performs §1–6).*
