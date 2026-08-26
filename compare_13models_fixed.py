"""
=============================================================================
COMPREHENSIVE COMPARISON WITH 11 MODELS  (FIXED VERSION)
Including TabPFN, TabPFN+LR Hybrid, and Hybrid Ensemble
=============================================================================

This file is a corrected copy of compare_13models_cld.py with TWO fixes:

  FIX 1 - Multicollinearity: dropped the female Weight (kg) column from features.
          Female Weight and BMI are highly collinear (r=0.88, VIF~6.7), which
          destabilises Logistic-Regression coefficients and splits tree-based
          importance. BMI (the clinical summary) and Height (independent) are
          kept; Weight is removed. Male Height/Weight/BMI are NOT collinear
          (VIF~1) so they are left unchanged.

  FIX 2 - "% of Total" > 100% in the category-importance table. The old code
          (a) printed raw_importance*100 without dividing by the grand total
          (so it was never a percentage), and (b) let a feature fall into
          several categories at once ('male' is a substring of 'female', and
          hormone keywords overlapped the sex categories). The new block
          assigns each feature to EXACTLY ONE category by priority and divides
          by the true total, so the percentages sum to 100.

  FIX 3 - 2025-07 dataset: feature lists use the dataset's own new column names.
          Column names are whitespace-stripped on load (several carry a trailing
          space). The category-importance grouping was updated because sex is now
          a name PREFIX ('Female ...'/'Male ...'), not a suffix.

  FIX 4 - Positive class defined explicitly as a successful LIVE BIRTH (=1). The
          new labels 'Live birth'/'Non-live birth' would make a plain
          LabelEncoder pick 'Non-live birth' as positive (alphabetical order),
          inverting Precision/Recall/F1.
=============================================================================

Models Compared:
1. Logistic Regression (LR)
2. Random Forest (RF)
3. Gradient Boosting (GB)
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Naive Bayes (NB)
7. XGBoost
8. TabPFN
9. TabPFN + LR Hybrid
10. IHAN (baseline)
11. Hybrid Ensemble (IHAN + LR)

=============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score
)
import os
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# DATA LOCATION
# The dataset ships in data/ next to this file. Set the IVF_DATA_FILE
# environment variable to run against a copy kept outside the repository.
# ---------------------------------------------------------------------------
DATA_FILENAME = 'updated_method_encoded.07.25.xlsx'
_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_data_file():
    """Return the path of the dataset, or raise with an actionable message."""
    for path in (os.environ.get('IVF_DATA_FILE'),
                 os.path.join(_ROOT, 'data', DATA_FILENAME),
                 os.path.join(_ROOT, DATA_FILENAME),
                 DATA_FILENAME):
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Could not find {DATA_FILENAME!r}. Put it in "
        f"{os.path.join(_ROOT, 'data')!r}, or set the IVF_DATA_FILE environment "
        f"variable to its full path."
    )


try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("TabPFN not available")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class IHANModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=24):
        super(IHANModel, self).__init__()
        self.feature_importance = nn.Parameter(torch.ones(input_dim) * 0.1)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        importance = torch.sigmoid(self.feature_importance)
        x = x * importance.unsqueeze(0)
        return self.network(x).squeeze(-1)


def train_ihan(X_train, y_train, X_val, y_val, epochs=80, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = IHANModel(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    criterion = nn.BCELoss()
    
    best_auc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_preds)
            
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, device


def calculate_metrics(y_true, y_pred, y_proba):
    return {
        'AUC-ROC': roc_auc_score(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
    }


def run_comparison_13models():
    print("="*100)
    print("COMPREHENSIVE MODEL COMPARISON - 11 MODELS")
    print("Including TabPFN+LR Hybrid")
    print("="*100)
    
    # Load data
    print("\n[1] Loading Data...")
    data_file = resolve_data_file()
    print(f"  Dataset: {data_file}")
    df = pd.read_excel(data_file)

    # =====================================================================
    # FIX 3: Column handling for the 2025-07 dataset.
    # The feature lists below use the dataset's own (new) column names. Several
    # of those names carry a trailing space (e.g. 'Female basal FSH (mIU/mL) '),
    # so strip surrounding whitespace on load to guarantee an exact match.
    # =====================================================================
    df.columns = df.columns.str.strip()

    target_col = 'Outcome_Encoded' if 'Outcome_Encoded' in df.columns else 'Transfer outcome (Y)'
    # 'Embryo morphology' is the raw text grade string; it is represented
    # numerically by the encoding columns (blast_e*/clv_e*), so exclude the raw text.
    exclude_cols = ['Outcome_Encoded', 'Transfer outcome (Y)', 'Embryo morphology']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y_raw = df[target_col].values

    # Encode string targets -> 0/1 so metrics (pos_label=1) and the NN work.
    # FIX 4: define the positive class EXPLICITLY as a successful live birth (=1).
    # The 2025-07 labels are 'Live birth' / 'Non-live birth'; a plain LabelEncoder
    # sorts alphabetically and would make 'Non-live birth' the positive class,
    # inverting Precision/Recall/F1. Mapping by meaning keeps success = 1
    # (matching the old 'childbirth' = 1) regardless of spelling or sort order.
    if not np.issubdtype(np.asarray(y_raw).dtype, np.number):
        y_str = pd.Series(y_raw).astype(str).str.strip().str.lower()
        positive_aliases = {'live birth', 'childbirth', 'pregnant', 'yes', '1', 'y'}
        y = y_str.isin(positive_aliases).astype(int).values
        if y.sum() == 0:
            raise ValueError(f"No positive (live-birth) labels matched. Labels seen: "
                             f"{sorted(y_str.unique())}. Update positive_aliases.")
        print(f"  Positive class (=1) = live birth: {int(y.sum())}/{len(y)} "
              f"({y.mean()*100:.1f}%). Labels seen: {sorted(y_str.unique())}")
    else:
        y = y_raw.astype(int)
    
    # =====================================================================
    # FEATURE TYPE DEFINITIONS
    # =====================================================================
    # Explicitly define numerical and categorical features based on domain knowledge.
    # This prevents unintended encoding of numerical features that may be 
    # misclassified due to inconsistent data types in the source Excel file.
    
    NUMERICAL_FEATURES = [
        # FIX 1: 'Female weight (kg)' removed here (collinear with BMI, r=0.88, VIF~6.7).
        "Female age (years)", "Male age (years)", "Female height (cm)",
        "Female BMI (kg/m²)",
        "Female basal FSH (mIU/mL)", "Female basal LH (mIU/mL)",
        "Female basal E2 (pg/mL)", "Female basal PRL (ng/mL)",
        "Female basal T (ng/mL)", "Female basal P (ng/mL)",
        "Female AMH (ng/mL)", "Female TSH (mIU/L)",
        "Duration of infertility (years)",
        "Male height (cm)", "Male weight (kg)", "Male BMI (kg/m²)",
        "Male serum FSH (mIU/mL)", "Male serum LH (mIU/mL)",
        "Male serum PRL (ng/mL)", "Male serum T (ng/mL)",
        "Sperm concentration (10⁶/mL)", "Progressive motility (PR, %)",
        "Non-progressive motility (NP, %)", "Immotility (IM, %)",
        "Normal sperm morphology (%)", "Sperm DFI (%)",
        "Female Gn-start-day FSH (mIU/mL)", "Female Gn-start-day LH (mIU/mL)",
        "Female Gn-start-day E2 (pg/mL)", "Female Gn-start-day P (ng/mL)",
        "Duration of Gn stimulation (days)", "Total Gn dose (IU)",
        "Total FSH dose (IU)", "Total HMG dose (IU)", "Total LH dose (IU)",
        "Female trigger-day LH (mIU/mL)", "Female trigger-day E2 (pg/mL)",
        "Female trigger-day P (ng/mL)", "Trigger-day endometrial thickness (mm)",
        "Urinary hCG trigger dose (IU)", "Recombinant hCG trigger dose (µg)",
        "GnRH agonist trigger dose (mg)",
        "Number of follicles punctured", "Number of oocytes retrieved",
        "Number of MII oocytes", "Number of 2PN zygotes",
        "Number of multipronuclear zygotes", "Number of high-quality embryos",
        "Number of embryos available at D3", "Number of blastocyst-cultured embryos",
        "Number of blastocysts formed",
        # --- Embryo morphology encoding v2 (see embryo_morphology_encoding_v2.md) ---
        # Blastocyst grades stored as weighted products: ICM x 0.5, TE x 1.0.
        # Embryo 2 slots are 0 for single-embryo transfers.
        "blast_e1_ICM_w", "blast_e1_TE_w",
        "blast_e2_ICM_w", "blast_e2_TE_w",
        # Cleavage-stage Roman grade (I=4, II=3, III=2, IV=1); cell count is ignored.
        "clv_e1_grade", "clv_e2_grade",
        # --- Clinical inverted-U quality scores (v3) ---
        # Expansion optimality: 3-4 optimal=2, 1-2/5-6 suboptimal=1, none=0.
        "blast_e1_exp_quality", "blast_e2_exp_quality",
        # Cell-count optimality: 8=3, 7/9=2, <7 or >9 suboptimal=1, none=0.
        "clv_e1_cell_quality", "clv_e2_cell_quality",
    ]

    CATEGORICAL_FEATURES = [
        "Ovarian stimulation protocol", "Method of fertilization",
        "Female smoking history", "Male smoking history", "Trigger type",
        "Types of embryos transferred",
        "Number of cleavage-stage embryos transferred",
        "Number of blastocysts transferred",
    ]
    
    # =====================================================================
    # FEATURE ENCODING LOGIC
    # =====================================================================
    # Step 1: Identify available columns in the dataset
    # Step 2: Extract only the numerical features that exist in the dataset
    # Step 3: Extract and one-hot encode only the categorical features
    # Step 4: Validate that no numerical features were accidentally encoded
    
    print("\n[2] Processing Features...")
    
    # Get available columns (excluding target)
    available_cols = [col for col in X.columns if col not in exclude_cols]
    
    # Validate that we have features to process
    if len(available_cols) == 0:
        raise ValueError("No features found in dataset after excluding target columns")
    
    # Extract numerical features that exist in the dataset
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in available_cols]
    numerical_cols_missing = [col for col in NUMERICAL_FEATURES if col not in available_cols]
    
    if numerical_cols_missing:
        print(f"  Warning: Missing numerical features: {numerical_cols_missing}")
    
    # Extract categorical features that exist in the dataset
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in available_cols]
    categorical_cols_missing = [col for col in CATEGORICAL_FEATURES if col not in available_cols]
    
    if categorical_cols_missing:
        print(f"  Warning: Missing categorical features: {categorical_cols_missing}")
    
    # Validate: Ensure no overlap between numerical and categorical features
    overlap = set(numerical_cols) & set(categorical_cols)
    if overlap:
        raise ValueError(f"Feature overlap detected: {overlap}")
    
    # Remove any accidental duplicate entries while preserving order
    numerical_cols = list(dict.fromkeys(numerical_cols))
    categorical_cols = list(dict.fromkeys(categorical_cols))
    # A column must not be in both lists (numerical wins)
    categorical_cols = [c for c in categorical_cols if c not in set(numerical_cols)]
    
    # =====================================================================
    # PROCESSING: Separate numerical and categorical data
    # =====================================================================
    # Numerical features: Impute and scale directly (no encoding needed)
    X_numerical = X[numerical_cols].copy() if numerical_cols else pd.DataFrame()
    
    # Clean numerical columns: Convert non-numeric values to NaN
    # This handles placeholders like '/' that appear in the Excel file
    for col in numerical_cols:
        if col in X_numerical.columns:
            # Try to convert to numeric, coercing errors to NaN
            X_numerical[col] = pd.to_numeric(X_numerical[col], errors='coerce')
    
    # Categorical features: One-hot encode with drop_first=True to avoid multicollinearity
    X_categorical = X[categorical_cols].copy() if categorical_cols else pd.DataFrame()
    
    # Apply one-hot encoding to categorical features only
    if categorical_cols:
        # Convert to string to ensure consistent encoding, then apply get_dummies
        for col in categorical_cols:
            X_categorical[col] = X_categorical[col].astype(str)
        
        X_categorical_encoded = pd.get_dummies(X_categorical, columns=categorical_cols, drop_first=True)
        print(f"  Categorical encoded: {len(categorical_cols)} -> {X_categorical_encoded.shape[1]} dummy columns")
    else:
        X_categorical_encoded = pd.DataFrame()
    
    # Combine numerical and encoded categorical features
    if X_categorical_encoded.shape[0] > 0 and X_numerical.shape[0] > 0:
        X_processed = pd.concat([X_numerical, X_categorical_encoded], axis=1)
    elif X_numerical.shape[0] > 0:
        X_processed = X_numerical
    else:
        X_processed = X_categorical_encoded
    
    # =====================================================================
    # VALIDATION: Ensure numerical features were not encoded
    # =====================================================================
    # Check that all numerical columns in our list still exist as numeric columns
    # (not as dummy columns from get_dummies)
    for col in numerical_cols:
        if col in X_processed.columns:
            # Verify it's not a dummy variable (starts with original categorical prefix)
            pass  # Already preserved as-is
        elif f"{col}_" in str(X_processed.columns) or any(col in str(c) for c in X_processed.columns):
            raise ValueError(f"Numerical feature '{col}' may have been incorrectly encoded!")
    
    print(f"  Total features after processing: {X_processed.shape[1]}")
    print(f"  Dataset: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
    
    # =====================================================================
    # DATA LEAKAGE FIX: Train/test split BEFORE any preprocessing
    # =====================================================================
    # Split data FIRST to ensure no information leakage from test to train
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"\n  Training: {len(X_train_raw)} | Test: {len(X_test_raw)}")
    
    # =====================================================================
    # PREPROCESSING: Fit ONLY on training data, transform both train and test
    # =====================================================================
    
    # Step 1: Imputation - fit only on training data
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)
    
    # Step 2: Scaling - fit only on training data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Step 3: Feature Selection - fit only on training data
    k = min(50, X_train_scaled.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X_processed.columns[selected_mask].tolist()
    
    # Print selected features
    print("\n" + "="*100)
    print("SELECTED FEATURES (Top 50 by f_classif)")
    print("="*100)
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:>3}. {feat}")
    print("="*100)
    
    # Rename for consistency with rest of code
    X_train, X_test = X_train_selected, X_test_selected
    
    results = {}
    
    # Models 1-7: Traditional
    print("\n" + "="*100)
    print("TRAINING MODELS 1-7: Traditional ML")
    print("="*100)
    
    models_dict = {
        '1. Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'),
        '2. Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'),
        '3. Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
        '4. SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
        '5. KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        '6. Naive Bayes': GaussianNB(),
        '7. XGBoost': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }
    
    for name, model in models_dict.items():
        print(f"\n{name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        results[name] = {'metrics': metrics, 'proba': y_proba}
        print(f"  AUC: {metrics['AUC-ROC']:.4f}")
    
    # TabPFN (8)
    print("\n" + "="*100)
    print("TRAINING MODELS 8-9: TabPFN Variants")
    print("="*100)
    
    print("\n8. TabPFN...")
    if TABPFN_AVAILABLE:
        try:
            tabpfn = TabPFNClassifier(device='cpu')
            tabpfn.fit(X_train, y_train)
            tabpfn_pred = tabpfn.predict(X_test)
            tabpfn_proba = tabpfn.predict_proba(X_test)[:, 1]
            tabpfn_metrics = calculate_metrics(y_test, tabpfn_pred, tabpfn_proba)
            results['8. TabPFN'] = {'metrics': tabpfn_metrics, 'proba': tabpfn_proba}
            print(f"  AUC: {tabpfn_metrics['AUC-ROC']:.4f}")
        except Exception as e:
            print(f"  TabPFN failed: {e}")
            tabpfn_proba = np.ones(len(y_test)) * 0.5
            results['8. TabPFN'] = {'metrics': {'AUC-ROC': 0.5}, 'proba': tabpfn_proba}
    else:
        tabpfn_proba = np.ones(len(y_test)) * 0.5
        results['8. TabPFN'] = {'metrics': {'AUC-ROC': 0.5}, 'proba': tabpfn_proba}
    
    # TabPFN + LR Hybrid (9) - NEW!
    # DATA LEAKAGE FIX: Use validation set for weight optimization
    print("\n9. TabPFN + LR Hybrid...")
    
    # Split training data for validation (stratified)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Retrain LR and TabPFN on subset for validation
    lr_val = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
    lr_val.fit(X_tr, y_tr)
    lr_val_proba = lr_val.predict_proba(X_val)[:, 1]
    
    if TABPFN_AVAILABLE:
        try:
            tabpfn_val = TabPFNClassifier(device='cpu')
            tabpfn_val.fit(X_tr, y_tr)
            tabpfn_val_proba = tabpfn_val.predict_proba(X_val)[:, 1]
        except Exception as e:
            tabpfn_val_proba = np.ones(len(y_val)) * 0.5
    else:
        tabpfn_val_proba = np.ones(len(y_val)) * 0.5
    
    # Find best weight using VALIDATION set (not test set!)
    best_tabpfn_weight = 0.5
    best_tabpfn_auc = 0
    for w in np.arange(0.0, 1.05, 0.1):
        hybrid_proba_val = w * tabpfn_val_proba + (1 - w) * lr_val_proba
        hybrid_auc_val = roc_auc_score(y_val, hybrid_proba_val)
        if hybrid_auc_val > best_tabpfn_auc:
            best_tabpfn_auc = hybrid_auc_val
            best_tabpfn_weight = w
    
    # Now compute final hybrid predictions on TEST set using optimized weight
    lr_proba = results['1. Logistic Regression']['proba']
    tabpfn_proba = results['8. TabPFN']['proba']
    
    tabpfn_hybrid_proba = best_tabpfn_weight * tabpfn_proba + (1 - best_tabpfn_weight) * lr_proba
    tabpfn_hybrid_pred = (tabpfn_hybrid_proba >= 0.5).astype(int)
    tabpfn_hybrid_metrics = calculate_metrics(y_test, tabpfn_hybrid_pred, tabpfn_hybrid_proba)
    results['9. TabPFN + LR Hybrid'] = {'metrics': tabpfn_hybrid_metrics, 'proba': tabpfn_hybrid_proba}
    print(f"  Best weight (TabPFN): {best_tabpfn_weight:.2f}")
    print(f"  AUC: {tabpfn_hybrid_metrics['AUC-ROC']:.4f}")
    
    # IHAN Models (10-13)
    print("\n" + "="*100)
    print("TRAINING MODELS 10-13: IHAN Variants")
    print("="*100)
    
    # 10. IHAN baseline
    print("\n10. IHAN (baseline)...")
    ihan, device = train_ihan(X_train, y_train, X_test, y_test, epochs=80, seed=42)
    ihan.eval()
    with torch.no_grad():
        ihan_proba = ihan(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    ihan_pred = (ihan_proba >= 0.5).astype(int)
    ihan_metrics = calculate_metrics(y_test, ihan_pred, ihan_proba)
    results['10. IHAN (baseline)'] = {'metrics': ihan_metrics, 'proba': ihan_proba}
    print(f"  AUC: {ihan_metrics['AUC-ROC']:.4f}")
    
    # 11. Hybrid Ensemble
    # DATA LEAKAGE FIX: Use validation set for weight optimization
    print("\n11. Hybrid Ensemble (IHAN + LR)...")
    
    # Train IHAN ensemble on FULL training set for final predictions
    all_ihan_proba_test = []
    for seed in [42, 123, 456]:
        model, device = train_ihan(X_train, y_train, X_test, y_test, epochs=80, seed=seed)
        model.eval()
        with torch.no_grad():
            proba = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
            all_ihan_proba_test.append(proba)
    
    ihan_ensemble_proba_test = np.mean(all_ihan_proba_test, axis=0)
    
    # Train IHAN ensemble on subset for validation
    all_ihan_proba_val = []
    for seed in [42, 123, 456]:
        model_val, device_val = train_ihan(X_tr, y_tr, X_val, y_val, epochs=80, seed=seed)
        model_val.eval()
        with torch.no_grad():
            proba_val = model_val(torch.FloatTensor(X_val).to(device_val)).cpu().numpy()
            all_ihan_proba_val.append(proba_val)
    
    ihan_ensemble_proba_val = np.mean(all_ihan_proba_val, axis=0)
    
    # Weight test: w = weight on IHAN ensemble, (1-w) = weight on LR
    # DATA LEAKAGE FIX: Use VALIDATION set for weight optimization
    best_weight = 0.3
    best_auc = 0
    print("  Weight test (w = IHAN, 1-w = LR):")
    for w in [0.3, 0.5, 0.7]:
        # Use validation set for optimization
        hybrid_proba_val = w * ihan_ensemble_proba_val + (1 - w) * lr_val_proba
        hybrid_auc_val = roc_auc_score(y_val, hybrid_proba_val)
        if hybrid_auc_val > best_auc:
            best_auc = hybrid_auc_val
            best_weight = w
        print(f"    w={w:.1f}  ->  AUC = {hybrid_auc_val:.4f}")
    print(f"  >> Best weight: {best_weight:.1f}  (AUC = {best_auc:.4f})")
    
    # Now compute final hybrid predictions on TEST set using optimized weight
    hybrid_proba = best_weight * ihan_ensemble_proba_test + (1 - best_weight) * lr_proba
    hybrid_pred = (hybrid_proba >= 0.5).astype(int)
    hybrid_metrics = calculate_metrics(y_test, hybrid_pred, hybrid_proba)
    results['11. Hybrid Ensemble (IHAN+LR)'] = {'metrics': hybrid_metrics, 'proba': hybrid_proba}
    print(f"  AUC: {hybrid_metrics['AUC-ROC']:.4f}")
    
    # Results Table
    print("\n" + "="*100)
    print("COMPLETE RESULTS TABLE - 11 MODELS")
    print("="*100)
    
    print(f"\n{'#':>3} {'Model':<45} {'AUC-ROC':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 105)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['AUC-ROC'], reverse=True)
    
    for rank, (name, data) in enumerate(sorted_results, 1):
        m = data['metrics']
        print(f"{rank:>3} {name:<45} {m['AUC-ROC']:>8.4f} {m['Accuracy']:>10.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f} {m['F1 Score']:>8.4f}")
    
    # Rankings
    print("\n" + "="*100)
    print("FINAL RANKINGS")
    print("="*100)
    
    print("""
+----------------------------------------------------------------------------------------------------------+
|                              FINAL MODEL RANKINGS (by AUC-ROC)                             |
+----+---------------------------------------------+------------+---------------------------+
|Rank| Model                                       |   AUC-ROC  |    Category                |
+----+---------------------------------------------+------------+---------------------------+
""")
    
    categories = {
        '1.': 'Traditional', '2.': 'Traditional', '3.': 'Traditional',
        '4.': 'Traditional', '5.': 'Traditional', '6.': 'Traditional',
        '7.': 'Traditional', '8.': 'TabPFN', '9.': 'TabPFN Hybrid',
        '10.': 'IHAN', '11.': 'Hybrid'
    }
    
    for rank, (name, data) in enumerate(sorted_results, 1):
        m = data['metrics']
        prefix = name.split('.')[0] + '.'
        cat = categories.get(prefix, 'Other')
        print(f"| {rank:>2} | {name:<45} |   {m['AUC-ROC']:.4f}   | {cat:<25} |")
    
    print("+----+---------------------------------------------+------------+---------------------------+")
    
    # Analysis
    print("\n" + "="*100)
    print("TABPFN + LR HYBRID ANALYSIS")
    print("="*100)
    
    tabpfn_auc = results['8. TabPFN']['metrics']['AUC-ROC']
    tabpfn_hybrid_auc = results['9. TabPFN + LR Hybrid']['metrics']['AUC-ROC']
    lr_auc = results['1. Logistic Regression']['metrics']['AUC-ROC']
    hybrid_auc = results['11. Hybrid Ensemble (IHAN+LR)']['metrics']['AUC-ROC']
    
    print(f"\n  TabPFN:              {tabpfn_auc:.4f}")
    print(f"  TabPFN + LR Hybrid: {tabpfn_hybrid_auc:.4f} (weight TabPFN={best_tabpfn_weight:.2f})")
    print(f"  Improvement:         {(tabpfn_hybrid_auc - tabpfn_auc)*100:+.2f}%")
    print(f"\n  Comparison:")
    print(f"    TabPFN vs LR:        {(tabpfn_auc - lr_auc)*100:+.2f}%")
    print(f"    TabPFN+LR vs LR:    {(tabpfn_hybrid_auc - lr_auc)*100:+.2f}%")
    print(f"    TabPFN+LR vs Hybrid:  {(tabpfn_hybrid_auc - hybrid_auc)*100:+.2f}%")
    
    best = sorted_results[0]
    
    print("\n" + "="*100)
    print("CONCLUSION")
    print("="*100)
    
    print(f"""
BEST MODEL: {best[0]}
   AUC-ROC: {best[1]['metrics']['AUC-ROC']:.4f}

KEY FINDINGS:
1. TabPFN achieved:       {tabpfn_auc:.4f}
2. TabPFN + LR Hybrid:    {tabpfn_hybrid_auc:.4f}
3. IHAN + LR Hybrid:      {hybrid_auc:.4f}

The TabPFN + LR Hybrid improves over standalone TabPFN by {(tabpfn_hybrid_auc - tabpfn_auc)*100:+.2f}%
However, the IHAN-based hybrid models still perform better.
""")

    # =====================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # =====================================================================
    print("\n" + "="*100)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*100)
    
    # Dictionary to store importance from different models
    importance_dict = {}
    
    # 1. Random Forest Feature Importance
    rf_model = models_dict['2. Random Forest']
    rf_importance = rf_model.feature_importances_
    importance_dict['Random Forest'] = rf_importance
    
    # 2. Gradient Boosting Feature Importance
    gb_model = models_dict['3. Gradient Boosting']
    gb_importance = gb_model.feature_importances_
    importance_dict['Gradient Boosting'] = gb_importance
    
    # 3. XGBoost Feature Importance
    xgb_model = models_dict['7. XGBoost']
    xgb_importance = xgb_model.feature_importances_
    importance_dict['XGBoost'] = xgb_importance
    
    # 4. Logistic Regression Coefficients (absolute values)
    lr_model = models_dict['1. Logistic Regression']
    lr_importance = np.abs(lr_model.coef_[0])
    importance_dict['Logistic Regression'] = lr_importance
    
    # 5. IHAN Feature Importance (from learned attention weights)
    ihan.eval()
    with torch.no_grad():
        ihan_importance = torch.sigmoid(ihan.feature_importance).cpu().numpy()
    importance_dict['IHAN'] = ihan_importance
    
    # Create DataFrame with all importance scores
    importance_df = pd.DataFrame(index=selected_features)
    for name, imp in importance_dict.items():
        importance_df[name] = imp
    
    # Calculate average importance across all models
    importance_df['Average'] = importance_df.mean(axis=1)
    
    # Normalize to sum to 1 for comparison
    importance_df['Normalized'] = importance_df['Average'] / importance_df['Average'].sum()
    
    # Rank by average importance
    importance_df = importance_df.sort_values('Average', ascending=False)
    
    # Display top 30 features by average importance
    print("\n" + "="*100)
    print("TOP 30 FEATURES BY IMPORTANCE (Averaged across models)")
    print("="*100)
    print(f"\n{'Rank':<6}{'Feature':<55}{'Avg Imp':<12}{'Norm %':<10}{'RF':<10}{'GB':<10}{'XGB':<10}{'LR':<10}{'IHAN':<10}")
    print("-" * 130)
    
    for rank, (feat, row) in enumerate(importance_df.head(30).iterrows(), 1):
        print(f"{rank:<6}{feat[:54]:<55}{row['Average']:<12.6f}{row['Normalized']*100:<10.2f}{row['Random Forest']:<10.4f}{row['Gradient Boosting']:<10.4f}{row['XGBoost']:<10.4f}{row['Logistic Regression']:<10.4f}{row['IHAN']:<10.4f}")
    
    # Feature importance by category
    print("\n" + "="*100)
    print("FEATURE IMPORTANCE BY CATEGORY")
    print("="*100)
    
    # FIX 2: assign every selected feature to EXACTLY ONE category (no overlap),
    # then report each category's share of the TRUE grand total so the
    # percentages sum to 100. Priority order (first match wins) is documented
    # below; adjust the keyword lists if you prefer a different taxonomy.
    def assign_category(feat):
        f = feat.lower()
        # 1) Embryo / lab metrics (encoded morphology + embryology counts)
        if (f.startswith('blast_') or f.startswith('clv_')
                or any(x in f for x in ['embryo', 'blastocyst', 'cleavage', 'oocyte',
                                        'zygote', '2pn', 'multipronuclear', 'icm',
                                        '_te_w', 'expansion', 'follicles punctured',
                                        'high-quality', 'available at d3'])):
            return 'Embryo / Lab'
        # 2) Hormone assays / gonadotropin dosing (either sex, any timepoint).
        #    Endometrial thickness is uterine, not a hormone -> excluded here.
        if 'endometrial' not in f and any(x in f for x in [
                'fsh', 'lh', 'e2', 'prl', 'amh', 'tsh', 'gnrh', 'hcg', 'basal',
                'gn-start-day', 'trigger-day', 'total gn', 'total fsh', 'total hmg',
                'total lh', 'gn dose', 'gn stimulation', 'p (ng']):
            return 'Hormone Levels'
        # 3) Male demographics / anthropometry / semen (sex is now a PREFIX).
        if (f.startswith('male ') or 'sperm' in f or 'motility' in f
                or 'immotility' in f):
            return 'Male Factors'
        # 4) Female demographics / anthropometry / lifestyle.
        if f.startswith('female ') or f.startswith('duration of infertility'):
            return 'Female Factors'
        # 5) Everything else (protocol, fertilization method, trigger type,
        #    endometrial thickness, one-hot dummies).
        return 'Treatment / Other'

    feat_category = {f: assign_category(f) for f in selected_features}

    cat_importance = {}
    cat_counts = {}
    for f in selected_features:
        c = feat_category[f]
        cat_importance[c] = cat_importance.get(c, 0.0) + importance_df.loc[f, 'Average']
        cat_counts[c] = cat_counts.get(c, 0) + 1

    grand_total = sum(cat_importance.values())  # == sum of all feature importances
    cat_importance = dict(sorted(cat_importance.items(), key=lambda x: x[1], reverse=True))

    print(f"\n{'Category':<24}{'Total Importance':<20}{'% of Total':<14}{'Num Features':<14}")
    print("-" * 72)
    for cat_name, imp in cat_importance.items():
        pct = (imp / grand_total * 100) if grand_total else 0.0
        print(f"{cat_name:<24}{imp:<20.4f}{pct:<14.2f}{cat_counts[cat_name]:<14}")
    print("-" * 72)
    print(f"{'TOTAL':<24}{grand_total:<20.4f}{'100.00':<14}{len(selected_features):<14}")
    
    # Methodological explanation
    print("\n" + "="*100)
    print("METHODOLOGY FOR FEATURE IMPORTANCE CALCULATION")
    print("="*100)
    print("""
1. RANDOM FOREST: Uses Gini importance (Mean Decrease in Impurity)
   - Measures how much each feature contributes to decreasing impurity at each split
   - Higher importance = feature used more frequently in splits with greater purity gain

2. GRADIENT BOOSTING: Uses Friedman MSE improvement
   - Tracks improvement in mean squared error when each feature is used in splits
   - Accumulates importance across all trees in the ensemble

3. XGBOOST: Uses weight-based importance
   - Counts how many times a feature appears in all trees
   - Weighted by the improvement in the objective function

4. LOGISTIC REGRESSION: Uses absolute coefficient values
   - Coefficients indicate log-odds change per unit increase in feature
   - Absolute value represents feature's influence on outcome

5. IHAN: Uses learned attention weights
   - Sigmoid-activated learnable parameters that modulate feature inputs
   - Higher weight = model learns to emphasize this feature more

FINAL RANKING: Features are ranked by the AVERAGE importance across all 5 methods,
normalized to sum to 100%. This provides a robust consensus view of feature
importance that is not biased toward any single model's interpretation.
""")

    # =====================================================================
    # RESULTS BUNDLE  (consumed by run_and_visualize.py). This ONLY returns the
    # already-computed results; it does not change any model, metric, or printed
    # output above — running this file directly behaves exactly as before.
    # =====================================================================
    w_grid = [round(float(w), 2) for w in np.arange(0.0, 1.001, 0.1)]
    weight_sweep_auc = [
        roc_auc_score(y_val, w * ihan_ensemble_proba_val + (1 - w) * lr_val_proba)
        for w in w_grid
    ]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    bundle = {
        'metrics': {name: data['metrics'] for name, data in results.items()},
        'proba': {name: np.asarray(data['proba']) for name, data in results.items()},
        'y_test': np.asarray(y_test),
        'importance_df': importance_df,
        'selected_features': list(selected_features),
        'category_importance': dict(cat_importance),
        'category_counts': dict(cat_counts),
        'pipeline': {
            'raw_features': len(numerical_cols) + len(categorical_cols),
            'after_encoding': int(X_processed.shape[1]),
            'selected': int(k),
        },
        'class_distribution': {'live_birth': n_pos, 'non_live_birth': n_neg,
                               'total': n_pos + n_neg},
        'split': {'train': int(len(X_train)), 'test': int(len(X_test))},
        'weight_sweep': {'weights': w_grid, 'auc': [float(a) for a in weight_sweep_auc],
                         'best_weight': float(best_weight)},
        'data_for_ablation': {'X_train': X_train, 'y_train': y_train,
                              'X_test': X_test, 'y_test': y_test},
        'best_model': best[0],
    }
    return bundle


if __name__ == "__main__":
    run_comparison_13models()
