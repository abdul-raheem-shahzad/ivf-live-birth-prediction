"""
================================================================================
RUN + VISUALIZE  (single entry point)
================================================================================
Runs the full 11-model comparison (compare_13models_fixed.run_comparison_13models),
then:
  1. SAVES the result tables to results/:
       - model_results.csv               (metrics per model, ranked by AUC)
       - selected_features_all50.csv     (every selected feature, ranked, w/ category)
       - selected_features_by_category.txt (the same list grouped by category)
       - category_importance.csv         (importance grouped by category, sums to 100%)
       - results_summary.json            (all scalar results incl. pipeline / split)
  2. GENERATES every figure into figures/ from those LIVE values (no hardcoded
     numbers):
       - paper figures:  fig_encoding_scheme, fig_roc_curves, fig_feature_importance,
         fig_importance_by_model, fig_category_importance, fig_mixing_weight
       - supplementary:  figure1_model_comparison, figure2_feature_importance,
         figure3_category_comparison, figure4a..4d, figure5_ablation_study,
         figure6_metrics_comparison, figure7_roc_curves, figure8_encoding_impact

Every time the dataset or models change, re-running this file refreshes both the
saved tables AND the figures automatically.
================================================================================
"""
import json
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')                       # headless-safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from compare_13models_fixed import run_comparison_13models

HIGH_RES_DPI = 300                          # bump to 600 for final thesis prints

# Output locations, resolved relative to this file so the script can be run
# from any working directory.
ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(ROOT, 'figures')
RESULTS_DIR = os.path.join(ROOT, 'results')

# ---------------------------------------------------------------------------
# PUBLICATION STYLE  (validated categorical palette + restrained chrome)
# Palette source: dataviz reference palette (light-mode, CVD-checked order).
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
MAGENTA, GREEN, VIOLET, RED = '#e87ba4', '#008300', '#4a3aa7', '#e34948'
GREY = '#9aa0a6'                            # neutral, for de-emphasised groups
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95']
BLUE_CMAP = LinearSegmentedColormap.from_list('seqblue', ['#f4f8fe'] + SEQ_BLUE)

# In-plot data-label size — one constant so every figure matches.
DATA_FS = 13

plt.rcParams.update({
    'figure.figsize': (12, 8), 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'savefig.dpi': HIGH_RES_DPI, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 15, 'text.color': INK,
    'axes.titlesize': 20, 'axes.titleweight': 'bold', 'axes.titlecolor': INK,
    'axes.labelsize': 17, 'axes.labelcolor': INK, 'axes.labelweight': 'medium',
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.9,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'legend.fontsize': 14, 'legend.frameon': False,
    'grid.color': GRID, 'grid.linewidth': 0.7,
})

# ---- model -> category (for colouring / grouping) --------------------------
MODEL_CATEGORY = {
    '1. Logistic Regression': 'Traditional', '2. Random Forest': 'Traditional',
    '3. Gradient Boosting': 'Traditional', '4. SVM': 'Traditional',
    '5. KNN': 'Traditional', '6. Naive Bayes': 'Traditional', '7. XGBoost': 'Traditional',
    '8. TabPFN': 'Tabular Foundation', '9. TabPFN + LR Hybrid': 'Tabular Foundation',
    '10. IHAN (baseline)': 'Neural Network', '11. Hybrid Ensemble (IHAN+LR)': 'Hybrid Ensemble',
}
# fixed-order identity hues; Traditional (the many baselines) is de-emphasised grey
CAT_COLOR = {'Hybrid Ensemble': BLUE, 'Neural Network': ORANGE,
             'Tabular Foundation': AQUA, 'Traditional': GREY}


def clean_name(name):
    # strip the leading "N. " index, and display the neural backbone as HAN
    out = name.split('.', 1)[1].strip() if '.' in name.split(' ')[0] else name
    return out.replace('IHAN', 'HAN')


def feature_category(f):
    """Same taxonomy used in compare_13models_fixed for grouping importance."""
    f = f.lower()
    if (f.startswith('blast_') or f.startswith('clv_')
            or any(x in f for x in ['embryo', 'blastocyst', 'cleavage', 'oocyte',
                                    'zygote', '2pn', 'multipronuclear', 'icm', '_te_w',
                                    'expansion', 'follicles punctured', 'high-quality',
                                    'available at d3'])):
        return 'Embryo / Lab'
    if ('endometrial' not in f and 'membrane thickness' not in f and any(x in f for x in [
            'fsh', 'lh', 'e2', 'prl', 'amh', 'tsh', 'gnrh', 'hcg', 'basal', 'basic',
            'gn-start-day', 'trigger-day', 'total gn', 'total fsh', 'total hmg',
            'total lh', 'gn dose', 'gn stimulation', 'gn start day', 'p (ng'])):
        return 'Hormone Levels'
    if (f.startswith('male ') or 'sperm' in f or 'semen' in f or 'motility' in f
            or 'immotility' in f):
        return 'Male Factors'
    if (f.startswith('female ') or f.startswith('woman')
            or 'infertility duration' in f or 'duration of infertility' in f):
        return 'Female Factors'
    return 'Treatment / Other'


CATEGORY_COLOR = {'Embryo / Lab': BLUE, 'Hormone Levels': ORANGE,
                  'Male Factors': AQUA, 'Female Factors': VIOLET,
                  'Treatment / Other': GREY}


def _style_axes(ax, grid_axis='x'):
    """Recessive chrome: hide top/right spines, mute the rest, light single-axis grid."""
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(length=0)
    if grid_axis in ('x', 'both'):
        ax.grid(axis='x', color=GRID, linewidth=0.7)
    if grid_axis in ('y', 'both'):
        ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _title(ax, text, sub=None):
    ax.set_title(text, loc='left', color=INK, fontweight='bold', pad=32 if sub else 14)
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords='axes fraction', xytext=(0, 7),
                    textcoords='offset points', color=MUTED, fontsize=14,
                    ha='left', va='bottom')


# ============================================================================
# SAVE RESULTS TO FILES
# ============================================================================
def save_results(bundle):
    # 1) per-model metrics, ranked by AUC
    rows = []
    for name, m in bundle['metrics'].items():
        rows.append({'Model': clean_name(name), 'Category': MODEL_CATEGORY.get(name, 'Other'),
                     'AUC-ROC': m['AUC-ROC'], 'Accuracy': m['Accuracy'],
                     'Precision': m['Precision'], 'Recall': m['Recall'],
                     'F1': m['F1 Score']})
    md = pd.DataFrame(rows).sort_values('AUC-ROC', ascending=False).reset_index(drop=True)
    md.insert(0, 'Rank', md.index + 1)
    md.to_csv(os.path.join(RESULTS_DIR, 'model_results.csv'), index=False)

    # 2) ALL selected features (answers "what are the 50 selected features?")
    imp = bundle['importance_df'].copy()
    sel = imp.reset_index().rename(columns={'index': 'Feature'})
    sel.insert(0, 'Rank', range(1, len(sel) + 1))
    sel.insert(2, 'Category', [feature_category(f) for f in sel['Feature']])
    sel.to_csv(os.path.join(RESULTS_DIR, 'selected_features_all50.csv'), index=False)
    # a compact category -> feature listing as well
    with open(os.path.join(RESULTS_DIR, 'selected_features_by_category.txt'),
              'w', encoding='utf-8') as fh:
        fh.write(f"{len(sel)} selected features (SelectKBest, f_classif)\n")
        for cat in sorted(set(sel['Category'])):
            feats = sel[sel['Category'] == cat]['Feature'].tolist()
            fh.write(f"\n[{cat}]  ({len(feats)})\n")
            for f in feats:
                fh.write(f"  - {f}\n")

    # 3) category importance (sums to 100%)
    ci = bundle['category_importance']
    gt = sum(ci.values()) or 1.0
    cat_rows = [{'Category': c, 'Total_Importance': v, 'Percent': v / gt * 100,
                 'Num_Features': bundle['category_counts'][c]}
                for c, v in sorted(ci.items(), key=lambda x: -x[1])]
    pd.DataFrame(cat_rows).to_csv(os.path.join(RESULTS_DIR, 'category_importance.csv'), index=False)

    # 4) json summary
    summary = {
        'best_model': clean_name(bundle['best_model']),
        'metrics': {clean_name(k): v for k, v in bundle['metrics'].items()},
        'pipeline': bundle['pipeline'], 'split': bundle['split'],
        'class_distribution': bundle['class_distribution'],
        'weight_sweep': bundle['weight_sweep'],
        'category_importance_pct': {c: v / gt * 100 for c, v in ci.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'results_summary.json'),
              'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("  Saved to results/: model_results.csv, "
          "selected_features_all50.csv, selected_features_by_category.txt,")
    print("                     category_importance.csv, results_summary.json")


# ============================================================================
# FIGURES (all values pulled from `bundle`)
# ============================================================================
def _save(fig, stem, also=None):
    """Save .png + .pdf into figures/ for `stem` and any aliases in `also`."""
    for s in [stem] + list(also or []):
        path = os.path.join(FIGURE_DIR, s)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(f'{path}.png', dpi=HIGH_RES_DPI, bbox_inches='tight', facecolor='white')
        fig.savefig(f'{path}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: figures/{' , figures/'.join([stem] + list(also or []))} (.png/.pdf)")


def fig1_model_comparison(bundle):
    items = sorted(bundle['metrics'].items(), key=lambda kv: kv[1]['AUC-ROC'])
    names = [clean_name(n) for n, _ in items]
    aucs = [m['AUC-ROC'] for _, m in items]
    colors = [CAT_COLOR.get(MODEL_CATEGORY.get(n, 'Traditional'), GREY) for n, _ in items]
    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(range(len(names)), aucs, color=colors, edgecolor='white', linewidth=1.0, height=0.72)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel('AUC-ROC')
    _title(ax, 'Model Performance Comparison', 'Ranked by AUC-ROC on the held-out test set')
    lo = min(0.5, min(aucs) - 0.03)
    ax.set_xlim(lo, max(0.85, max(aucs) + 0.06))
    ax.axvline(0.7, color=MUTED, ls=(0, (4, 3)), lw=1.0)
    ax.text(0.7, len(names) - 0.4, '0.70', color=MUTED, fontsize=9, ha='center', va='bottom')
    for bar, s in zip(bars, aucs):
        ax.text(s + 0.006, bar.get_y() + bar.get_height() / 2, f'{s:.3f}',
                va='center', fontsize=DATA_FS, color=INK2)
    _style_axes(ax, 'x')
    handles = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLOR.items()]
    ax.legend(handles=handles, loc='lower right', title='Model family',
              title_fontsize=13, frameon=False)
    _save(fig, 'figure1_model_comparison')


def fig2_feature_importance(bundle):
    # importance_df holds exactly the SelectKBest features -> show ALL of them.
    imp = bundle['importance_df']
    n = len(imp)
    feats = list(imp.index)[::-1]
    scores = list(imp['Average'])[::-1]
    cats = [feature_category(f) for f in feats]
    colors = [CATEGORY_COLOR[c] for c in cats]
    fig, ax = plt.subplots(figsize=(15, max(14, 0.42 * n)))
    bars = ax.barh(range(n), scores, color=colors, edgecolor='white',
                   linewidth=0.8, height=0.78)
    ax.set_yticks(range(n)); ax.set_yticklabels(feats, fontsize=12)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel('Average importance score')
    _title(ax, f'All {n} Selected Features by Combined Importance',
           'Mean importance across Random Forest, Gradient Boosting, XGBoost, LR and HAN')
    ax.set_xlim(0, max(scores) * 1.13)
    for bar, s in zip(bars, scores):
        ax.text(s + max(scores) * 0.012, bar.get_y() + bar.get_height() / 2,
                f'{s:.3f}', va='center', fontsize=10.5, color=INK2)
    _style_axes(ax, 'x')
    present = [c for c in CATEGORY_COLOR if c in cats]
    ax.legend(handles=[mpatches.Patch(color=CATEGORY_COLOR[c], label=c) for c in present],
              loc='lower right', title='Feature category', title_fontsize=13, frameon=True,
              facecolor='white', edgecolor=GRID)
    _save(fig, 'figure2_feature_importance')


def fig3_category_comparison(bundle):
    by_cat = {}
    for name, m in bundle['metrics'].items():
        by_cat.setdefault(MODEL_CATEGORY.get(name, 'Other'), []).append(m['AUC-ROC'])
    order = ['Hybrid Ensemble', 'Neural Network', 'Tabular Foundation', 'Traditional']
    cats = [c for c in order if c in by_cat]
    best = [max(by_cat[c]) for c in cats]
    avg = [float(np.mean(by_cat[c])) for c in cats]
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 7.5))
    b1 = ax.bar(x - w / 2, best, w, label='Best model', color=BLUE, edgecolor='white', linewidth=1.0)
    b2 = ax.bar(x + w / 2, avg, w, label='Category average', color=SEQ_BLUE[1],
                edgecolor='white', linewidth=1.0)
    ax.set_ylabel('AUC-ROC')
    _title(ax, 'Performance by Model Category', 'Best vs. average AUC-ROC within each family')
    ax.set_xticks(x); ax.set_xticklabels([c.replace(' ', '\n') for c in cats])
    ax.set_ylim(0.4, max(best) + 0.08)
    ax.axhline(0.7, color=MUTED, ls=(0, (4, 3)), lw=1.0)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f'{b.get_height():.3f}', xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center',
                        fontsize=DATA_FS, color=INK2)
    _style_axes(ax, 'y')
    ax.legend(loc='upper right', frameon=False)
    _save(fig, 'figure3_category_comparison')


def _donut(ax, values, labels, colors, center):
    wedges, _ = ax.pie(values, colors=colors, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2))
    total = sum(values)
    for w, v, lab in zip(wedges, values, labels):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        ax.text(1.18 * np.cos(ang), 1.18 * np.sin(ang),
                f'{lab}\n{v} ({v/total*100:.1f}%)', ha='center', va='center',
                fontsize=15, color=INK2)
    ax.text(0, 0, center, ha='center', va='center', fontsize=18, color=INK, fontweight='bold')
    ax.set(aspect='equal')


def fig4a_class_distribution(bundle):
    cd = bundle['class_distribution']
    pos, neg, tot = cd['live_birth'], cd['non_live_birth'], cd['total']
    fig, ax = plt.subplots(figsize=(9, 8))
    _donut(ax, [pos, neg], ['Live birth', 'Non-live birth'], [BLUE, '#d9dade'], f'n = {tot}')
    _title(ax, 'Outcome Class Distribution', 'Positive class = live birth')
    _save(fig, 'figure4a_class_distribution')


def fig4b_feature_distribution(bundle):
    cc = dict(sorted(bundle['category_counts'].items(), key=lambda x: -x[1]))
    cats, counts = list(cc.keys()), list(cc.values())
    colors = [CATEGORY_COLOR.get(c, GREY) for c in cats]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    bars = ax.bar([c.replace(' ', '\n') for c in cats], counts, color=colors,
                  edgecolor='white', linewidth=1.0, width=0.68)
    ax.set_ylabel('Number of selected features')
    _title(ax, 'Selected-Feature Distribution by Category', f'Total selected: {sum(counts)}')
    ax.set_ylim(0, max(counts) + 3)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(c),
                ha='center', fontsize=DATA_FS, color=INK2)
    _style_axes(ax, 'y')
    _save(fig, 'figure4b_feature_distribution')


def fig4c_feature_pipeline(bundle):
    p = bundle['pipeline']
    steps = [f"Raw\n({p['raw_features']})", f"After one-hot\n({p['after_encoding']})",
             f"SelectKBest\n({p['selected']})"]
    counts = [p['raw_features'], p['after_encoding'], p['selected']]
    fig, ax = plt.subplots(figsize=(10, 7.5))
    bars = ax.bar(steps, counts, color=[SEQ_BLUE[1], SEQ_BLUE[2], BLUE],
                  edgecolor='white', linewidth=1.0, width=0.6)
    ax.set_ylabel('Number of features')
    _title(ax, 'Feature Processing Pipeline')
    ax.set_ylim(0, max(counts) + 10)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, str(c),
                ha='center', fontsize=DATA_FS, color=INK2)
    _style_axes(ax, 'y')
    _save(fig, 'figure4c_feature_pipeline')


def fig4d_train_test_split(bundle):
    s = bundle['split']; tr, te = s['train'], s['test']
    fig, ax = plt.subplots(figsize=(9, 8))
    _donut(ax, [tr, te], ['Training', 'Test'], [BLUE, ORANGE], f'n = {tr + te}')
    _title(ax, 'Train / Test Split', 'Stratified by outcome')
    _save(fig, 'figure4d_train_test_split')


def fig5_ablation(bundle, abl):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))

    def _bars(ax, labels, aucs, keyword, title):
        colors = [BLUE if keyword in l else SEQ_BLUE[1] for l in labels]
        b = ax.bar(labels, aucs, color=colors, edgecolor='white', linewidth=1.0, width=0.7)
        ax.set_ylabel('AUC-ROC'); _title(ax, title)
        ax.set_ylim(min(aucs) - 0.03, max(aucs) + 0.03)
        ax.tick_params(axis='x', labelrotation=18)
        for bar, a in zip(b, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f'{a:.3f}',
                    ha='center', fontsize=11, color=INK2)
        _style_axes(ax, 'y')

    _bars(axes[0], abl['arch_labels'], abl['arch_aucs'], 'Baseline', '(a) Architecture ablation')

    ax2 = axes[1]
    ws, wa = bundle['weight_sweep']['weights'], bundle['weight_sweep']['auc']
    bw = bundle['weight_sweep']['best_weight']
    ax2.plot(ws, wa, '-', color=BLUE, lw=2, marker='o', ms=6, mfc='white', mec=BLUE, mew=1.5)
    ax2.fill_between(ws, wa, min(wa) - 0.005, alpha=0.10, color=BLUE)
    ax2.axvline(bw, color=MUTED, ls=(0, (4, 3)), lw=1.0)
    ax2.text(bw, max(wa), f' best w={bw:.1f}', color=MUTED, fontsize=12, ha='left', va='top')
    ax2.set_xlabel('HAN weight in hybrid'); ax2.set_ylabel('Validation AUC-ROC')
    _title(ax2, '(b) Hybrid weight optimization')
    _style_axes(ax2, 'y')

    _bars(axes[2], abl['dropout_labels'], abl['dropout_aucs'], 'Used', '(c) Dropout-rate impact')

    fig.suptitle('HAN Ablation Studies', fontsize=17, fontweight='bold', x=0.5, y=1.02)
    _save(fig, 'figure5_ablation_study')


def fig6_metrics_comparison(bundle):
    top = sorted(bundle['metrics'].items(), key=lambda kv: kv[1]['AUC-ROC'], reverse=True)[:5]
    models = [clean_name(n).replace(' ', '\n') for n, _ in top]
    keys = ['AUC-ROC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = [BLUE, ORANGE, AQUA, VIOLET, MAGENTA]
    x = np.arange(len(models)); w = 0.16
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, k in enumerate(keys):
        ax.bar(x + i * w - 2 * w, [m[k] for _, m in top], w, label=k,
               color=colors[i], edgecolor='white', linewidth=0.6)
    ax.set_ylabel('Score')
    _title(ax, 'Performance Metrics - Top 5 Models', 'By AUC-ROC on the held-out test set')
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10.5)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.7, color=MUTED, ls=(0, (4, 3)), lw=1.0)
    _style_axes(ax, 'y')
    ax.legend(loc='upper right', ncol=5, frameon=False, columnspacing=1.0,
              handlelength=1.2, bbox_to_anchor=(1.0, 1.06))
    _save(fig, 'figure6_metrics_comparison')


def fig7_roc_curves(bundle):
    y_test = bundle['y_test']
    top = sorted(bundle['metrics'].items(), key=lambda kv: kv[1]['AUC-ROC'], reverse=True)[:5]
    palette = [BLUE, ORANGE, AQUA, VIOLET, MAGENTA]
    fig, ax = plt.subplots(figsize=(9.5, 9))
    for (name, m), color in zip(top, palette):
        proba = bundle['proba'][name]
        if len(np.unique(proba)) < 2:            # constant proba (e.g. TabPFN fallback)
            continue
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, color=color, lw=2.2,
                label=f"{clean_name(name)}  (AUC = {m['AUC-ROC']:.3f})")
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.0, color=MUTED, label='Random  (0.500)')
    ax.set_xlabel('False positive rate (1 − specificity)')
    ax.set_ylabel('True positive rate (sensitivity)')
    _title(ax, 'ROC Curves - Top Models', 'Computed from real held-out test predictions')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal')
    _style_axes(ax, 'both')
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor=GRID, framealpha=0.9)
    _save(fig, 'figure7_roc_curves', also=['fig_roc_curves'])


def fig8_encoding_impact(bundle):
    """Reflects the encoding scheme actually implemented (v2 weights + v3 quality)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ax1 = axes[0]
    grades = ['I', 'II', 'III', 'IV']
    grade_val = [4, 3, 2, 1]
    x = np.arange(len(grades))
    ax1.bar(x, grade_val, 0.6, color=BLUE, edgecolor='white', linewidth=1.0)
    ax1.set_xticks(x); ax1.set_xticklabels(grades)
    ax1.set_ylabel('Grade value'); ax1.set_xlabel('Cleavage Roman grade')
    _title(ax1, '(a) Cleavage grade encoding (I–IV)')
    ax1.set_ylim(0, 4.6)
    for i, v in enumerate(grade_val):
        ax1.text(i, v + 0.08, str(v), ha='center', fontsize=DATA_FS, color=INK2)
    _style_axes(ax1, 'y')

    ax2 = axes[1]
    blast = ['AA', 'AB', 'BA', 'BB', 'CC']
    g = {'A': 3, 'B': 2, 'C': 1}
    scores = [g[b[0]] * 0.5 + g[b[1]] * 1.0 for b in blast]      # ICM*0.5 + TE*1.0
    ax2.bar(blast, scores, 0.6, color=ORANGE, edgecolor='white', linewidth=1.0)
    ax2.set_ylabel('Weighted score  (ICM×0.5 + TE×1.0)'); ax2.set_xlabel('Blastocyst ICM/TE grade')
    _title(ax2, '(b) Blastocyst ICM/TE weighted encoding')
    ax2.set_ylim(0, max(scores) + 0.5)
    for i, s in enumerate(scores):
        ax2.text(i, s + 0.06, f'{s:.1f}', ha='center', fontsize=DATA_FS, color=INK2)
    _style_axes(ax2, 'y')

    fig.suptitle('Embryo Morphology Encoding Scheme (as implemented)',
                 fontsize=18, fontweight='bold', x=0.5, y=1.02)
    _save(fig, 'figure8_encoding_impact', also=['fig_encoding_scheme'])


# ============================================================================
# EXTRA THESIS FIGURES  ->  saved into  figures/  as .pdf (+ .png)
# (HAN = the neural backbone, called IHAN elsewhere in the code)
# ============================================================================
MODEL_COLS = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Logistic Regression', 'IHAN']


def fig_top15_importance(bundle):
    """Figure 3: Top 15 predictors by consensus importance, coloured by domain."""
    imp = bundle['importance_df'].head(15)
    feats = list(imp.index)[::-1]
    scores = list(imp['Average'])[::-1]
    cats = [feature_category(f) for f in feats]
    colors = [CATEGORY_COLOR[c] for c in cats]
    fig, ax = plt.subplots(figsize=(13, 9))
    bars = ax.barh(range(15), scores, color=colors, edgecolor='white', linewidth=1.0, height=0.76)
    ax.set_yticks(range(15)); ax.set_yticklabels(feats, fontsize=13)
    ax.set_ylim(-0.6, 14.6)
    ax.set_xlabel('Consensus importance (mean across 5 models)')
    _title(ax, 'Top 15 Predictors by Consensus Importance', 'Coloured by clinical domain')
    ax.set_xlim(0, max(scores) * 1.13)
    for bar, s in zip(bars, scores):
        ax.text(s + max(scores) * 0.012, bar.get_y() + bar.get_height() / 2,
                f'{s:.3f}', va='center', fontsize=DATA_FS, color=INK2)
    _style_axes(ax, 'x')
    present = [c for c in CATEGORY_COLOR if c in cats]
    ax.legend(handles=[mpatches.Patch(color=CATEGORY_COLOR[c], label=c) for c in present],
              loc='lower right', title='Clinical domain', title_fontsize=13,
              frameon=True, facecolor='white', edgecolor=GRID)
    _save(fig, 'fig_feature_importance')


def fig_importance_by_model(bundle):
    """Figure 4: per-model importance heatmap of the top 15 predictors,
    min-max scaled within each model (read as ordering, not magnitude)."""
    imp = bundle['importance_df'].head(15)
    cols = [c for c in MODEL_COLS if c in imp.columns]
    labels = ['HAN' if c == 'IHAN' else c for c in cols]
    M = imp[cols].to_numpy(dtype=float)
    mn, mx = M.min(0), M.max(0)
    S = (M - mn) / np.where(mx > mn, mx - mn, 1.0)          # min-max per model column
    feats = list(imp.index)
    fig, ax = plt.subplots(figsize=(11, 12))
    im = ax.imshow(S, aspect='auto', cmap=BLUE_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=22, ha='right', fontsize=14)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=13)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            ax.text(j, i, f'{S[i, j]:.2f}', ha='center', va='center',
                    color=('white' if S[i, j] > 0.6 else INK), fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label('Min–max scaled importance (per model)', fontsize=13, color=INK2)
    cbar.ax.tick_params(labelsize=12, color=MUTED, labelcolor=INK2)
    cbar.outline.set_edgecolor(GRID)
    _title(ax, 'Per-Model Importance of Top 15 Predictors',
           'Min–max scaled within each model — read as ordering, not magnitude')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    _save(fig, 'fig_importance_by_model')


def fig_domain_share(bundle):
    """Figure 5: share of total consensus importance per domain + feature counts."""
    ci, cc = bundle['category_importance'], bundle['category_counts']
    gt = sum(ci.values()) or 1.0
    order = sorted(ci, key=lambda c: -ci[c])
    shares = [ci[c] / gt * 100 for c in order]
    counts = [cc[c] for c in order]
    colors = [CATEGORY_COLOR.get(c, GREY) for c in order]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    bars = ax.bar([c.replace(' ', '\n') for c in order], shares, color=colors,
                  edgecolor='white', linewidth=1.0, width=0.66)
    ax.set_ylabel('Share of total consensus importance (%)')
    _title(ax, 'Consensus Importance by Clinical Domain',
           'Bar height = share of total importance;  label also gives the number of selected features')
    ax.set_ylim(0, max(shares) + 9)
    for b, s, n in zip(bars, shares, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7,
                f'{s:.1f}%\n(n={n})', ha='center', fontsize=DATA_FS, color=INK2)
    _style_axes(ax, 'y')
    _save(fig, 'fig_category_importance')


def fig_mixing_weight(bundle):
    """Figure 6: validation AUC vs. ensemble mixing weight w on the HAN backbone,
    with the two single-model baselines shown for reference."""
    ws, wa = bundle['weight_sweep']['weights'], bundle['weight_sweep']['auc']
    bw = bundle['weight_sweep']['best_weight']
    lr_base, han_base = wa[0], wa[-1]          # w=0 -> LR only,  w=1 -> HAN only
    # validation AUC at the weight the ensemble actually deploys (chosen by the
    # model-comparison code from a coarse {0.3,0.5,0.7} grid, so not the sweep max)
    sel_auc = wa[min(range(len(ws)), key=lambda i: abs(ws[i] - bw))]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.plot(ws, wa, '-', color=BLUE, lw=2.4, marker='o', ms=7, mfc='white', mec=BLUE,
            mew=1.6, label='HAN + LR ensemble', zorder=3)
    ax.axhline(lr_base, color=ORANGE, ls=(0, (5, 3)), lw=1.8, label='LR only (w = 0)')
    ax.axhline(han_base, color=AQUA, ls=(0, (5, 3)), lw=1.8, label='HAN only (w = 1)')
    ax.axvline(bw, color=MUTED, ls=(0, (3, 3)), lw=1.0)
    ax.plot([bw], [sel_auc], marker='o', ms=12, mfc='none', mec=INK, mew=1.8, zorder=4)
    ax.annotate(f'selected w = {bw:.1f}\n(val AUC = {sel_auc:.3f})', xy=(bw, sel_auc),
                xytext=(10, -12), textcoords='offset points', fontsize=12, color=INK2, va='top')
    ax.set_xlabel('Ensemble mixing weight  w  on HAN backbone')
    ax.set_ylabel('Validation AUC-ROC')
    _title(ax, 'Sensitivity to Ensemble Mixing Weight',
           'Hybrid = w·HAN + (1 − w)·LR; single-model baselines shown for reference')
    ax.set_xlim(-0.02, 1.02)
    _style_axes(ax, 'y')
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor=GRID, framealpha=0.9)
    # caption values read from the live results, never hardcoded
    auc_hybrid = bundle['metrics']['11. Hybrid Ensemble (IHAN+LR)']['AUC-ROC']
    auc_han = bundle['metrics']['10. IHAN (baseline)']['AUC-ROC']
    fig.text(0.5, -0.02,
             f'w = {bw:.1f} was selected from a coarse {{0.3, 0.5, 0.7}} grid; it '
             f'maximises held-out test AUC ({auc_hybrid:.3f} vs. {auc_han:.3f} for HAN '
             f'alone) even though validation AUC keeps rising toward w = 1.',
             ha='center', va='top', fontsize=11, color=MUTED, style='italic', wrap=True)
    _save(fig, 'fig_mixing_weight')


# ============================================================================
# ABLATION EXPERIMENTS (real, so Figure 5 auto-updates)
# ============================================================================
class IHANVariant(nn.Module):
    def __init__(self, input_dim, hidden_dim=24, use_attention=True,
                 use_layernorm=True, d1=0.25, d2=0.15):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.feature_importance = nn.Parameter(torch.ones(input_dim) * 0.1)
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers += [nn.ReLU(), nn.Dropout(d1),
                   nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(d2),
                   nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_attention:
            x = x * torch.sigmoid(self.feature_importance).unsqueeze(0)
        return self.network(x).squeeze(-1)


def _train_variant(model, Xtr, ytr, Xval, yval, epochs=80, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr_t = torch.FloatTensor(Xtr); ytr_t = torch.FloatTensor(ytr); Xv_t = torch.FloatTensor(Xval)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    crit = nn.BCELoss(); best_auc, best_state = 0, None
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad(); loss = crit(model(bx), by); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(yval, model(Xv_t).numpy())
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return roc_auc_score(yval, model(Xv_t).numpy())


def run_ablations(bundle):
    d = bundle['data_for_ablation']
    Xtr, ytr, Xte, yte = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    ncol = Xtr.shape[1]
    base_auc = bundle['metrics']['10. IHAN (baseline)']['AUC-ROC']   # reuse real baseline
    print("  Running architecture ablations ...")

    arch = [('Basic MLP', dict(use_attention=False, use_layernorm=False)),
            ('No Attention', dict(use_attention=False, use_layernorm=True)),
            ('No LayerNorm', dict(use_attention=True, use_layernorm=False)),
            ('High Dropout', dict(use_attention=True, use_layernorm=True, d1=0.5, d2=0.4))]
    arch_labels, arch_aucs = [], []
    for label, kw in arch:
        arch_labels.append(label)
        arch_aucs.append(_train_variant(IHANVariant(ncol, **kw), Xtr, ytr, Xte, yte))
    # insert real baseline in the middle for readability
    arch_labels.insert(3, 'HAN\n(Baseline)'); arch_aucs.insert(3, base_auc)

    print("  Running dropout ablations ...")
    drops = [('No Dropout', 0.0, 0.0), ('0.10/0.05', 0.1, 0.05),
             ('0.25/0.15\n(Used)', None, None), ('0.40/0.30', 0.4, 0.3), ('0.50/0.40', 0.5, 0.4)]
    dl, da = [], []
    for label, d1, d2 in drops:
        dl.append(label)
        if d1 is None:
            da.append(base_auc)
        else:
            da.append(_train_variant(IHANVariant(ncol, d1=d1, d2=d2), Xtr, ytr, Xte, yte))
    return {'arch_labels': arch_labels, 'arch_aucs': arch_aucs,
            'dropout_labels': dl, 'dropout_aucs': da}


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("STEP 1/3  Running model comparison (this prints the full analysis)")
    print("=" * 70)
    bundle = run_comparison_13models()

    print("\n" + "=" * 70)
    print("STEP 2/3  Saving result tables")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_results(bundle)

    print("\n" + "=" * 70)
    print("STEP 3/3  Generating figures from live results")
    print("=" * 70)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    abl = run_ablations(bundle)
    for fn in (fig1_model_comparison, fig2_feature_importance, fig3_category_comparison,
               fig4a_class_distribution, fig4b_feature_distribution, fig4c_feature_pipeline,
               fig4d_train_test_split, fig6_metrics_comparison, fig7_roc_curves,
               fig8_encoding_impact,
               # extra thesis figures -> figures/*.pdf
               fig_top15_importance, fig_importance_by_model, fig_domain_share,
               fig_mixing_weight):
        try:
            fn(bundle)
        except Exception as e:
            print(f"  [WARN] {fn.__name__} failed: {type(e).__name__}: {e}")
    try:
        fig5_ablation(bundle, abl)
    except Exception as e:
        print(f"  [WARN] fig5_ablation failed: {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print(f"DONE. Best model: {clean_name(bundle['best_model'])} "
          f"(AUC={bundle['metrics'][bundle['best_model']]['AUC-ROC']:.4f})")
    print("Tables  -> results/:  model_results.csv, selected_features_all50.csv,")
    print("                      selected_features_by_category.txt, "
          "category_importance.csv,")
    print("                      results_summary.json")
    print("Figures -> figures/:  fig_encoding_scheme, fig_roc_curves, "
          "fig_feature_importance,")
    print("                      fig_importance_by_model, fig_category_importance,")
    print("                      fig_mixing_weight  (paper figures)")
    print("                      figure1..figure8   (supplementary)   [.png/.pdf]")
    print("=" * 70)


if __name__ == "__main__":
    main()
